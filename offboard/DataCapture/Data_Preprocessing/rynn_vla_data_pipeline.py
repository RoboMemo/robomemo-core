"""
RynnVLA-001 Data Curation Pipeline  v3.1
-----------------------------------------
步骤 1: 低频抽帧
步骤 2: 姿态过滤（剔除正脸/背影）+ 三层标注 → 输出视频
         - 层 A: YOLOv8-Pose  手臂骨架（肩→肘→手腕）
         - 层 B: MediaPipe     十指骨架（21关键点，置信度可调）
         - 层 C: YOLOv8-World  工具目标框（螺丝刀等）
步骤 3: 动作标注（已禁用，保留注释供后续启用）

依赖安装:
    pip install opencv-python-headless tqdm ultralytics mediapipe
模型文件（首次运行自动下载，或手动放置）:
    hand_landmarker.task  →  与脚本同目录，或通过 --hand_model 指定路径

修复记录 v3.1:
    - 移除重复的 cv2 导入（原第26行）
    - 移除未使用的 os、json 导入
    - _is_valid_frame 返回的 confs 统一转为 numpy，避免 Tensor 比较歧义
    - filter_frames_to_video 中丢弃统计的 confs 比较改用 float() 显式转换
    - _draw_hand_skeleton 不再重复 cv2.imread，直接使用传入的 frame 做 RGB 转换
    - 修复 f-string 装饰线改为普通字符串，消除 pyflakes 警告
"""

import cv2
import argparse
import urllib.request
from pathlib import Path
from tqdm import tqdm

# ── YOLOv8-Pose + YOLOv8-World ───────────────────────────────────────────────
try:
    from ultralytics import YOLO, YOLOWorld
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False
    print("Warning: ultralytics not installed. Run: pip install ultralytics")

# ── MediaPipe HandLandmarker (Tasks API, mediapipe >= 0.10) ───────────────────
try:
    import mediapipe as mp
    from mediapipe.tasks.python import vision as mp_vision
    from mediapipe.tasks.python import BaseOptions as mp_BaseOptions
    HAS_MEDIAPIPE = True
except ImportError:
    HAS_MEDIAPIPE = False
    print("Warning: mediapipe not installed. Run: pip install mediapipe")

# ── 步骤 3 动作标注（已禁用） ─────────────────────────────────────────────────
# try:
#     from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
#     from qwen_vl_utils import process_vision_info
#     import torch
#     HAS_QWEN = True
# except ImportError:
#     HAS_QWEN = False
HAS_QWEN = False  # 步骤 3 已禁用，强制设为 False


# ─────────────────────────────────────────────────────────────────────────────
# 常量：YOLOv8-Pose COCO 17关键点索引
#   0:鼻子  1:左眼  2:右眼  3:左耳  4:右耳
#   5:左肩  6:右肩  7:左肘  8:右肘
#   9:左手腕  10:右手腕
#   11:左髋  12:右髋  13:左膝  14:右膝  15:左踝  16:右踝
BODY_SKELETON_PAIRS = [
    (5, 7), (7, 9),    # 左肩→左肘→左手腕
    (6, 8), (8, 10),   # 右肩→右肘→右手腕
]

# MediaPipe 手部21关键点骨架连接对
# 0:手腕  1-4:拇指  5-8:食指  9-12:中指  13-16:无名指  17-20:小指
HAND_SKELETON_PAIRS = [
    # 手掌横向连接
    (0, 1), (1, 5), (5, 9), (9, 13), (13, 17), (17, 0),
    # 拇指
    (1, 2), (2, 3), (3, 4),
    # 食指
    (5, 6), (6, 7), (7, 8),
    # 中指
    (9, 10), (10, 11), (11, 12),
    # 无名指
    (13, 14), (14, 15), (15, 16),
    # 小指
    (17, 18), (18, 19), (19, 20),
]

# 各手指指尖关键点索引（用于绘制大圆点）
FINGERTIP_INDICES = {4, 8, 12, 16, 20}

# 颜色配置（BGR）
COLOR_BODY_BONE = (0, 200, 255)   # 手臂骨架连线：黄色
COLOR_BODY_KP   = (0, 255, 0)     # 手臂关键点：绿色
COLOR_HAND_BONE = (255, 100, 0)   # 手指骨架连线：蓝色
COLOR_HAND_KP   = (255, 200, 0)   # 手指关键点：浅蓝色
COLOR_FINGERTIP = (0, 255, 255)   # 指尖：青色
COLOR_TOOL_BOX  = (0, 0, 255)     # 工具目标框：红色

# MediaPipe HandLandmarker 模型下载地址
HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
# ─────────────────────────────────────────────────────────────────────────────


def _ensure_hand_model(model_path: str) -> str:
    """确保 MediaPipe hand_landmarker.task 模型文件存在，不存在则自动下载"""
    p = Path(model_path)
    if not p.exists():
        print(f"  Downloading MediaPipe hand_landmarker model to {p} ...")
        urllib.request.urlretrieve(HAND_MODEL_URL, str(p))
        print("  Download complete.")
    return str(p)


class RynnVLADataPipeline:
    def __init__(
        self,
        input_dir,
        output_dir,
        fps=2,
        # ── 姿态过滤阈值 ──────────────────────────────────────────────────
        shoulder_thresh=0.35,
        wrist_thresh=0.3,
        face_thresh=0.5,
        # ── 十指骨架阈值（可调） ──────────────────────────────────────────
        hand_detect_conf=0.5,    # MediaPipe 手部检测置信度
        hand_presence_conf=0.5,  # MediaPipe 手部存在置信度
        hand_track_conf=0.5,     # MediaPipe 手部追踪置信度
        hand_model="hand_landmarker.task",
        # ── 工具检测 ──────────────────────────────────────────────────────
        tool_classes=None,
        tool_conf=0.25,
    ):
        """
        参数说明:
            fps                : 抽帧帧率（默认 2 FPS）
            shoulder_thresh    : 肩膀关键点置信度阈值（背影判断，默认 0.35）
            wrist_thresh       : 手腕关键点置信度阈值（默认 0.3）
            face_thresh        : 面部关键点置信度阈值（默认 0.5）
            hand_detect_conf   : MediaPipe 手部检测置信度（默认 0.5，越高越严格）
            hand_presence_conf : MediaPipe 手部存在置信度（默认 0.5）
            hand_track_conf    : MediaPipe 手部追踪置信度（默认 0.5）
            hand_model         : hand_landmarker.task 文件路径（不存在时自动下载）
            tool_classes       : YOLOv8-World 检测的工具类别列表
            tool_conf          : 工具检测置信度阈值（默认 0.25）
        """
        self.input_dir  = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.fps        = fps

        self.shoulder_thresh    = shoulder_thresh
        self.wrist_thresh       = wrist_thresh
        self.face_thresh        = face_thresh
        self.hand_detect_conf   = hand_detect_conf
        self.hand_presence_conf = hand_presence_conf
        self.hand_track_conf    = hand_track_conf
        self.tool_conf          = tool_conf
        self.tool_classes       = tool_classes or [
            "screwdriver", "drill", "electric drill",
            "wrench", "hammer", "pliers", "cutter"
        ]

        self.frames_dir   = self.output_dir / "frames"
        self.filtered_dir = self.output_dir / "filtered_videos"
        # self.dataset_dir  = self.output_dir / "rynn_dataset"  # 步骤3暂时禁用

        for d in [self.frames_dir, self.filtered_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # ── 加载 YOLOv8-Pose ─────────────────────────────────────────────
        if HAS_YOLO:
            print("Loading YOLOv8-Pose model (yolov8n-pose.pt)...")
            self.pose_model = YOLO('yolov8n-pose.pt')

            print("Loading YOLOv8-World model (yolov8s-world.pt)...")
            self.world_model = YOLOWorld('yolov8s-world.pt')
            self.world_model.set_classes(self.tool_classes)
            print(f"  Tool detection classes: {self.tool_classes}")

        # ── 加载 MediaPipe HandLandmarker ────────────────────────────────
        if HAS_MEDIAPIPE:
            model_path = _ensure_hand_model(hand_model)
            print(
                f"Loading MediaPipe HandLandmarker "
                f"(detect={hand_detect_conf}, "
                f"presence={hand_presence_conf}, "
                f"track={hand_track_conf})..."
            )
            options = mp_vision.HandLandmarkerOptions(
                base_options=mp_BaseOptions(model_asset_path=model_path),
                running_mode=mp_vision.RunningMode.IMAGE,
                num_hands=2,
                min_hand_detection_confidence=hand_detect_conf,
                min_hand_presence_confidence=hand_presence_conf,
                min_tracking_confidence=hand_track_conf,
            )
            self.hand_detector = mp_vision.HandLandmarker.create_from_options(options)
            print("  MediaPipe HandLandmarker loaded.")

        # ── 步骤 3 模型加载已禁用 ─────────────────────────────────────────
        # if HAS_QWEN:
        #     print("Loading Qwen2-VL-7B model...")
        #     self.qwen_model = Qwen2VLForConditionalGeneration.from_pretrained(
        #         "Qwen/Qwen2-VL-7B-Instruct", torch_dtype=torch.bfloat16, device_map="auto")
        #     self.qwen_processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

    # ─────────────────────────────────────────────────────────────────────────
    # 步骤 1：低频抽帧
    # ─────────────────────────────────────────────────────────────────────────
    def extract_frames(self, video_path):
        """以 self.fps 的帧率从原始视频中提取关键帧，保存为 JPG"""
        print(f"\n[Step 1] Extracting frames from '{video_path.name}' at {self.fps} FPS...")
        video_name = video_path.stem
        out_folder = self.frames_dir / video_name
        out_folder.mkdir(exist_ok=True)

        cap = cv2.VideoCapture(str(video_path))
        original_fps   = cap.get(cv2.CAP_PROP_FPS) or 30
        frame_interval = max(1, int(original_fps / self.fps))

        count, saved = 0, 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if count % frame_interval == 0:
                cv2.imwrite(str(out_folder / f"frame_{saved:06d}.jpg"), frame)
                saved += 1
            count += 1
        cap.release()
        print(f"  Extracted {saved} frames -> {out_folder}")
        return out_folder

    # ─────────────────────────────────────────────────────────────────────────
    # 步骤 2 辅助：帧有效性判断（YOLOv8-Pose）
    # ─────────────────────────────────────────────────────────────────────────
    def _is_valid_frame(self, pose_results):
        """
        判断是否为有效的第一视角操作帧，返回 (is_valid: bool, confs: np.ndarray | None)。
        confs 统一转为 numpy 数组，避免 Tensor 比较歧义。

        丢弃条件（任意一条触发即丢弃）:
          1. 正脸：鼻子或任意眼睛置信度 > face_thresh
          2. 背影：肩膀置信度 > shoulder_thresh 且无正脸
        保留条件（必须满足）:
          3. 手腕置信度 > wrist_thresh
        """
        for result in pose_results:
            if result.keypoints is None or len(result.keypoints.xy) == 0:
                continue
            if result.keypoints.conf is None:
                continue

            # 统一转为 numpy，避免 Tensor.__bool__ 的潜在问题
            confs = result.keypoints.conf[0].cpu().numpy()
            if len(confs) < 11:
                continue

            face_visible = (
                float(confs[0]) > self.face_thresh or
                float(confs[1]) > self.face_thresh or
                float(confs[2]) > self.face_thresh
            )
            shoulder_visible = (
                float(confs[5]) > self.shoulder_thresh or
                float(confs[6]) > self.shoulder_thresh
            )
            back_view     = shoulder_visible and not face_visible
            wrist_visible = (
                float(confs[9])  > self.wrist_thresh or
                float(confs[10]) > self.wrist_thresh
            )

            if face_visible or back_view:
                return False, confs
            if wrist_visible:
                return True, confs

        return False, None

    # ─────────────────────────────────────────────────────────────────────────
    # 步骤 2 辅助：绘制手臂骨架（YOLOv8-Pose，层 A）
    # ─────────────────────────────────────────────────────────────────────────
    def _draw_body_skeleton(self, frame, pose_results):
        """绘制肩→肘→手腕骨架连线与关键点圆点"""
        for result in pose_results:
            if result.keypoints is None or len(result.keypoints.xy) == 0:
                continue
            if result.keypoints.conf is None:
                continue
            kps   = result.keypoints.xy[0].cpu().numpy()
            confs = result.keypoints.conf[0].cpu().numpy()

            for (i, j) in BODY_SKELETON_PAIRS:
                if i >= len(kps) or j >= len(kps):
                    continue
                if float(confs[i]) < 0.2 or float(confs[j]) < 0.2:
                    continue
                x1, y1 = int(kps[i][0]), int(kps[i][1])
                x2, y2 = int(kps[j][0]), int(kps[j][1])
                if (x1, y1) == (0, 0) or (x2, y2) == (0, 0):
                    continue
                cv2.line(frame, (x1, y1), (x2, y2), COLOR_BODY_BONE, 2)

            for idx in [5, 6, 7, 8, 9, 10]:
                if idx >= len(kps) or float(confs[idx]) < 0.2:
                    continue
                x, y = int(kps[idx][0]), int(kps[idx][1])
                if (x, y) == (0, 0):
                    continue
                r = 6 if idx in (9, 10) else 4
                cv2.circle(frame, (x, y), r, COLOR_BODY_KP, -1)
                cv2.circle(frame, (x, y), r + 1, (0, 0, 0), 1)
        return frame

    # ─────────────────────────────────────────────────────────────────────────
    # 步骤 2 辅助：绘制十指骨架（MediaPipe，层 B）
    # ─────────────────────────────────────────────────────────────────────────
    def _draw_hand_skeleton(self, frame):
        """
        使用 MediaPipe HandLandmarker 检测手部21关键点，
        绘制十指完整骨架连线（蓝色）和指尖圆点（青色）。

        直接使用传入的 frame（BGR）做 RGB 转换后送入 MediaPipe，
        不再重复 cv2.imread，避免 I/O 冗余和坐标系不一致风险。
        置信度由初始化时的 hand_detect_conf / hand_presence_conf 控制。
        """
        if not HAS_MEDIAPIPE:
            return frame

        h, w = frame.shape[:2]

        # BGR -> RGB，送入 MediaPipe
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.hand_detector.detect(mp_img)

        for hand_landmarks in result.hand_landmarks:
            # 归一化坐标 -> 像素坐标
            pts = [
                (int(lm.x * w), int(lm.y * h))
                for lm in hand_landmarks
            ]

            # 绘制骨架连线
            for (i, j) in HAND_SKELETON_PAIRS:
                if i >= len(pts) or j >= len(pts):
                    continue
                cv2.line(frame, pts[i], pts[j], COLOR_HAND_BONE, 2)

            # 绘制关键点圆点
            for idx, pt in enumerate(pts):
                if idx in FINGERTIP_INDICES:
                    cv2.circle(frame, pt, 6, COLOR_FINGERTIP, -1)
                    cv2.circle(frame, pt, 7, (0, 0, 0), 1)
                else:
                    cv2.circle(frame, pt, 3, COLOR_HAND_KP, -1)

        return frame

    # ─────────────────────────────────────────────────────────────────────────
    # 步骤 2 辅助：绘制工具目标框（YOLOv8-World，层 C）
    # ─────────────────────────────────────────────────────────────────────────
    def _draw_tool_boxes(self, frame, world_results):
        """绘制 YOLOv8-World 检测到的工具目标框（红色）及类别标签"""
        for result in world_results:
            if result.boxes is None or len(result.boxes) == 0:
                continue
            for box in result.boxes:
                conf = float(box.conf[0])
                if conf < self.tool_conf:
                    continue
                cls_id   = int(box.cls[0])
                cls_name = (
                    self.tool_classes[cls_id]
                    if cls_id < len(self.tool_classes) else "tool"
                )
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_TOOL_BOX, 2)

                label = f"{cls_name} {conf:.2f}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                cv2.rectangle(
                    frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), COLOR_TOOL_BOX, -1
                )
                cv2.putText(
                    frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA
                )
        return frame

    # ─────────────────────────────────────────────────────────────────────────
    # 步骤 2：姿态过滤 + 三层标注 → 输出视频
    # ─────────────────────────────────────────────────────────────────────────
    def filter_frames_to_video(self, frames_folder, original_video_path):
        """
        对抽帧结果逐帧进行：
          1. 背影/正脸过滤（YOLOv8-Pose）
          2. 层 A: 手臂骨架标注（YOLOv8-Pose）
          3. 层 B: 十指骨架标注（MediaPipe HandLandmarker，置信度可调）
          4. 层 C: 工具目标框标注（YOLOv8-World）
        最终将保留帧重新编码为 MP4 视频输出。
        """
        print(f"\n[Step 2] Filtering + Annotating '{frames_folder.name}'...")
        print(
            f"  Body filter : face>{self.face_thresh} | "
            f"shoulder>{self.shoulder_thresh} | wrist>{self.wrist_thresh}"
        )
        print(
            f"  Hand model  : detect={self.hand_detect_conf} | "
            f"presence={self.hand_presence_conf} | track={self.hand_track_conf}"
        )

        # 读取原始视频分辨率
        cap_orig = cv2.VideoCapture(str(original_video_path))
        orig_w = int(cap_orig.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap_orig.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap_orig.release()

        output_video_path = self.filtered_dir / f"{frames_folder.name}_filtered.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(output_video_path), fourcc, self.fps, (orig_w, orig_h))

        all_frames = sorted(list(frames_folder.glob("*.jpg")))
        valid_count = drop_face = drop_back = drop_no_hand = 0

        for frame_path in tqdm(all_frames, desc="  Processing"):
            frame = cv2.imread(str(frame_path))
            if frame is None:
                continue

            if not HAS_YOLO:
                # Mock 模式：无 YOLO 时直接写入所有帧
                frame = cv2.resize(frame, (orig_w, orig_h))
                writer.write(frame)
                valid_count += 1
                continue

            # ── 姿态检测与过滤 ────────────────────────────────────────────
            pose_results = self.pose_model(str(frame_path), verbose=False)
            is_valid, confs = self._is_valid_frame(pose_results)

            if not is_valid:
                if confs is not None:
                    face_vis = (
                        float(confs[0]) > self.face_thresh or
                        float(confs[1]) > self.face_thresh or
                        float(confs[2]) > self.face_thresh
                    )
                    shoulder_vis = (
                        float(confs[5]) > self.shoulder_thresh or
                        float(confs[6]) > self.shoulder_thresh
                    )
                    if face_vis:
                        drop_face += 1
                    elif shoulder_vis:
                        drop_back += 1
                    else:
                        drop_no_hand += 1
                else:
                    drop_no_hand += 1
                continue

            # ── 层 A: 手臂骨架标注（YOLOv8-Pose） ────────────────────────
            frame = self._draw_body_skeleton(frame, pose_results)

            # ── 层 B: 十指骨架标注（MediaPipe HandLandmarker） ────────────
            # 注意：直接传入 frame（已含层A标注），MediaPipe 在其上叠加绘制
            frame = self._draw_hand_skeleton(frame)

            # ── 层 C: 工具目标框标注（YOLOv8-World） ─────────────────────
            world_results = self.world_model(str(frame_path), verbose=False)
            frame = self._draw_tool_boxes(frame, world_results)

            # ── resize 到原始分辨率后写入视频 ────────────────────────────
            frame = cv2.resize(frame, (orig_w, orig_h))
            writer.write(frame)
            valid_count += 1

        writer.release()
        total = len(all_frames)
        sep = "-" * 50
        print("\n  " + sep)
        print("  Filter Summary")
        print("  " + sep)
        print(f"  Total frames  : {total}")
        print(f"  Kept frames   : {valid_count}  ({valid_count / max(total, 1) * 100:.1f}%)")
        print(f"  Drop (face)   : {drop_face}")
        print(f"  Drop (back)   : {drop_back}")
        print(f"  Drop (no hand): {drop_no_hand}")
        print(f"  Output video  : {output_video_path}")
        print("  " + sep)
        return output_video_path, valid_count

    # ─────────────────────────────────────────────────────────────────────────
    # 步骤 3：动作标注（已禁用）
    # ─────────────────────────────────────────────────────────────────────────
    # def generate_captions(self, valid_frames_folder):
    #     """步骤 3: 使用 Qwen2-VL-7B 对连续帧片段生成动作指令标注"""
    #     print(f"Generating captions for {valid_frames_folder.name}...")
    #     dataset_records = []
    #     frames = sorted(list(valid_frames_folder.glob("*.jpg")))
    #     window_size = 4
    #     for i in tqdm(range(0, len(frames) - window_size + 1, window_size)):
    #         clip_frames = frames[i:i+window_size]
    #         if not HAS_QWEN:
    #             caption = "Mock action: manipulating object"
    #         else:
    #             messages = [{
    #                 "role": "user",
    #                 "content": [
    #                     {"type": "video", "video": [str(f) for f in clip_frames]},
    #                     {"type": "text", "text": "请用简短的中文描述视频中手部正在执行的具体动作，"
    #                                               "例如'拧螺丝'、'拿起螺丝刀'、'拆卸外壳'。"
    #                                               "只需输出动作本身，不要多余的描述。"}
    #                 ]
    #             }]
    #             text = self.qwen_processor.apply_chat_template(
    #                 messages, tokenize=False, add_generation_prompt=True)
    #             image_inputs, video_inputs = process_vision_info(messages)
    #             inputs = self.qwen_processor(
    #                 text=[text], images=image_inputs, videos=video_inputs,
    #                 padding=True, return_tensors="pt").to("cuda")
    #             generated_ids = self.qwen_model.generate(**inputs, max_new_tokens=20)
    #             generated_ids_trimmed = [
    #                 out_ids[len(in_ids):]
    #                 for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    #             caption = self.qwen_processor.batch_decode(
    #                 generated_ids_trimmed, skip_special_tokens=True,
    #                 clean_up_tokenization_spaces=False)[0].strip()
    #         dataset_records.append({
    #             "video_id": valid_frames_folder.name,
    #             "frame_paths": [str(f.name) for f in clip_frames],
    #             "instruction": caption
    #         })
    #     output_json = self.dataset_dir / f"{valid_frames_folder.name}_annotations.json"
    #     with open(output_json, 'w', encoding='utf-8') as f:
    #         json.dump(dataset_records, f, ensure_ascii=False, indent=2)
    #     return dataset_records

    # ─────────────────────────────────────────────────────────────────────────
    # 主流程
    # ─────────────────────────────────────────────────────────────────────────
    def run(self):
        video_extensions = ["*.mp4", "*.flv", "*.mkv", "*.avi", "*.mov"]
        video_files = []
        for ext in video_extensions:
            video_files.extend(self.input_dir.glob(ext))

        if not video_files:
            print(f"No video files found in {self.input_dir}")
            return

        print(f"Found {len(video_files)} video(s) in {self.input_dir}")
        summary = []

        for video_path in video_files:
            sep = "=" * 58
            print(f"\n{sep}")
            print(f"  Processing: {video_path.name}")
            print(f"{sep}")

            frames_folder = self.extract_frames(video_path)
            output_video, valid_count = self.filter_frames_to_video(
                frames_folder, video_path
            )

            # 步骤 3 已禁用
            # if valid_count > 0:
            #     records = self.generate_captions(self.filtered_dir / frames_folder.name)

            summary.append({
                "source_video":   video_path.name,
                "filtered_video": str(output_video),
                "valid_frames":   valid_count,
            })

        sep = "=" * 58
        print(f"\n{sep}")
        print("  Pipeline completed.")
        print(f"  Filtered videos -> {self.filtered_dir}")
        print("-" * 58)
        for s in summary:
            print(f"  {s['source_video']}")
            print(f"    -> {s['valid_frames']} valid frames")
            print(f"    -> {s['filtered_video']}")
        print(f"{sep}")


# ─────────────────────────────────────────────────────────────────────────────
# 入口
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RynnVLA-001 Data Curation Pipeline v3.1",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例:\n"
            "  python rynn_vla_data_pipeline.py \\\n"
            "    --input_dir  F:\\raw_video \\\n"
            "    --output_dir F:\\rynn_dataset \\\n"
            "    --fps 2\n\n"
            "十指骨架置信度调整（值越高越严格，检测到的手越少但更准确）:\n"
            "  --hand_detect_conf   0.7   # 提高手部检测阈值\n"
            "  --hand_presence_conf 0.6   # 提高手部存在阈值\n"
            "  --hand_track_conf    0.6   # 提高手部追踪阈值\n\n"
            "背影/正脸过滤灵敏度调整（值越小越严格）:\n"
            "  --shoulder_thresh 0.25    # 更严格地过滤背影\n"
            "  --face_thresh     0.4     # 更严格地过滤正脸\n"
        )
    )
    # 基础参数
    parser.add_argument("--input_dir",  type=str, required=True, help="原始视频所在目录")
    parser.add_argument("--output_dir", type=str, required=True, help="处理结果输出目录")
    parser.add_argument("--fps",        type=int, default=2,     help="抽帧帧率（默认 2 FPS）")

    # 姿态过滤参数
    parser.add_argument("--shoulder_thresh", type=float, default=0.35,
                        help="肩膀关键点置信度阈值（背影判断，默认 0.35）")
    parser.add_argument("--wrist_thresh",    type=float, default=0.3,
                        help="手腕关键点置信度阈值（默认 0.3）")
    parser.add_argument("--face_thresh",     type=float, default=0.5,
                        help="面部关键点置信度阈值（默认 0.5）")

    # 十指骨架置信度参数（可调）
    parser.add_argument("--hand_detect_conf",   type=float, default=0.5,
                        help="MediaPipe 手部检测置信度（默认 0.5，越高越严格）")
    parser.add_argument("--hand_presence_conf", type=float, default=0.5,
                        help="MediaPipe 手部存在置信度（默认 0.5）")
    parser.add_argument("--hand_track_conf",    type=float, default=0.5,
                        help="MediaPipe 手部追踪置信度（默认 0.5）")
    parser.add_argument("--hand_model", type=str, default="hand_landmarker.task",
                        help="MediaPipe hand_landmarker.task 模型文件路径（默认与脚本同目录）")

    # 工具检测参数
    parser.add_argument("--tool_conf", type=float, default=0.25,
                        help="工具检测置信度阈值（默认 0.25）")
    parser.add_argument("--tool_classes", type=str, nargs="+",
                        default=["screwdriver", "drill", "electric drill",
                                 "wrench", "hammer", "pliers", "cutter"],
                        help="YOLOv8-World 检测的工具类别（空格分隔）")

    args = parser.parse_args()

    pipeline = RynnVLADataPipeline(
        input_dir          = args.input_dir,
        output_dir         = args.output_dir,
        fps                = args.fps,
        shoulder_thresh    = args.shoulder_thresh,
        wrist_thresh       = args.wrist_thresh,
        face_thresh        = args.face_thresh,
        hand_detect_conf   = args.hand_detect_conf,
        hand_presence_conf = args.hand_presence_conf,
        hand_track_conf    = args.hand_track_conf,
        hand_model         = args.hand_model,
        tool_classes       = args.tool_classes,
        tool_conf          = args.tool_conf,
    )
    pipeline.run()
