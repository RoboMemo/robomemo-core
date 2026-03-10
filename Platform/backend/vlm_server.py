import sys
import json
import cv2
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
import torch

def extract_frames(video_path, num_frames=16):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = [int(i) for i in torch.linspace(0, total_frames - 1, num_frames)]
    
    frames = []
    for i in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    cap.release()
    return frames

def main():
    if len(sys.argv) < 3:
        print(json.dumps({"error": "Usage: python vlm_server.py <video_path> <prompt>"}))
        return

    video_path = sys.argv[1]
    prompt = sys.argv[2]

    try:
        # Load model and processor
        processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-Instruct")
        model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolVLM-Instruct")

        # Extract frames from video
        video_frames = extract_frames(video_path)

        # Prepare messages for the model
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"} for _ in video_frames
                ] + [
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        # Process inputs
        inputs = processor(text=messages, images=video_frames, return_tensors="pt")

        # Generate response
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=100)
        
        # Decode and print the result
        result_text = processor.decode(outputs[0], skip_special_tokens=True)
        print(json.dumps({"result": result_text}))

    except Exception as e:
        print(json.dumps({"error": str(e)}))

if __name__ == "__main__":
    main()
