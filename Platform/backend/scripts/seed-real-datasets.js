/**
 * Seed real datasets into RoboMemo DB
 * Sources: GenRobot 10Kh H5 (local), LeRobot HF datasets (downloaded)
 */
const Database = require('better-sqlite3');
const path = require('path');
const fs = require('fs');

const DB_PATH = path.join(__dirname, '..', 'data', 'robomemo.db');
const db = new Database(DB_PATH);
db.pragma('journal_mode = WAL');
db.pragma('foreign_keys = ON');

// ── Clear old data ──────────────────────────────────────────────────
console.log('Clearing old datasets and episodes...');
db.exec('DELETE FROM episodes');
db.exec('DELETE FROM datasets');
console.log('  Done.');

// ── Helper ──────────────────────────────────────────────────────────
const insertDataset = db.prepare(`
  INSERT OR REPLACE INTO datasets (id, name, description, format, robot_type, source, task_desc, episode_count, sensor_config, created_at)
  VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
`);

// Check episode table columns
const cols = db.prepare("PRAGMA table_info(episodes)").all().map(c => c.name);
console.log('Episode columns:', cols.join(', '));

const insertEpisode = db.prepare(`
  INSERT OR REPLACE INTO episodes (id, dataset_id, name, description, skill, category, h5_path, frame_count, duration, fps, robot, bimanual, sensors, created_at)
  VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
`);

// ─────────────────────────────────────────────────────────────────────
// 1. GenRobot 10Kh-RealOmin-OpenData (local H5 files)
// ─────────────────────────────────────────────────────────────────────
console.log('\n=== Seeding GenRobot 10Kh-RealOmin-OpenData ===');

const GENROBOT_BASE = path.join(__dirname, '..', 'data', 'genrobot_open_dataset');
const GENROBOT_FPS = 30;

const genrobotSensorConfig = {
  name: 'GenDAS Gripper Sensor Suite',
  sensors: [
    { name: 'Mid Fisheye Camera', type: 'camera', location: 'chest', resolution: '640x480', modality: 'rgb' },
    { name: 'IMU 6-axis', type: 'imu', location: 'base', axes: 6, rate_hz: 100 },
    { name: 'Left Tactile Array', type: 'tactile', location: 'left_gripper', dimensions: '12x8', rate_hz: 30 },
    { name: 'Right Tactile Array', type: 'tactile', location: 'right_gripper', dimensions: '12x8', rate_hz: 30 },
    { name: 'Magnetic Encoder', type: 'encoder', location: 'joints', channels: 8, rate_hz: 30 }
  ]
};

const sceneDescriptions = {
  'Clutter_Tidy-Up': 'Desktop and table clutter tidying tasks',
  'Cooking_and_Kitchen_Clean': 'Kitchen cooking and cleaning manipulation',
  'Folding_Clothes_and_Zipper_Operations': 'Garment folding and zipper operations',
  'Organize_Clutter': 'Household object organization tasks',
  'Shoes_Handling': 'Shoe lacing and organization tasks'
};

const skillDescriptions = {
  'carton_sorting_clutter': 'Sort and organize scattered carton boxes',
  'flexible_grasping_and_sorting': 'Grasp and sort flexible/deformable objects',
  'irregular_object_clutter': 'Tidy up irregularly shaped objects',
  'small_object_storage': 'Pick and store small household items',
  'clean_bowl': 'Clean a bowl with wiping motions',
  'clean_container': 'Clean and organize food containers',
  'unscrew_bottle_cap_and_pour': 'Unscrew a bottle cap and pour contents',
  'fold_and_store_clothes': 'Fold garments and store them neatly',
  'zip_clothes': 'Operate zipper on clothing items',
  'desktop_object_sorting': 'Sort objects on a desktop workspace',
  'drawer_to_take_items': 'Open drawer and retrieve items from it',
  'fold_and_store_shopping_bag': 'Fold and organize shopping bags',
  'fold_towel': 'Fold a towel using bimanual coordination',
  'lace_up_shoes_with_both_hands': 'Thread and tie shoe laces with both hands',
  'organize_scattered_shoes': 'Collect and organize scattered shoes'
};

insertDataset.run(
  'genrobot_10kh',
  '10Kh-RealOmin-OpenData',
  'Largest open embodied intelligence dataset. 10,000+ hours of real household robot manipulation. 10 home scenarios, 30 skills. Data format: H5 with mid-fisheye camera, 6-axis IMU, dual tactile arrays, magnetic encoders.',
  'h5',
  'GenDAS Gripper',
  'genrobot2025/10Kh-RealOmin-OpenData',
  'Bimanual household manipulation: folding, cleaning, sorting, organizing',
  0, // will update after counting
  JSON.stringify(genrobotSensorConfig)
);

let genrobotEpCount = 0;
const scenes = fs.readdirSync(GENROBOT_BASE).filter(d => 
  fs.statSync(path.join(GENROBOT_BASE, d)).isDirectory() && !d.startsWith('.')
);

for (const scene of scenes.sort()) {
  const scenePath = path.join(GENROBOT_BASE, scene);
  const skills = fs.readdirSync(scenePath).filter(d => 
    fs.statSync(path.join(scenePath, d)).isDirectory() && !d.startsWith('.')
  );
  
  for (const skill of skills.sort()) {
    const skillPath = path.join(scenePath, skill);
    const epFiles = fs.readdirSync(skillPath).filter(f => f.endsWith('.h5') && !f.startsWith('._'));
    
    for (const epFile of epFiles.sort()) {
      const epId = `genrobot_${scene}_${skill}_${epFile.replace('.h5', '')}`;
      const epName = `${skill.replace(/_/g, ' ')} (${epFile.replace('.h5', '')})`;
      const h5Path = path.join(skillPath, epFile);
      const fileSize = fs.statSync(h5Path).size;
      
      // Use frame counts from our scan (we know them)
      // Read from file size heuristic: ~40KB per frame for this dataset
      const estimatedFrames = Math.round(fileSize / 40000);
      
      insertEpisode.run(
        epId,
        'genrobot_10kh',
        epName,
        skillDescriptions[skill] || `${scene}: ${skill}`,
        skill,
        scene,
        path.relative(path.join(__dirname, '..'), h5Path),
        estimatedFrames,
        estimatedFrames / GENROBOT_FPS,
        GENROBOT_FPS,
        'GenDAS Gripper',
        1, // bimanual
        JSON.stringify(['mid_fisheye_color', 'imu_6axis', 'tactile_left', 'tactile_right', 'magnetic_encoder'])
      );
      genrobotEpCount++;
    }
  }
}

db.prepare('UPDATE datasets SET episode_count = ? WHERE id = ?').run(genrobotEpCount, 'genrobot_10kh');
console.log(`  Inserted ${genrobotEpCount} episodes`);

// ─────────────────────────────────────────────────────────────────────
// 2. LeRobot PushT
// ─────────────────────────────────────────────────────────────────────
console.log('\n=== Seeding LeRobot PushT ===');

const PUSHT_BASE = path.join(__dirname, '..', 'data', 'datasets', 'lerobot_pusht');
const pushtInfo = JSON.parse(fs.readFileSync(path.join(PUSHT_BASE, 'meta', 'info.json')));

const pushtSensorConfig = {
  name: 'PushT Setup',
  sensors: [
    { name: 'Top-down Camera', type: 'camera', location: 'overhead', resolution: '96x96', modality: 'rgb' }
  ]
};

insertDataset.run(
  'lerobot_pusht',
  'LeRobot PushT',
  'Push-T benchmark: a 2D end-effector pushes a T-shaped block to a target pose. 206 human demonstrations with reward signals. Standard benchmark for diffusion policy and action-chunking methods.',
  'lerobot_v3',
  'Planar 2-DOF',
  'lerobot/pusht',
  'Push T-shaped block to target pose',
  pushtInfo.total_episodes,
  JSON.stringify(pushtSensorConfig)
);

// Seed representative episodes (not all 206)
const pushtEpsToSeed = [0, 1, 2, 5, 10, 20, 50, 100, 150, 200];
const framesPerEp = Math.round(pushtInfo.total_frames / pushtInfo.total_episodes);

for (const i of pushtEpsToSeed) {
  if (i >= pushtInfo.total_episodes) continue;
  insertEpisode.run(
    `pusht_ep_${String(i).padStart(4, '0')}`,
    'lerobot_pusht',
    `PushT Episode ${i}`,
    'Push T-shaped block to target configuration',
    'push_to_target',
    'manipulation_2d',
    `data/datasets/lerobot_pusht/videos/observation.image/chunk-000/file-000.mp4`,
    framesPerEp,
    framesPerEp / pushtInfo.fps,
    pushtInfo.fps,
    'Planar 2-DOF',
    0,
    JSON.stringify(['observation.image'])
  );
}
console.log(`  Inserted ${pushtEpsToSeed.filter(i => i < pushtInfo.total_episodes).length} representative episodes (of ${pushtInfo.total_episodes} total)`);

// ─────────────────────────────────────────────────────────────────────
// 3. LeRobot xArm Lift
// ─────────────────────────────────────────────────────────────────────
console.log('\n=== Seeding LeRobot xArm Lift ===');

const XARM_BASE = path.join(__dirname, '..', 'data', 'datasets', 'lerobot_xarm_lift');
const xarmInfo = JSON.parse(fs.readFileSync(path.join(XARM_BASE, 'meta', 'info.json')));

const xarmSensorConfig = {
  name: 'xArm Workspace Setup',
  sensors: [
    { name: 'Workspace Camera', type: 'camera', location: 'overhead', resolution: '84x84', modality: 'rgb' },
    { name: 'Joint State', type: 'joint_state', location: 'arm', dof: 4 }
  ]
};

insertDataset.run(
  'lerobot_xarm_lift',
  'LeRobot xArm Lift Medium',
  'xArm robot arm lifting objects. 800 replay episodes from D4RL-style medium-quality demonstrations. Single camera + 4-DOF joint state observations.',
  'lerobot_v3',
  'xArm',
  'lerobot/xarm_lift_medium_replay',
  'Lift object from table using xArm robot',
  xarmInfo.total_episodes,
  JSON.stringify(xarmSensorConfig)
);

const xarmEpsToSeed = [0, 1, 2, 5, 10, 50, 100, 200, 400, 600, 799];
const xarmFramesPerEp = Math.round(xarmInfo.total_frames / xarmInfo.total_episodes);

for (const i of xarmEpsToSeed) {
  if (i >= xarmInfo.total_episodes) continue;
  insertEpisode.run(
    `xarm_lift_ep_${String(i).padStart(4, '0')}`,
    'lerobot_xarm_lift',
    `xArm Lift Episode ${i}`,
    'Lift object from table surface',
    'lift',
    'manipulation_3d',
    `data/datasets/lerobot_xarm_lift/videos/observation.image/chunk-000/file-000.mp4`,
    xarmFramesPerEp,
    xarmFramesPerEp / xarmInfo.fps,
    xarmInfo.fps,
    'xArm',
    0,
    JSON.stringify(['observation.image', 'observation.state'])
  );
}
console.log(`  Inserted ${xarmEpsToSeed.filter(i => i < xarmInfo.total_episodes).length} representative episodes (of ${xarmInfo.total_episodes} total)`);

// ─────────────────────────────────────────────────────────────────────
// 4. ALOHA Static Cups Open (4 cameras, bimanual)
// ─────────────────────────────────────────────────────────────────────
console.log('\n=== Seeding ALOHA Static Cups Open ===');

const CUPS_BASE = path.join(__dirname, '..', 'data', 'datasets', 'lerobot_aloha_cups');
const cupsInfo = JSON.parse(fs.readFileSync(path.join(CUPS_BASE, 'meta', 'info.json')));

const cupsSensorConfig = {
  name: 'ALOHA Static 4-Camera Bimanual',
  sensors: [
    { name: 'High Camera', type: 'camera', location: 'overhead', resolution: '480x640', modality: 'rgb' },
    { name: 'Low Camera', type: 'camera', location: 'table_level', resolution: '480x640', modality: 'rgb' },
    { name: 'Left Wrist Camera', type: 'camera', location: 'left_wrist', resolution: '480x640', modality: 'rgb' },
    { name: 'Right Wrist Camera', type: 'camera', location: 'right_wrist', resolution: '480x640', modality: 'rgb' },
    { name: 'Left Arm Joints', type: 'joint_state', location: 'left_arm', dof: 7 },
    { name: 'Right Arm Joints', type: 'joint_state', location: 'right_arm', dof: 7 }
  ]
};

insertDataset.run(
  'aloha_static_cups',
  'ALOHA Static — Cups Open',
  'Bimanual ALOHA robot opening cups. 50 teleoperated demonstrations with 4 synchronized camera views (overhead, table-level, left wrist, right wrist). Standard benchmark for bimanual manipulation with ACT policy.',
  'lerobot_v3',
  'ALOHA (Bimanual)',
  'lerobot/aloha_static_cups_open',
  'Open cups using bimanual coordination',
  cupsInfo.total_episodes,
  JSON.stringify(cupsSensorConfig)
);

const cupsFramesPerEp = Math.round(cupsInfo.total_frames / cupsInfo.total_episodes);
const cupsEpsToSeed = Array.from({length: Math.min(cupsInfo.total_episodes, 20)}, (_, i) => i);

for (const i of cupsEpsToSeed) {
  insertEpisode.run(
    `aloha_cups_ep_${String(i).padStart(4, '0')}`,
    'aloha_static_cups',
    `Cups Open Episode ${i}`,
    'Bimanual cup opening with 4 camera views',
    'open_cup',
    'bimanual_manipulation',
    `data/datasets/lerobot_aloha_cups/videos/observation.images.cam_high/chunk-000/file-000.mp4`,
    cupsFramesPerEp,
    cupsFramesPerEp / cupsInfo.fps,
    cupsInfo.fps,
    'ALOHA',
    1,
    JSON.stringify(['cam_high', 'cam_low', 'cam_left_wrist', 'cam_right_wrist', 'joint_state_left', 'joint_state_right'])
  );
}
console.log(`  Inserted ${cupsEpsToSeed.length} episodes (of ${cupsInfo.total_episodes} total)`);

// ─────────────────────────────────────────────────────────────────────
// 5. ALOHA Mobile Shrimp (3 cameras, mobile bimanual)
// ─────────────────────────────────────────────────────────────────────
console.log('\n=== Seeding ALOHA Mobile Shrimp ===');

const SHRIMP_BASE = path.join(__dirname, '..', 'data', 'datasets', 'lerobot_aloha_shrimp');
const shrimpInfo = JSON.parse(fs.readFileSync(path.join(SHRIMP_BASE, 'meta', 'info.json')));

const shrimpSensorConfig = {
  name: 'ALOHA Mobile 3-Camera Bimanual',
  sensors: [
    { name: 'High Camera', type: 'camera', location: 'head', resolution: '480x640', modality: 'rgb' },
    { name: 'Left Wrist Camera', type: 'camera', location: 'left_wrist', resolution: '480x640', modality: 'rgb' },
    { name: 'Right Wrist Camera', type: 'camera', location: 'right_wrist', resolution: '480x640', modality: 'rgb' },
    { name: 'Left Arm Joints', type: 'joint_state', location: 'left_arm', dof: 7 },
    { name: 'Right Arm Joints', type: 'joint_state', location: 'right_arm', dof: 7 },
    { name: 'Mobile Base', type: 'base_velocity', location: 'base', dof: 2 }
  ]
};

insertDataset.run(
  'aloha_mobile_shrimp',
  'ALOHA Mobile — Shrimp Cooking',
  'Mobile bimanual ALOHA robot cooking shrimp. 18 teleoperated demonstrations with 3 synchronized cameras (head + 2 wrists). Long-horizon mobile manipulation task requiring navigation + bimanual cooking skills.',
  'lerobot_v3',
  'ALOHA Mobile (Bimanual)',
  'lerobot/aloha_mobile_shrimp',
  'Cook shrimp: approach stove, pick tools, flip shrimp',
  shrimpInfo.total_episodes,
  JSON.stringify(shrimpSensorConfig)
);

const shrimpFramesPerEp = Math.round(shrimpInfo.total_frames / shrimpInfo.total_episodes);

for (let i = 0; i < shrimpInfo.total_episodes; i++) {
  insertEpisode.run(
    `aloha_shrimp_ep_${String(i).padStart(4, '0')}`,
    'aloha_mobile_shrimp',
    `Shrimp Cooking Episode ${i}`,
    'Mobile bimanual shrimp cooking with 3 camera views',
    'cook_shrimp',
    'mobile_bimanual_cooking',
    `data/datasets/lerobot_aloha_shrimp/videos/observation.images.cam_high/chunk-000/file-000.mp4`,
    shrimpFramesPerEp,
    shrimpFramesPerEp / shrimpInfo.fps,
    shrimpInfo.fps,
    'ALOHA Mobile',
    1,
    JSON.stringify(['cam_high', 'cam_left_wrist', 'cam_right_wrist', 'joint_state_left', 'joint_state_right', 'base_velocity'])
  );
}
console.log(`  Inserted ${shrimpInfo.total_episodes} episodes`);

// ─────────────────────────────────────────────────────────────────────
// 6. RoboForce Titan (placeholder — real data not yet arrived)
// ─────────────────────────────────────────────────────────────────────
console.log('\n=== Seeding RoboForce Titan (awaiting real data) ===');

const rfSensorConfig = {
  name: 'RoboForce Titan Sensor Suite',
  sensors: [
    { name: 'Chest RGBD Camera', type: 'rgbd', location: 'chest', resolution: '640x480', modality: 'rgb+depth' },
    { name: 'Left Wrist RGBD Camera', type: 'rgbd', location: 'left_wrist', resolution: '640x480', modality: 'rgb+depth' },
    { name: 'Right Wrist RGBD Camera', type: 'rgbd', location: 'right_wrist', resolution: '640x480', modality: 'rgb+depth' },
    { name: 'Left EOAT F/T (6-axis)', type: 'force_torque', location: 'left_eoat', axes: 6, rate_hz: 1000 },
    { name: 'Right EOAT F/T (6-axis)', type: 'force_torque', location: 'right_eoat', axes: 6, rate_hz: 1000 },
    { name: 'Left Arm Joints', type: 'joint_state', location: 'left_arm', dof: 7 },
    { name: 'Right Arm Joints', type: 'joint_state', location: 'right_arm', dof: 7 }
  ]
};

insertDataset.run(
  'roboforce_titan_screw_v1',
  'RoboForce Titan — Screw Driving Tasks',
  'Bimanual screw-tightening dataset from RoboForce Titan robot. 3x RGBD cameras (chest + 2 wrists) + 2x 6-axis Force/Torque at EOAT. Tasks include M3-M5 screw insertion and tightening with torque verification. ⚠️ Awaiting real data delivery — current episodes are structured placeholders.',
  'roboforce_native',
  'RoboForce Titan',
  'roboforce/titan-screw-v1',
  'Screw-driving: locate, align, insert, tighten, torque verify',
  5,
  JSON.stringify(rfSensorConfig)
);

const rfEpisodes = [
  { id: 'rf_ep_m3_hex', name: 'M3x10 Hex Screw — Station A', skill: 'screw_tighten', frames: 555, dur: 18.5, desc: 'M3x10 hex head screw insertion and tightening to 0.5Nm' },
  { id: 'rf_ep_m4_phillips', name: 'M4x16 Phillips — Station A', skill: 'screw_tighten', frames: 669, dur: 22.3, desc: 'M4x16 Phillips head screw with auto-alignment correction' },
  { id: 'rf_ep_m5_hex', name: 'M5x20 Hex Screw — Station B', skill: 'screw_tighten', frames: 474, dur: 15.8, desc: 'M5x20 hex head screw in angled surface mount' },
  { id: 'rf_ep_crossthread', name: 'Cross-thread Recovery — Station A', skill: 'error_recovery', frames: 843, dur: 28.1, desc: 'Detect and recover from cross-threading during M4 insertion' },
  { id: 'rf_ep_multi', name: 'Multi-screw Sequence — Station B', skill: 'multi_screw', frames: 1056, dur: 35.2, desc: 'Sequential tightening of 4 screws in diagonal pattern' },
];

for (const ep of rfEpisodes) {
  insertEpisode.run(
    ep.id, 'roboforce_titan_screw_v1', ep.name, ep.desc, ep.skill, 'screw_driving',
    null, ep.frames, ep.dur, 30, 'RoboForce Titan', 1,
    JSON.stringify(['chest_rgbd', 'left_wrist_rgbd', 'right_wrist_rgbd', 'left_ft_6axis', 'right_ft_6axis'])
  );
}
console.log('  Inserted 5 placeholder episodes');

// ── Summary ──────────────────────────────────────────────────────────
console.log('\n=== Final DB State ===');
const datasets = db.prepare('SELECT id, name, episode_count FROM datasets').all();
const totalEps = db.prepare('SELECT COUNT(*) as c FROM episodes').get().c;
console.log(`Datasets: ${datasets.length}`);
for (const d of datasets) {
  const actualEps = db.prepare('SELECT COUNT(*) as c FROM episodes WHERE dataset_id = ?').get(d.id).c;
  console.log(`  ${d.id}: "${d.name}" — ${actualEps} episodes (declared: ${d.episode_count})`);
}
console.log(`Total episodes in DB: ${totalEps}`);

db.close();
console.log('\nDone! ✅');
