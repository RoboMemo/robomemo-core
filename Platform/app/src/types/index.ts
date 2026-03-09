export interface Dataset {
  id: string;
  name: string;
  description: string;
  format: 'lerobot' | 'rtx' | 'rlds' | 'openx';
  robotType: string;
  sensorConfig: SensorConfig;
  taskDescription?: string;
  environment?: Record<string, any>;
  createdAt: string;
  updatedAt: string;
  episodeCount: number;
  frameCount: number;
  size: number;
  version: string;
  license: string;
}

export interface Episode {
  id: string;
  datasetId: string;
  name: string;
  description?: string;
  createdAt: string;
  frameCount: number;
  duration: number;
  metadata: Record<string, any>;
}

export interface Frame {
  id: string;
  episodeId: string;
  datasetId: string;
  frameIndex: number;
  timestamp: string;
  observations: Observation;
  action?: Action;
  reward?: number;
  done?: boolean;
}

export interface Observation {
  images?: Record<string, ImageData>;
  depth?: Record<string, ImageData>;
  proprioception?: ProprioceptionData;
  forceTorque?: Record<string, ForceTorqueData>;
  tactile?: Record<string, TactileData>;
}

export interface ImageData {
  url: string;
  width: number;
  height: number;
  encoding: string;
}

export interface ProprioceptionData {
  jointPositions: number[];
  jointVelocities?: number[];
  jointEffort?: number[];
  endEffectorPose?: Pose;
}

export interface ForceTorqueData {
  force: [number, number, number];
  torque: [number, number, number];
}

export interface TactileData {
  pressure: number[][];
  temperature?: number[][];
  resolution: [number, number];
}

export interface Pose {
  position: [number, number, number];
  orientation: [number, number, number, number];
}

export interface Action {
  jointPositions?: number[];
  jointVelocities?: number[];
  cartesianPose?: Pose;
  gripper?: number;
}

export interface Annotation {
  id: string;
  frameId: string;
  episodeId: string;
  datasetId: string;
  type: 'bbox' | 'mask' | 'keypoints' | 'label';
  label: string;
  bbox?: BoundingBox;
  mask?: MaskData;
  keypoints?: Keypoint[];
  attributes?: Record<string, any>;
  confidence: number;
  annotator: string;
  verified: boolean;
  verifier?: string;
  verifiedAt?: string;
  createdAt: string;
  updatedAt: string;
}

export interface BoundingBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface MaskData {
  data: number[][];
  width: number;
  height: number;
}

export interface Keypoint {
  x: number;
  y: number;
  visible: boolean;
  name?: string;
}

export interface Collection {
  id: string;
  name: string;
  description: string;
  datasetId: string;
  episodeIds: string[];
  filterCriteria?: Record<string, any>;
  tags: string[];
  createdAt: string;
  updatedAt: string;
}

export interface Simulator {
  id: string;
  name: string;
  type: 'isaaclab' | 'mujoco' | 'gazebo';
  description: string;
  status: 'available' | 'running' | 'error';
  config: Record<string, any>;
  supportedRobots: string[];
  supportedSensors: string[];
  currentScene?: string;
  currentRobot?: string;
  activeSensors?: string[];
  startedAt?: string;
  stoppedAt?: string;
  lastUpdate?: string;
}

export interface SensorConfig {
  id?: string;
  name: string;
  description?: string;
  sensors: Sensor[];
}

export interface Sensor {
  type: string;
  name: string;
  location: string;
  parameters?: Record<string, any>;
}

export interface AugmentationModel {
  id: string;
  name: string;
  provider: string;
  description: string;
  capabilities: string[];
  supportedInputs: string[];
  supportedOutputs: string[];
  status: 'available' | 'unavailable' | 'loading';
}

export interface ExportFormat {
  id: string;
  name: string;
  description: string;
  version: string;
  url?: string;
  features: string[];
  fileExtensions: string[];
}

export interface PlatformStats {
  datasets: number;
  collections: number;
  episodes: number;
  frames: number;
  annotations: number;
  simulators: number;
  totalStorage: number;
  timestamp: string;
}

export interface TimelineEntry {
  date: string;
  datasets: number;
  episodes: number;
  annotations: number;
  collections: number;
}

// ========== Structured VQA Types ==========

export interface VQAActionItem {
  action: string;
  timestamp: string;
  frame_range: [number, number];
  description: string;
}

export interface VQATemporal {
  action_sequence: VQAActionItem[];
  relationships: string[];
}

export interface VQASpatialRelationship {
  timestamp: string;
  relationship: string;
  details: string;
}

export interface VQASpatial {
  key_relationships: VQASpatialRelationship[];
  trajectory_spatial: string;
}

export interface VQAObject {
  name: string;
  properties: {
    color: string;
    material: string;
    shape: string;
    size: string;
  };
  state_changes: string[];
}

export interface VQAAttribute {
  objects: VQAObject[];
}

export interface VQAContact {
  timestamp: string;
  contact_type: string;
  force_level: 'light' | 'medium' | 'strong';
  contact_points: string;
  area: string;
}

export interface VQAMechanics {
  contacts: VQAContact[];
  force_profile: string;
}

export interface VQAActionJustification {
  action: string;
  reason: string;
  constraints: string[];
}

export interface VQAReasoning {
  action_justifications: VQAActionJustification[];
  overall_strategy: string;
}

export interface VQASummary {
  task_description: string;
  start_state: string;
  end_state: string;
  success: boolean;
  key_milestones: string[];
  duration: string;
}

export interface VQAMotionSegment {
  segment: string;
  time_range: string;
  motion_type: 'linear' | 'curved' | 'rotational';
  velocity: 'slow' | 'medium' | 'fast';
  waypoints: string[];
}

export interface VQATrajectory {
  motion_segments: VQAMotionSegment[];
  overall_path: string;
}

export interface VQAKeyFrame {
  frame_idx: number;
  timestamp: string;
  significance: string;
}

export interface VQAVideoInfo {
  total_frames: number;
  fps: number;
  duration: number;
}

export interface VQAMetadata {
  video_path: string;
  video_info: VQAVideoInfo;
  num_frames_analyzed: number;
  model: string;
  frame_timestamps: string[];
}

export interface VQAConfidenceScores {
  temporal: number;
  spatial: number;
  attribute: number;
  mechanics: number;
  reasoning: number;
  summary: number;
  trajectory: number;
}

export interface StructuredVQAAnalysis {
  temporal: VQATemporal;
  spatial: VQASpatial;
  attribute: VQAAttribute;
  mechanics: VQAMechanics;
  reasoning: VQAReasoning;
  summary: VQASummary;
  trajectory: VQATrajectory;
  visual_evidence: { key_frames: VQAKeyFrame[] };
  confidence_scores: VQAConfidenceScores;
  metadata: VQAMetadata;
  error?: string;
}

export interface VQAAnnotationRecord {
  id: string;
  videoPath: string;
  type: 'structured_vqa';
  provider: string;
  model: string;
  analysis: StructuredVQAAnalysis;
  createdAt: string;
  updatedAt: string;
}

export interface VLMProvider {
  id: string;
  name: string;
  description: string;
  capabilities: string[];
  maxFrames: number;
  apiKeyRequired: boolean;
  pricing: string;
  recommended?: boolean;
  models?: string[];
}
