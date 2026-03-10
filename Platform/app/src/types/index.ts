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

// ========== Visual Grounding Types ==========

/** Bounding box in normalized coordinates [0,1] relative to frame dimensions */
export interface GroundingBBox {
  x: number;      // top-left x (0-1)
  y: number;      // top-left y (0-1)
  width: number;  // box width (0-1)
  height: number; // box height (0-1)
  label?: string; // what this box represents
}

/** A single visual evidence anchor — ties a VQA claim to specific frame(s) + regions */
export interface VisualGrounding {
  frame_indices: number[];        // which frames support this claim
  timestamps: string[];           // human-readable timestamps (e.g. "2.1s")
  bboxes?: GroundingBBox[];       // spatial regions of interest (per frame or shared)
  description: string;            // what the visual evidence shows
  confidence: number;             // 0-1, how confident the grounding is
}

// ========== Structured VQA Types ==========

export interface VQAActionItem {
  action: string;
  timestamp: string;
  frame_range: [number, number];
  description: string;
  grounding: VisualGrounding;     // ← grounded in visual evidence
}

export interface VQATemporal {
  action_sequence: VQAActionItem[];
  relationships: string[];
}

export interface VQASpatialRelationship {
  timestamp: string;
  relationship: string;
  details: string;
  grounding: VisualGrounding;     // ← grounded in visual evidence
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
  grounding: VisualGrounding;     // ← grounded: where this object appears
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
  grounding: VisualGrounding;     // ← grounded: contact visual evidence
}

export interface VQAMechanics {
  contacts: VQAContact[];
  force_profile: string;
}

export interface VQAActionJustification {
  action: string;
  reason: string;
  constraints: string[];
  grounding: VisualGrounding;     // ← grounded: what evidence supports this reasoning
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
  grounding_start: VisualGrounding;  // ← grounded: initial state evidence
  grounding_end: VisualGrounding;    // ← grounded: final state evidence
}

export interface VQAMotionSegment {
  segment: string;
  time_range: string;
  motion_type: 'linear' | 'curved' | 'rotational';
  velocity: 'slow' | 'medium' | 'fast';
  waypoints: string[];
  grounding: VisualGrounding;     // ← grounded: trajectory visual evidence
}

export interface VQATrajectory {
  motion_segments: VQAMotionSegment[];
  overall_path: string;
}

export interface VQAKeyFrame {
  frame_idx: number;
  timestamp: string;
  significance: string;
  thumbnail_url?: string;         // ← optional thumbnail for display
  bboxes?: GroundingBBox[];       // ← highlighted regions in this keyframe
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
  /** URLs/paths to extracted frame images for grounding display */
  frame_image_urls?: string[];
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

/** Cross-category temporal consistency check result */
export interface TemporalConsistencyCheck {
  consistent: boolean;
  conflicts: Array<{
    category_a: string;
    category_b: string;
    claim_a: string;
    claim_b: string;
    timestamp_a: string;
    timestamp_b: string;
    description: string;
  }>;
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
  temporal_consistency?: TemporalConsistencyCheck;  // ← cross-category consistency
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
