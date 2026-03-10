# Embodied Data Collection and Annotation Platform

A comprehensive platform for collecting, annotating, and augmenting embodied intelligence data with support for mainstream formats and simulation environments.

## Features

### Data Collection
- **Real Robot Collection**: Connect to physical robots via WebSocket for real-time data streaming
- **Simulation Collection**: Integrated with Isaac Lab, MuJoCo, and Gazebo
- **Multi-Sensor Support**:
  - RGB/Depth/RGB-D cameras (wrist, chest, head mounted)
  - Force/Torque sensors (6-axis at wrist)
  - Tactile sensors (high-resolution pressure arrays)
  - Joint position/velocity/effort
  - IMU and LiDAR

### Data Formats
- **LeRobot** (HuggingFace)
- **RT-X** (Google Robotics Transformer X)
- **RLDS** (Reinforcement Learning Datasets)
- **Open X-Embodiment**
- **HDF5** and **ROS Bag**

### Data Annotation
- Bounding box annotation
- Polygon/mask annotation
- Keypoint annotation
- Frame-by-frame playback
- Batch verification workflow

### Auto-Annotation with VLM
- **Video Segmentation**: Automatically segment videos into meaningful actions
- **Natural Language Queries**: Ask questions about video content
- **Video Summarization**: Generate brief, detailed, structured, or instructional summaries
- **Semantic Search**: Search across episodes using natural language
- **Batch Processing**: Auto-annotate entire datasets
- **Supported Models**: GPT-4V, Claude 3 Vision, Qwen2-VL, LLaVA, InternVL

### Data Visualization
- Multi-camera video playback
- Real-time sensor data plots (Force/Torque, Joint positions)
- 3D trajectory visualization
- Tactile sensor heatmaps

### Data Augmentation
- **Cross-Embodiment Transfer**: Transfer skills between different robot types
- **Cross-Context Generation**: Generate data in different environments
- **Domain Randomization**: Improve sim-to-real transfer
- **World Models**: Integration with Emu3.5 (PKU) and Rynn-002

### Simulator Integration
- **Isaac Lab**: NVIDIA's robot learning platform
- **MuJoCo**: Multi-Joint dynamics with Contact
- **Gazebo**: ROS-compatible simulation environment

## Quick Start

### Start Backend
```bash
cd backend
npm install
npm start
```

The backend will start on `http://localhost:3001` with WebSocket support for real-time data streaming.

### Start Frontend
```bash
cd app
npm install
npm run dev
```

The frontend will be available at `http://localhost:5173`.

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Frontend      │────▶│   Backend API   │────▶│   Simulators    │
│   (React)       │     │   (Node.js)     │     │   (Isaac/Mujo)  │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                               │
                               ▼
                        ┌─────────────────┐
                        │   Database      │
                        │   (JSON/SQLite) │
                        └─────────────────┘
```

## API Endpoints

### Datasets
- `GET /api/datasets` - List all datasets
- `POST /api/datasets` - Create new dataset
- `GET /api/datasets/:id` - Get dataset details
- `GET /api/datasets/:id/episodes` - Get dataset episodes

### Collections
- `GET /api/collections` - List all collections
- `POST /api/collections` - Create new collection
- `POST /api/collections/:id/episodes` - Add episode to collection

### Annotations
- `GET /api/annotations/frame/:frameId` - Get frame annotations
- `POST /api/annotations` - Create annotation
- `PUT /api/annotations/:id` - Update annotation
- `DELETE /api/annotations/:id` - Delete annotation

### Simulators
- `GET /api/simulators` - List available simulators
- `POST /api/simulators/:id/start` - Start simulator
- `POST /api/simulators/:id/stop` - Stop simulator
- `POST /api/simulators/:id/step` - Step simulation

### Augmentation
- `GET /api/augmentation/models` - List world models
- `POST /api/augmentation/generate` - Generate augmented data
- `POST /api/augmentation/cross-embodiment` - Cross-embodiment transfer
- `POST /api/augmentation/cross-context` - Cross-context generation

### Export
- `GET /api/export/formats` - List export formats
- `POST /api/export/dataset/:id` - Export dataset

### Auto-Annotation (VLM)
- `GET /api/autoannotation/models` - List available VLM models
- `POST /api/autoannotation/segment` - Segment video into actions
- `POST /api/autoannotation/query` - Natural language query on video
- `POST /api/autoannotation/batch` - Batch auto-annotate episodes
- `POST /api/autoannotation/search` - Semantic video search
- `POST /api/autoannotation/summarize` - Generate video summary
- `POST /api/autoannotation/compare` - Compare multiple videos

## WebSocket Protocol

Connect to `ws://localhost:3001` for real-time data streaming.

### Client Registration
```json
{
  "type": "register",
  "clientType": "collector|visualization"
}
```

### Sensor Data
```json
{
  "type": "sensor_data",
  "sensorType": "rgb|depth|force_torque|...",
  "timestamp": 1234567890,
  "data": {...}
}
```

## Configuration

### Sensor Configurations

Default configurations are provided for:
- **Bimanual Manipulation**: Dual arm with wrist cameras and F/T sensors
- **Humanoid Full Body**: Multi-camera setup with tactile sensors
- **Mobile Manipulation**: Base with arm and LiDAR

### Simulator Scenes

Available scenes:
- Tabletop Manipulation
- Kitchen Environment
- Factory Floor
- Office Space

## Data Pipeline

```
Collection → Storage → Annotation → Augmentation → Export
    │            │           │            │           │
    ▼            ▼           ▼            ▼           ▼
  Sensors    Dataset    Bounding    Emu3.5/    LeRobot
  (Real/     (Episodes/  Box/Mask   Rynn-002   RT-X/
   Sim)       Frames)    /Keypoint              RLDS
```

## Development

### Project Structure
```
├── app/                    # Frontend React application
│   ├── src/
│   │   ├── sections/       # Page components
│   │   ├── services/       # API and WebSocket services
│   │   ├── types/          # TypeScript type definitions
│   │   └── components/ui/  # shadcn/ui components
├── backend/                # Backend Node.js application
│   └── src/
│       ├── routes/         # API route handlers
│       ├── models/         # Database models
│       └── server.js       # Main server entry
└── database/               # Data storage directory
```

### Adding New Simulators

1. Add simulator config to `backend/src/models/Database.js`
2. Implement simulator-specific API in `backend/src/routes/simulators.js`
3. Add UI components in `app/src/sections/SimulatorControl.tsx`

### Adding New World Models

1. Add model info to `backend/src/routes/augmentation.js`
2. Implement generation logic
3. Update UI in `app/src/sections/DataAugmentation.tsx`

## License

MIT License
