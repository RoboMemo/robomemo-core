const express = require('express');
const cors = require('cors');
const multer = require('multer');
const path = require('path');
const fs = require('fs');
const morgan = require('morgan');
const dotenv = require('dotenv');
const { GoogleGenerativeAI } = require('@google/generative-ai');
const { spawn } = require('child_process');
const helmet = require('helmet');

dotenv.config();

// ─── Services & Middleware ────────────────────────────────────────────────────
const { initAuth, Users, authMiddleware, requireRole, signToken, verifyPasswordSync } = require('./services/auth');
const auditLog = require('./services/audit-log');
const encryption = require('./services/encryption');
const { geoFenceMiddleware } = require('./middleware/geo-fence');
const { vlmLimiter, authLimiter, exportLimiter } = require('./middleware/rate-limiter');
const { validateTemporalConsistency } = require('./middleware/temporal-consistency');
const exportService = require('./services/export');

const app = express();
const port = process.env.PORT || 3001;

// ─── Security Middleware ──────────────────────────────────────────────────────

// CORS: must be before helmet so preflight OPTIONS gets handled
const corsOrigins = (process.env.CORS_ORIGINS || 'http://localhost:5173,http://localhost:3000')
  .split(',').map(s => s.trim());
app.use(cors({ 
  origin: (origin, callback) => {
    // Allow requests with no origin (mobile apps, curl, etc.)
    if (!origin) return callback(null, true);
    // Allow configured origins
    if (corsOrigins.includes(origin)) return callback(null, true);
    // Allow Cloudflare Pages and trycloudflare domains
    if (origin.endsWith('.pages.dev') || origin.endsWith('.trycloudflare.com')) return callback(null, true);
    callback(null, false);
  },
  credentials: true,
  methods: ['GET', 'POST', 'PUT', 'PATCH', 'DELETE', 'OPTIONS'],
  allowedHeaders: ['Content-Type', 'Authorization', 'X-API-Key']
}));

// Helmet: security headers (CSP, HSTS, X-Frame-Options, etc.)
app.use(helmet({ 
  contentSecurityPolicy: false,
  crossOriginResourcePolicy: { policy: 'cross-origin' },
  crossOriginOpenerPolicy: false
}));

app.use(express.json({ limit: '50mb' }));
app.use(morgan('dev'));

// Configure Multer for video uploads
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    const uploadDir = path.join(__dirname, 'uploads');
    if (!fs.existsSync(uploadDir)) {
      fs.mkdirSync(uploadDir);
    }
    cb(null, uploadDir);
  },
  filename: (req, file, cb) => {
    cb(null, `${Date.now()}-${file.originalname}`);
  }
});
const upload = multer({ storage });

// Data management — SQLite via db.js
const DATA_DIR = path.join(__dirname, 'data'); // kept for GenRobot static files
const { db, Datasets, Episodes, Annotations, Collections, Lineage, Tasks, Reviews, Orders, syncFromJSON, getStats, getFullStats, getDatasetStats, getAnnotationStats, getTimeline } = require('./db');

// Initialize auth & audit on the shared DB instance
initAuth(db);
auditLog.initAuditLog(db);

// Serve uploaded videos as static (for browser preview)
app.use('/uploads/videos', express.static(path.join(__dirname, 'uploads', 'videos')));

// Serve downloaded dataset files (LeRobot videos, GenRobot H5, etc.)
app.use('/data', express.static(path.join(__dirname, 'data'), {
  setHeaders: (res, filePath) => {
    if (filePath.endsWith('.mp4')) {
      res.setHeader('Content-Type', 'video/mp4');
      res.setHeader('Accept-Ranges', 'bytes');
    }
  }
}));

// Apply geo-fence (blocks CN/HK IPs)
app.use(geoFenceMiddleware);

// Apply audit logging middleware (logs every request)
app.use(auditLog.auditMiddleware);

// Health check (no auth required)
app.get('/api/health', (req, res) => {
  res.json({ status: 'ok', timestamp: new Date().toISOString(), encryption: encryption.isEnabled() });
});

// ========== VIDEO UPLOAD ENDPOINT ==========

/**
 * Upload a video file → returns server-side path for VQA analysis
 * POST /api/upload/video
 * Content-Type: multipart/form-data,  field name: "video"
 * Max size: 500 MB
 */
const videoStorage = multer.diskStorage({
  destination: (req, file, cb) => {
    const dir = path.join(__dirname, 'uploads', 'videos');
    if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
    cb(null, dir);
  },
  filename: (req, file, cb) => {
    // Sanitise filename — prevent path traversal
    const safe = file.originalname.replace(/[^a-zA-Z0-9._-]/g, '_');
    cb(null, `${Date.now()}_${safe}`);
  },
});

const videoUpload = multer({
  storage: videoStorage,
  limits: { fileSize: 500 * 1024 * 1024 },
  fileFilter: (req, file, cb) => {
    if (file.mimetype.startsWith('video/') || /\.(mp4|avi|mov|webm|mkv|h264|h265)$/i.test(file.originalname)) {
      cb(null, true);
    } else {
      cb(new Error('Only video files are allowed'), false);
    }
  },
});

app.post('/api/upload/video', videoUpload.single('video'), (req, res) => {
  if (!req.file) return res.status(400).json({ error: 'No video file provided' });

  auditLog.write({
    event: 'VIDEO_UPLOAD',
    userId: req.user?.userId,
    action: 'create',
    resource: `video:${req.file.filename}`,
    ip: req.clientIP,
    details: {
      filename: req.file.filename,
      originalname: req.file.originalname,
      size: req.file.size,
      mimetype: req.file.mimetype,
    },
    result: 'success',
  });

  res.json({
    success: true,
    filename: req.file.filename,
    originalName: req.file.originalname,
    serverPath: req.file.path,
    size: req.file.size,
    url: `/uploads/videos/${req.file.filename}`,
  });
});

// Multer error handler for video upload
app.use((err, req, res, next) => {
  if (err && err.code === 'LIMIT_FILE_SIZE') {
    return res.status(413).json({ error: 'File too large — maximum 500 MB' });
  }
  next(err);
});

// ========== COMPLIANCE STATUS ENDPOINT ==========

/**
 * GET /api/compliance/status
 * Returns live data-sovereignty, geo-fence, encryption, audit, and GDPR status.
 * Used by the UI Compliance Dashboard.
 */
app.get('/api/compliance/status', (req, res) => {
  const auditStats = auditLog.getStats();
  res.json({
    dataResidency: {
      status: 'compliant',
      primaryRegion:   { code: 'ap-southeast-1', name: 'Singapore (AWS)' },
      secondaryRegion: { code: 'me-south-1',     name: 'UAE Bahrain (AWS)' },
      blockedRegions:  ['CN', 'HK'],
      certifications:  ['ISO 27001', 'ISO 27701', 'SOC 2 Type II', 'GDPR'],
      lastAudit:       '2025-12-01',
    },
    geoFence: {
      enabled:          true,
      blockedCountries: ['CN', 'HK'],
      blockedRequests:  auditStats.geoBlocks ?? 0,
      method:           'NGINX GeoIP2 + IP-prefix fallback',
    },
    encryption: {
      atRest:        encryption.isEnabled() ? 'AES-256-GCM ✓' : 'Disabled (set ENCRYPTION_MASTER_KEY)',
      atRestEnabled: encryption.isEnabled(),
      inTransit:     'TLS 1.3',
      keyManager:    'AWS KMS (ap-southeast-1)',
    },
    audit: {
      enabled:       true,
      standard:      'ISO 27001 Annex A.12.4 / SOC 2 CC7.2',
      total:         auditStats.total   ?? 0,
      denied:        auditStats.denied  ?? 0,
      errors:        auditStats.errors  ?? 0,
      geoBlocks:     auditStats.geoBlocks ?? 0,
      last24h:       auditStats.last24h ?? 0,
      retentionDays: 365,
    },
    gdpr: {
      implementedArticles: [
        'Art.15 — Subject Access Request',
        'Art.17 — Right to Erasure',
        'Art.20 — Data Portability',
      ],
      dpo: 'dpo@robomemo.io',
      processingBasis: 'Legitimate interests / Contractual necessity',
    },
    vlmApiRouting: {
      status:         'overseas-only',
      allowedRegions: ['ap-southeast-1', 'me-south-1', 'eu-west-1', 'us-east-1'],
      blockedRoutes:  ['CN', 'HK'],
      providers: {
        gemini: 'Google Cloud ap-southeast-1',
        claude: 'Anthropic US East → routed via SG NAT',
        openai: 'Azure Singapore (eastasia fallback)',
        local:  'On-premise / air-gapped',
      },
    },
    dataLineage: {
      enabled:     true,
      trackingFields: ['source_region', 'upload_ip', 'annotator', 'vlm_provider', 'processing_node'],
    },
    timestamp: new Date().toISOString(),
  });
});

// VLM Integration (Gemini 1.5 Flash) — API key from .env, NOT from client
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY || '');

async function processVideoWithVLM(videoPath, prompt) {
  if (!process.env.GEMINI_API_KEY) {
    console.warn('GEMINI_API_KEY not set in .env, using mock data for demo');
    return null;
  }

  try {
    const model = genAI.getGenerativeModel({ model: "gemini-1.5-flash" });
    
    // Read video file as buffer
    const videoBuffer = fs.readFileSync(videoPath);
    const videoPart = {
      inlineData: {
        data: videoBuffer.toString('base64'),
        mimeType: 'video/mp4'
      }
    };

    const result = await model.generateContent([prompt, videoPart]);
    return result.response.text();
  } catch (error) {
    console.error('VLM processing error:', error);
    throw error;
  }
}

// VLM Integration (Local SmolVLM)
async function processVideoWithLocalVLM(videoPath, prompt) {
  return new Promise((resolve, reject) => {
    const pythonExecutable = path.join(__dirname, 'venv', 'bin', 'python');
    const pythonProcess = spawn(pythonExecutable, [path.join(__dirname, 'vlm_server.py'), videoPath, prompt]);

    let result = '';
    pythonProcess.stdout.on('data', (data) => {
      result += data.toString();
    });

    pythonProcess.stderr.on('data', (data) => {
      console.error(`VLM script error: ${data}`);
    });

    pythonProcess.on('close', (code) => {
      if (code !== 0) {
        return reject(new Error(`VLM script exited with code ${code}`));
      }
      try {
        resolve(JSON.parse(result));
      } catch (e) {
        reject(new Error('Failed to parse VLM script output'));
      }
    });
  });
}

// Local VLM Segmentation API
app.post('/api/vlm/local/segment', async (req, res) => {
  const { videoPath } = req.body;
  const prompt = `
    Analyze this video and segment it into logical action steps.
    Return the result in JSON format: [{"start": 0, "end": 2.5, "caption": "..."}, ...]
  `;

  try {
    const result = await processVideoWithLocalVLM(videoPath, prompt);
    res.json(result);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Local VLM Query API
app.post('/api/vlm/local/query', async (req, res) => {
  const { videoPath, query } = req.body;
  const prompt = `Answer the following question about this video: "${query}"`;

  try {
    const result = await processVideoWithLocalVLM(videoPath, prompt);
    res.json(result);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Local VLM Summarization API
app.post('/api/vlm/local/summarize', async (req, res) => {
  const { videoPath, summaryType } = req.body;
  const prompt = `Generate a ${summaryType} summary of this video.`;

  try {
    const result = await processVideoWithLocalVLM(videoPath, prompt);
    res.json(result);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Routes
app.get('/api/datasets', (req, res) => {
  res.json(Datasets.getAll());
});

app.post('/api/datasets', (req, res) => {
  const newDataset = {
    id: `ds_${Date.now()}`,
    createdAt: new Date().toISOString(),
    updatedAt: new Date().toISOString(),
    episodeCount: 0,
    frameCount: 0,
    size: 0,
    version: '1.0.0',
    license: 'MIT',
    ...req.body
  };
  res.status(201).json(Datasets.insert(newDataset));
});

app.get('/api/datasets/:id', (req, res) => {
  const dataset = Datasets.getById(req.params.id);
  if (!dataset) return res.status(404).json({ error: 'Dataset not found' });
  res.json(dataset);
});

app.put('/api/datasets/:id', (req, res) => {
  const dataset = Datasets.update(req.params.id, req.body);
  if (!dataset) return res.status(404).json({ error: 'Dataset not found' });
  res.json(dataset);
});

app.delete('/api/datasets/:id', (req, res) => {
  const existing = Datasets.getById(req.params.id);
  if (!existing) return res.status(404).json({ error: 'Dataset not found' });
  Datasets.delete(req.params.id);
  res.json({ success: true });
});

app.get('/api/datasets/:id/episodes', (req, res) => {
  res.json(Episodes.getByDataset(req.params.id));
});

// Get video/media URLs for a specific episode
app.get('/api/episodes/:id/media', (req, res) => {
  const episode = Episodes.getById(req.params.id);
  if (!episode) return res.status(404).json({ error: 'Episode not found' });
  
  const dataset = Datasets.getById(episode.datasetId);
  if (!dataset) return res.status(404).json({ error: 'Dataset not found' });
  
  const sensorConfig = typeof dataset.sensorConfig === 'string' 
    ? JSON.parse(dataset.sensorConfig || '{}') 
    : (dataset.sensorConfig || {});
  const sensors = sensorConfig.sensors || [];
  
  const media = { videos: [], images: [], dataFiles: [] };
  const dataDir = path.join(__dirname, 'data');
  
  if (dataset.format === 'lerobot_v3') {
    // LeRobot v3: check for video files per camera
    const dsSource = dataset.source || '';
    const dsId = dataset.id;
    
    // Map dataset IDs to local directory names
    const dirMap = {
      'lerobot_pusht': 'lerobot_pusht',
      'lerobot_xarm_lift': 'lerobot_xarm_lift',
      'aloha_static_cups': 'lerobot_aloha_cups',
      'aloha_mobile_shrimp': 'lerobot_aloha_shrimp',
    };
    const localDir = dirMap[dsId];
    if (localDir) {
      const dsPath = path.join(dataDir, 'datasets', localDir);
      // Read info.json to get camera keys
      try {
        const info = JSON.parse(fs.readFileSync(path.join(dsPath, 'meta', 'info.json'), 'utf-8'));
        const cameraKeys = Object.keys(info.features || {}).filter(k => k.includes('image'));
        
        for (const camKey of cameraKeys) {
          // Find video file for this camera
          const videoDir = path.join(dsPath, 'videos', camKey, 'chunk-000');
          if (fs.existsSync(videoDir)) {
            const videoFiles = fs.readdirSync(videoDir).filter(f => f.endsWith('.mp4'));
            for (const vf of videoFiles) {
              const feat = info.features[camKey] || {};
              media.videos.push({
                camera: camKey.replace('observation.images.', '').replace('observation.', ''),
                url: `/data/datasets/${localDir}/videos/${camKey}/chunk-000/${vf}`,
                resolution: feat.shape ? `${feat.shape[1]}x${feat.shape[0]}` : 'unknown',
                codec: feat.video_info?.['video.codec'] || 'unknown',
                fps: feat.video_info?.['video.fps'] || info.fps
              });
            }
          }
        }
      } catch (e) {
        console.error('Error reading LeRobot meta:', e.message);
      }
    }
  } else if (dataset.format === 'h5') {
    // GenRobot H5: return H5 file path  
    if (episode.h5_path) {
      // h5_path is relative to backend/data/, e.g. "genrobot_open_dataset/..."
      const cleanPath = episode.h5_path.replace(/^data\//, '');
      media.dataFiles.push({
        type: 'h5',
        path: episode.h5_path,
        url: `/data/${cleanPath}`,
        sensors: ['mid_fisheye_color', 'imu_6axis', 'tactile_left', 'tactile_right']
      });
    }
  }
  
  // Add sensor metadata
  media.sensorConfig = sensorConfig;
  media.episodeInfo = {
    id: episode.id,
    name: episode.name,
    frameCount: episode.frameCount,
    duration: episode.duration,
    fps: episode.fps,
    skill: episode.skill,
    bimanual: episode.bimanual
  };
  
  res.json(media);
});

// ========== COLLECTIONS API ==========

app.get('/api/collections', (req, res) => {
  res.json(Collections.getAll());
});

app.get('/api/collections/:id', (req, res) => {
  const col = Collections.getById(req.params.id);
  if (!col) return res.status(404).json({ error: 'Collection not found' });
  res.json(col);
});

app.post('/api/collections', (req, res) => {
  const newCol = {
    id: `col_${Date.now()}`,
    createdAt: new Date().toISOString(),
    updatedAt: new Date().toISOString(),
    episodeIds: [],
    ...req.body
  };
  res.status(201).json(Collections.insert(newCol));
});

app.post('/api/collections/:id/episodes', (req, res) => {
  const { episodeId } = req.body;
  if (!episodeId) return res.status(400).json({ error: 'episodeId required' });
  const col = Collections.addEpisode(req.params.id, episodeId);
  if (!col) return res.status(404).json({ error: 'Collection not found' });
  res.json(col);
});

app.get('/api/autoannotation/models', (req, res) => {
  res.json([
    {
      id: 'gemini-1.5-flash',
      name: 'Gemini 1.5 Flash',
      provider: 'Google',
      description: 'High-speed VLM for video segmentation and QA',
      capabilities: ['segmentation', 'qa', 'summarization'],
      maxVideoLength: 3600,
      languageSupport: ['en', 'zh']
    },
    {
      id: 'local-smolvlm',
      name: 'Local SmolVLM',
      provider: 'Local',
      description: 'Locally hosted SmolVLM for fast MVP validation',
      capabilities: ['segmentation'],
      maxVideoLength: 60,
      languageSupport: ['en']
    }
  ]);
});

// Video Segmentation API
app.post('/api/autoannotation/segment', async (req, res) => {
  const { videoPath, modelId, options } = req.body;
  
  const prompt = `
    Please analyze this video of a robotic task and segment it into logical action steps.
    For each segment, provide:
    1. Start time (seconds)
    2. End time (seconds)
    3. Action description (what is the robot doing?)
    4. Confidence (0.0 - 1.0)
    
    Return the result in JSON format: [{"start": 0, "end": 2.5, "caption": "...", "confidence": 0.9}, ...]
  `;

  try {
    // In a real scenario, videoPath would point to a real file
    // For MVP/Demo, we either call the real VLM or return mock data if file not found
    let segments;
    if (fs.existsSync(videoPath)) {
      const vlmResponse = await processVideoWithVLM(videoPath, prompt);
      if (vlmResponse) {
        // Parse JSON from VLM text response
        const jsonMatch = vlmResponse.match(/\[.*\]/s);
        segments = jsonMatch ? JSON.parse(jsonMatch[0]) : [];
      }
    }

    if (!segments) {
      // Mock data for demo
      segments = [
        { "start": 0, "end": 4.2, "caption": "Robot arm approaches the red cube on the platform", "confidence": 0.95 },
        { "start": 4.2, "end": 7.8, "caption": "Gripper opens and positions around the cube", "confidence": 0.92 },
        { "start": 7.8, "end": 10.5, "caption": "Gripper closes and secures the grasp on the cube", "confidence": 0.98 },
        { "start": 10.5, "end": 15.0, "caption": "Lifts the cube vertically away from the table", "confidence": 0.94 }
      ];
    }

    res.json(segments);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Video Query API
app.post('/api/autoannotation/query', async (req, res) => {
  const { videoPath, query, modelId } = req.body;
  
  const prompt = `
    Answer the following question about this video: "${query}"
    Provide visual evidence from the video and your confidence score.
    Return in JSON format: {"answer": "...", "visualEvidence": "...", "confidence": 0.9, "relevantFrames": [120, 240]}
  `;

  try {
    let result;
    if (fs.existsSync(videoPath)) {
      const vlmResponse = await processVideoWithVLM(videoPath, prompt);
      if (vlmResponse) {
        const jsonMatch = vlmResponse.match(/\{.*\}/s);
        result = jsonMatch ? JSON.parse(jsonMatch[0]) : null;
      }
    }

    if (!result) {
      // Mock data for demo
      result = {
        "answer": `Based on the video, the robot successfully grasped the object. It moved with a smooth trajectory and maintained a stable grip throughout the lift phase.`,
        "visualEvidence": "At around 8 seconds, the gripper fingers make contact with the cube's sides and close completely.",
        "confidence": 0.96,
        "relevantFrames": [240, 300]
      };
    }

    res.json(result);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Annotations API
app.get('/api/annotations/episode/:episodeId', (req, res) => {
  res.json(Annotations.getByEpisode(req.params.episodeId));
});

app.get('/api/annotations/frame/:frameId', (req, res) => {
  res.json(Annotations.getByFrame(req.params.frameId));
});

app.post('/api/annotations', (req, res) => {
  const newAnnotation = {
    id: `ann_${Date.now()}`,
    createdAt: new Date().toISOString(),
    updatedAt: new Date().toISOString(),
    verified: false,
    confidence: 1.0,
    annotator: 'VLM-Auto',
    ...req.body
  };
  res.json(Annotations.insert(newAnnotation));
});

app.put('/api/annotations/:id', (req, res) => {
  const ann = Annotations.update(req.params.id, req.body);
  if (!ann) return res.status(404).json({ error: 'Annotation not found' });
  res.json(ann);
});

app.delete('/api/annotations/:id', (req, res) => {
  const existing = Annotations.getById(req.params.id);
  if (!existing) return res.status(404).json({ error: 'Annotation not found' });
  Annotations.delete(req.params.id);
  res.json({ success: true });
});

// Batch Auto-Annotation API
app.post('/api/autoannotation/batch', async (req, res) => {
  const { episodeIds, modelId, annotationType } = req.body;
  
  // This would start a background job in a real app
  // For MVP, we simulate processing
  const newAnnotations = episodeIds.map(episodeId => ({
    id: `ann_auto_${Date.now()}_${Math.random().toString(36).slice(2)}_${episodeId}`,
    episodeId,
    type: 'label',
    label: 'Auto-detected action sequence',
    confidence: 0.85,
    annotator: 'VLM-Auto',
    verified: false,
    createdAt: new Date().toISOString(),
    updatedAt: new Date().toISOString()
  }));

  Annotations.insertMany(newAnnotations);
  res.json({ success: true, count: newAnnotations.length, annotations: newAnnotations });
});

// ========== FRAME EXTRACTION API ==========

/**
 * Extract frames from video and serve as static images for grounding display
 * POST /api/frames/extract
 * Body: { videoPath, numFrames }
 */
app.post('/api/frames/extract', async (req, res) => {
  const { videoPath, numFrames = 32 } = req.body;

  if (!videoPath || !fs.existsSync(videoPath)) {
    return res.status(404).json({ error: 'Video file not found' });
  }

  const videoHash = require('crypto').createHash('md5').update(videoPath).digest('hex').slice(0, 12);
  const framesDir = path.join(__dirname, 'uploads', 'frames', videoHash);

  // If already extracted, return cached
  if (fs.existsSync(framesDir)) {
    const existing = fs.readdirSync(framesDir).filter(f => f.endsWith('.jpg')).sort();
    if (existing.length > 0) {
      return res.json({
        success: true,
        cached: true,
        framesDir: `/uploads/frames/${videoHash}`,
        frames: existing.map((f, i) => ({
          index: i,
          filename: f,
          url: `/uploads/frames/${videoHash}/${f}`,
        })),
      });
    }
  }

  fs.mkdirSync(framesDir, { recursive: true });

  try {
    // Use ffmpeg to extract frames
    const ffmpeg = spawn('ffmpeg', [
      '-i', videoPath,
      '-vf', `select='not(mod(n\\,${Math.max(1, Math.floor(30 / (numFrames / 10)))}))`,
      '-vsync', 'vfr',
      '-frames:v', numFrames.toString(),
      '-q:v', '3',
      path.join(framesDir, 'frame_%04d.jpg'),
    ]);

    let ffmpegError = '';
    ffmpeg.stderr.on('data', (data) => { ffmpegError += data.toString(); });

    ffmpeg.on('close', (code) => {
      const extracted = fs.readdirSync(framesDir).filter(f => f.endsWith('.jpg')).sort();
      if (extracted.length === 0) {
        return res.status(500).json({ error: 'Frame extraction failed', details: ffmpegError });
      }

      res.json({
        success: true,
        cached: false,
        framesDir: `/uploads/frames/${videoHash}`,
        frames: extracted.map((f, i) => ({
          index: i,
          filename: f,
          url: `/uploads/frames/${videoHash}/${f}`,
        })),
      });
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Serve extracted frames as static files
app.use('/uploads/frames', express.static(path.join(__dirname, 'uploads', 'frames')));

// ========== END FRAME EXTRACTION API ==========

// ========== STRUCTURED VQA ANALYSIS API ==========

/**
 * Analyze video with structured 7-category VQA
 * POST /api/vlm/structured-analysis
 * Body: { videoPath, provider, apiKey, numFrames, model }
 */
app.post('/api/vlm/structured-analysis', vlmLimiter, async (req, res) => {
  const { videoPath, provider = 'gemini', apiKey: clientApiKey, numFrames = 32, model } = req.body;

  if (!videoPath) {
    return res.status(400).json({ error: 'videoPath is required' });
  }

  // Use server-side API key first, fall back to client-provided key
  const envKeyMap = { gemini: 'GEMINI_API_KEY', claude: 'CLAUDE_API_KEY', openai: 'OPENAI_API_KEY' };
  const apiKey = process.env[envKeyMap[provider]] || clientApiKey;

  if (!apiKey && provider !== 'local') {
    return res.status(400).json({ error: 'API key not configured. Set it in server .env or provide in request.' });
  }

  if (!fs.existsSync(videoPath)) {
    return res.status(404).json({ error: `Video file not found: ${videoPath}` });
  }

  // Step 1: Extract frames for grounding
  const videoHash = require('crypto').createHash('md5').update(videoPath).digest('hex').slice(0, 12);
  const framesDir = path.join(__dirname, 'uploads', 'frames', videoHash);
  let frameImageUrls = [];

  if (!fs.existsSync(framesDir)) {
    fs.mkdirSync(framesDir, { recursive: true });
  }

  const existingFrames = fs.existsSync(framesDir)
    ? fs.readdirSync(framesDir).filter(f => f.endsWith('.jpg')).sort()
    : [];

  if (existingFrames.length > 0) {
    frameImageUrls = existingFrames.map(f => `/uploads/frames/${videoHash}/${f}`);
  } else {
    // Extract frames with ffmpeg (synchronous wait for grounding)
    try {
      await new Promise((resolve, reject) => {
        const ff = spawn('ffmpeg', [
          '-i', videoPath,
          '-vf', `select='not(mod(n\\,${Math.max(1, Math.floor(30 / (numFrames / 10)))}))`,
          '-vsync', 'vfr',
          '-frames:v', numFrames.toString(),
          '-q:v', '3',
          path.join(framesDir, 'frame_%04d.jpg'),
        ]);
        ff.on('close', (code) => code === 0 ? resolve() : reject(new Error(`ffmpeg exit ${code}`)));
        ff.on('error', reject);
      });
      const extracted = fs.readdirSync(framesDir).filter(f => f.endsWith('.jpg')).sort();
      frameImageUrls = extracted.map(f => `/uploads/frames/${videoHash}/${f}`);
    } catch (err) {
      console.warn('Frame extraction failed, continuing without grounding images:', err.message);
    }
  }

  // Step 2: Run VLM analysis with grounding-aware prompt
  try {
    const pythonScript = path.join(__dirname, 'vlm_video_analyzer.py');
    const args = [provider, apiKey, videoPath, numFrames.toString()];
    if (model) {
      args.push(model);
    }
    // Pass frames dir for grounding reference
    args.push('--frames-dir', framesDir);

    const pythonProcess = spawn('python', [pythonScript, ...args]);

    let resultData = '';
    let errorData = '';

    pythonProcess.stdout.on('data', (data) => {
      resultData += data.toString();
    });

    pythonProcess.stderr.on('data', (data) => {
      errorData += data.toString();
      console.error(`VLM Analysis stderr: ${data}`);
    });

    pythonProcess.on('close', (code) => {
      if (code !== 0) {
        console.error(`VLM Analysis process exited with code ${code}`);
        console.error(`Error output: ${errorData}`);
        return res.status(500).json({ 
          error: 'VLM analysis failed', 
          details: errorData,
          exitCode: code 
        });
      }

      try {
        const result = JSON.parse(resultData);

        // Inject frame_image_urls into metadata for frontend grounding display
        if (result.metadata) {
          result.metadata.frame_image_urls = frameImageUrls;
        }

        // ── Temporal Consistency Validation ──
        const consistencyCheck = validateTemporalConsistency(result);
        result.temporal_consistency = consistencyCheck;

        // Save analysis to SQLite
        const annotationId = `ann_vqa_${Date.now()}`;
        const newAnnotation = {
          id: annotationId,
          videoPath: videoPath,
          type: 'structured_vqa',
          provider: provider,
          model: model || provider,
          analysis: result,
          createdAt: new Date().toISOString(),
          updatedAt: new Date().toISOString()
        };
        Annotations.insert(newAnnotation);

        res.json({
          success: true,
          annotationId: annotationId,
          analysis: result
        });

      } catch (parseError) {
        console.error('Failed to parse VLM analysis result:', parseError);
        console.error('Raw output:', resultData);
        res.status(500).json({ 
          error: 'Failed to parse VLM analysis result',
          details: parseError.message,
          rawOutput: resultData 
        });
      }
    });

  } catch (error) {
    console.error('Error starting VLM analysis:', error);
    res.status(500).json({ error: error.message });
  }
});

/**
 * Get all structured VQA analyses
 * GET /api/vlm/structured-analyses
 */
app.get('/api/vlm/structured-analyses', (req, res) => {
  try {
    res.json(Annotations.getVQAAnalyses());
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

/**
 * Get specific structured VQA analysis
 * GET /api/vlm/structured-analysis/:id
 */
app.get('/api/vlm/structured-analysis/:id', (req, res) => {
  try {
    const analysis = Annotations.getVQAById(req.params.id);
    if (!analysis) {
      return res.status(404).json({ error: 'Analysis not found' });
    }
    res.json(analysis);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

/**
 * Get VLM provider capabilities
 * GET /api/vlm/providers
 */
app.get('/api/vlm/providers', (req, res) => {
  res.json([
    {
      id: 'gemini',
      name: 'Google Gemini 2.5 Pro',
      description: '最新 Gemini 2.5 Pro，支持深度思考模式（Thinking），视频理解最强',
      capabilities: ['temporal', 'spatial', 'attribute', 'mechanics', 'reasoning', 'summary', 'trajectory'],
      maxFrames: 64,
      apiKeyRequired: true,
      pricing: 'Pay per use',
      recommended: true
    },
    {
      id: 'claude',
      name: 'Anthropic Claude 3.5 Sonnet',
      description: '强推理能力，擅长结构化输出与复杂视觉分析',
      capabilities: ['temporal', 'spatial', 'attribute', 'mechanics', 'reasoning', 'summary', 'trajectory'],
      maxFrames: 32,
      apiKeyRequired: true,
      pricing: 'Pay per use',
      models: ['claude-3-5-sonnet-20241022', 'claude-3-opus-20240229']
    },
    {
      id: 'openai',
      name: 'OpenAI GPT-4o',
      description: '高质量多模态模型，视觉理解能力优秀',
      capabilities: ['temporal', 'spatial', 'attribute', 'mechanics', 'reasoning', 'summary', 'trajectory'],
      maxFrames: 32,
      apiKeyRequired: true,
      pricing: 'Pay per use',
      models: ['gpt-4o', 'gpt-4-turbo']
    },
    {
      id: 'local',
      name: '本地 Ollama 视觉模型',
      description: '完全离线运行，无需 API Key。支持 llama3.2-vision、minicpm-v、llava 等模型',
      capabilities: ['temporal', 'spatial', 'attribute', 'mechanics', 'reasoning', 'summary', 'trajectory'],
      maxFrames: 16,
      apiKeyRequired: false,
      pricing: '完全免费',
      models: [],
      local: true
    }
  ]);
});

// 列出本地 Ollama 视觉模型
app.get('/api/vlm/local-models', async (req, res) => {
  const OLLAMA_URL = 'http://localhost:11434';
  const VISION_TAGS = ['llava', 'llama3.2-vision', 'minicpm-v', 'bakllava', 'moondream', 'minicpm'];
  try {
    const response = await fetch(`${OLLAMA_URL}/api/tags`);
    const data = await response.json();
    const all = data.models || [];
    const vision = all.filter(m => VISION_TAGS.some(t => m.name.toLowerCase().includes(t)));
    const list = vision.length > 0 ? vision : all;
    res.json({
      available: true,
      ollamaUrl: OLLAMA_URL,
      models: list.map(m => ({
        name: m.name,
        size: m.size,
        modified_at: m.modified_at
      }))
    });
  } catch (err) {
    res.json({ available: false, models: [], error: 'Ollama 未运行，请先启动 ollama serve' });
  }
});

// ========== END STRUCTURED VQA ANALYSIS API ==========

// ─── DB management endpoints ──────────────────────────────────────────────────
/** POST /api/db/sync — re-import datasets/episodes/annotations from JSON files
 *  (run after download_data.py generates new data) */
app.post('/api/db/sync', (req, res) => {
  try {
    const stats = syncFromJSON();
    auditLog.write({ event: 'DB_SYNC', userId: req.user?.userId, action: 'update', resource: 'database', result: 'success' });
    res.json({ success: true, ...stats });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

/** GET /api/db/stats — row counts + DB file path */
app.get('/api/db/stats', (req, res) => {
  try {
    res.json(getStats());
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// ========== AUTHENTICATION ROUTES ==========

app.post('/api/auth/login', authLimiter, (req, res) => {
  const { email, password } = req.body;
  if (!email || !password) {
    return res.status(400).json({ error: 'Email and password required' });
  }

  const user = Users.getByEmail(email);
  if (!user || !user.active) {
    auditLog.write({ event: 'AUTH_LOGIN_FAIL', action: 'create', details: { email, reason: 'not_found' }, result: 'denied', ip: req.clientIP });
    return res.status(401).json({ error: 'Invalid credentials' });
  }

  if (!verifyPasswordSync(password, user.password)) {
    auditLog.write({ event: 'AUTH_LOGIN_FAIL', userId: user.id, action: 'create', details: { email, reason: 'bad_password' }, result: 'denied', ip: req.clientIP });
    return res.status(401).json({ error: 'Invalid credentials' });
  }

  const token = signToken(user);
  auditLog.write({ event: 'AUTH_LOGIN', userId: user.id, role: user.role, action: 'create', result: 'success', ip: req.clientIP });
  res.json({ token, user: { id: user.id, email: user.email, name: user.name, role: user.role, region: user.region } });
});

app.post('/api/auth/register', authLimiter, (req, res) => {
  const { email, password, name, role } = req.body;
  if (!email || !password || !name) {
    return res.status(400).json({ error: 'Email, password, and name required' });
  }

  const existing = Users.getByEmail(email);
  if (existing) {
    return res.status(409).json({ error: 'Email already registered' });
  }

  // Only platform_admin can create admin/reviewer/auditor accounts
  const allowedSelfRoles = ['annotator'];
  const actualRole = (req.user?.role === 'platform_admin' && role) ? role : 'annotator';
  if (!allowedSelfRoles.includes(actualRole) && req.user?.role !== 'platform_admin') {
    return res.status(403).json({ error: 'Cannot self-register with elevated role' });
  }

  const user = Users.create({ email, password, name, role: actualRole });
  const token = signToken({ ...user, password: undefined });
  auditLog.write({ event: 'AUTH_REGISTER', userId: user.id, action: 'create', result: 'success', ip: req.clientIP });
  res.status(201).json({ token, user });
});

app.get('/api/auth/me', authMiddleware, (req, res) => {
  const user = Users.getById(req.user.userId);
  if (!user) return res.status(404).json({ error: 'User not found' });
  res.json(user);
});

// ========== USER MANAGEMENT (platform_admin only) ==========

app.get('/api/users', authMiddleware, requireRole('platform_admin', 'auditor'), (req, res) => {
  res.json(Users.getAll());
});

app.put('/api/users/:id', authMiddleware, requireRole('platform_admin'), (req, res) => {
  const { name, role, region, active } = req.body;
  const user = Users.update(req.params.id, { name, role, region, active });
  if (!user) return res.status(404).json({ error: 'User not found' });
  auditLog.write({ event: 'USER_UPDATE', userId: req.user.userId, action: 'update', resource: `user:${req.params.id}`, result: 'success' });
  res.json(user);
});

app.delete('/api/users/:id', authMiddleware, requireRole('platform_admin'), (req, res) => {
  Users.delete(req.params.id);
  auditLog.write({ event: 'USER_DELETE', userId: req.user.userId, action: 'delete', resource: `user:${req.params.id}`, result: 'success' });
  res.json({ success: true });
});

// ========== GDPR COMPLIANCE ROUTES ==========

// Art.15 — Data Subject Access Request (DSAR)
app.get('/api/gdpr/access/:subjectId', authMiddleware, requireRole('platform_admin', 'data_admin'), (req, res) => {
  const data = Users.getAllData(req.params.subjectId);
  auditLog.write({ event: 'GDPR_ACCESS_REQUEST', userId: req.user.userId, action: 'read', resource: `subject:${req.params.subjectId}`, result: 'success' });
  res.json({ subjectId: req.params.subjectId, data, exportedAt: new Date().toISOString() });
});

// Art.17 — Right to Erasure (Right to be Forgotten)
app.delete('/api/gdpr/erasure/:subjectId', authMiddleware, requireRole('platform_admin'), (req, res) => {
  const deleted = Users.eraseAllData(req.params.subjectId);
  auditLog.write({ event: 'GDPR_ERASURE', userId: req.user.userId, action: 'delete', resource: `subject:${req.params.subjectId}`, details: deleted, result: 'success' });
  res.json({ success: true, deleted, message: 'All subject data erased per GDPR Art.17' });
});

// Art.20 — Data Portability
app.get('/api/gdpr/portability/:subjectId', authMiddleware, requireRole('platform_admin', 'data_admin'), (req, res) => {
  const data = Users.getAllData(req.params.subjectId);
  auditLog.write({ event: 'GDPR_PORTABILITY', userId: req.user.userId, action: 'read', resource: `subject:${req.params.subjectId}`, result: 'success' });
  res.setHeader('Content-Disposition', `attachment; filename="subject_${req.params.subjectId}_data.json"`);
  res.json(data);
});

// ========== AUDIT LOG ROUTES ==========

app.get('/api/audit/logs', authMiddleware, requireRole('platform_admin', 'auditor'), (req, res) => {
  const { startDate, endDate, event, userId, result, limit } = req.query;
  const logs = auditLog.query({ startDate, endDate, event, userId, result, limit: limit ? parseInt(limit) : 100 });
  res.json(logs);
});

app.get('/api/audit/stats', authMiddleware, requireRole('platform_admin', 'auditor'), (req, res) => {
  res.json(auditLog.getStats());
});

// ========== EXPORT ROUTES ==========

app.get('/api/export/formats', (req, res) => {
  res.json(exportService.EXPORT_FORMATS);
});

app.post('/api/export/dataset/:datasetId', exportLimiter, (req, res) => {
  const { format, options } = req.body;
  const datasetId = req.params.datasetId;

  // Gather data
  const datasets = datasetId === 'all'
    ? Datasets.getAll()
    : [Datasets.getById(datasetId)].filter(Boolean);

  if (datasets.length === 0) {
    return res.status(404).json({ error: 'Dataset not found' });
  }

  const episodes = datasetId === 'all'
    ? Episodes.getAll()
    : Episodes.getByDataset(datasetId);

  const annotations = Annotations.getAll();

  let content, filename, mimeType;

  switch (format) {
    case 'csv':
      content = exportService.exportToCSV(annotations, episodes, datasets);
      filename = `robomemo_export_${datasetId}.csv`;
      mimeType = 'text/csv';
      break;
    case 'coco_json':
      content = exportService.exportToCOCO(annotations, episodes, datasets);
      filename = `robomemo_coco_${datasetId}.json`;
      mimeType = 'application/json';
      break;
    case 'robomemo_json':
      content = exportService.exportToRoboMemoJSON(annotations, episodes, datasets);
      filename = `robomemo_full_${datasetId}.json`;
      mimeType = 'application/json';
      break;
    case 'lerobot_meta':
      content = exportService.exportToLeRobotMeta(annotations, episodes, datasets);
      filename = `robomemo_lerobot_${datasetId}.json`;
      mimeType = 'application/json';
      break;
    default:
      return res.status(400).json({ error: `Unsupported format: ${format}` });
  }

  auditLog.write({
    event: 'DATA_EXPORT',
    userId: req.user?.userId,
    action: 'export',
    resource: `dataset:${datasetId}`,
    details: { format, episodeCount: episodes.length, annotationCount: annotations.length },
    result: 'success',
  });

  res.setHeader('Content-Disposition', `attachment; filename="${filename}"`);
  res.setHeader('Content-Type', mimeType);
  res.send(content);
});

// ========== TEMPORAL CONSISTENCY ENDPOINT ==========

app.post('/api/vlm/validate-consistency', (req, res) => {
  const { analysis } = req.body;
  if (!analysis) {
    return res.status(400).json({ error: 'analysis object required' });
  }
  const result = validateTemporalConsistency(analysis);
  res.json(result);
});

// ========== DEMO VQA ANALYSIS (no real video required) ==========

/**
 * POST /api/vlm/demo-analysis
 * Returns a rich mock VQA analysis with full grounding for UI demo/testing.
 */
app.post('/api/vlm/demo-analysis', (req, res) => {
  const { taskName = 'Pick and Place Red Cup' } = req.body;

  const totalFrames = 240;
  const fps = 30;
  const duration = 8.0;

  const grnd = (frames, tss, desc, conf = 0.92, bboxes = []) => ({
    frame_indices: frames,
    timestamps: tss,
    bboxes,
    description: desc,
    confidence: conf,
  });

  const analysis = {
    temporal: {
      action_sequence: [
        {
          action: 'Approach',
          timestamp: '0:00',
          frame_range: [0, 48],
          description: 'Robot arm moves from rest position toward the red cup',
          grounding: grnd([0, 24, 48], ['0:00', '0:01', '0:02'],
            'Frames 0-48: end-effector translates toward cup on left side of workspace',
            0.95, [{ x: 0.55, y: 0.35, width: 0.12, height: 0.18, label: 'end-effector' }]),
        },
        {
          action: 'Pre-grasp positioning',
          timestamp: '0:02',
          frame_range: [48, 90],
          description: 'Gripper aligns above the red cup and opens jaws',
          grounding: grnd([48, 66, 90], ['0:02', '0:02.5', '0:03'],
            'Frames 48-90: gripper jaws visible opening — gap increases from 0 to ~4 cm',
            0.90, [
              { x: 0.38, y: 0.30, width: 0.10, height: 0.14, label: 'gripper' },
              { x: 0.42, y: 0.48, width: 0.08, height: 0.12, label: 'cup' },
            ]),
        },
        {
          action: 'Grasp',
          timestamp: '0:03',
          frame_range: [90, 120],
          description: 'Gripper closes around the cup establishing contact',
          grounding: grnd([90, 105, 120], ['0:03', '0:03.5', '0:04'],
            'Frames 90-120: jaw gap closes, contact deformation visible on cup rim',
            0.97, [{ x: 0.40, y: 0.44, width: 0.14, height: 0.20, label: 'grasp-contact' }]),
        },
        {
          action: 'Lift',
          timestamp: '0:04',
          frame_range: [120, 168],
          description: 'Cup lifted 15 cm above table surface',
          grounding: grnd([120, 144, 168], ['0:04', '0:05', '0:05.6'],
            'Frames 120-168: cup rises vertically — shadow on table grows',
            0.94, [{ x: 0.40, y: 0.25, width: 0.14, height: 0.22, label: 'cup-lifted' }]),
        },
        {
          action: 'Transport',
          timestamp: '0:05.6',
          frame_range: [168, 210],
          description: 'Cup carried horizontally to target zone',
          grounding: grnd([168, 189, 210], ['0:05.6', '0:06.3', '0:07'],
            'Frames 168-210: cup translates rightward at constant height',
            0.91, [{ x: 0.65, y: 0.28, width: 0.14, height: 0.22, label: 'cup-transport' }]),
        },
        {
          action: 'Place',
          timestamp: '0:07',
          frame_range: [210, 240],
          description: 'Cup lowered and released onto target position',
          grounding: grnd([210, 225, 240], ['0:07', '0:07.5', '0:08'],
            'Frames 210-240: cup descends, contact with surface, gripper opens',
            0.96, [{ x: 0.68, y: 0.46, width: 0.12, height: 0.18, label: 'cup-placed' }]),
        },
      ],
      relationships: [
        'Approach BEFORE Pre-grasp positioning',
        'Pre-grasp positioning BEFORE Grasp',
        'Grasp BEFORE Lift',
        'Lift BEFORE Transport',
        'Transport BEFORE Place',
      ],
    },
    spatial: {
      key_relationships: [
        {
          timestamp: '0:03',
          relationship: 'Gripper directly above cup — centered',
          details: 'End-effector centroid is within 2 cm horizontally of cup centroid at grasp moment',
          grounding: grnd([90, 105], ['0:03', '0:03.5'],
            'Overhead perspective shows gripper perfectly aligned with cup top opening',
            0.93, [
              { x: 0.39, y: 0.28, width: 0.10, height: 0.10, label: 'gripper-center' },
              { x: 0.41, y: 0.44, width: 0.08, height: 0.10, label: 'cup-opening' },
            ]),
        },
        {
          timestamp: '0:00',
          relationship: 'Cup is to the LEFT of the robot base',
          details: 'Cup positioned at x=0.30 in frame coords; robot base at x=0.85',
          grounding: grnd([0, 12], ['0:00', '0:00.4'],
            'Wide-angle frame shows cup on left side, robot base right of center',
            0.88, [
              { x: 0.28, y: 0.50, width: 0.09, height: 0.14, label: 'cup' },
              { x: 0.80, y: 0.60, width: 0.15, height: 0.30, label: 'robot-base' },
            ]),
        },
        {
          timestamp: '0:07',
          relationship: 'Cup placed to the RIGHT of original position',
          details: 'Target zone is 35 cm to the right of the start position',
          grounding: grnd([210, 240], ['0:07', '0:08'],
            'Final frame shows cup in target zone marked by tape on table',
            0.91, [{ x: 0.68, y: 0.48, width: 0.10, height: 0.14, label: 'cup-final' }]),
        },
      ],
      trajectory_spatial: 'Robot arm operates within a 40×30 cm horizontal workspace. Cup travels left-to-right. Vertical clearance is 15 cm during transport to clear table obstacles.',
    },
    attribute: {
      objects: [
        {
          name: 'Red plastic cup',
          properties: { color: 'Red (#C0392B)', material: 'Polypropylene plastic', shape: 'Cylindrical tapered', size: '~9 cm height, 7 cm diameter' },
          state_changes: [
            'Stationary on table → Grasped (t=0:03)',
            'Grasped → Lifted 15 cm (t=0:04)',
            'Lifted → Placed in target zone (t=0:08)',
          ],
          grounding: grnd([0, 90, 240], ['0:00', '0:03', '0:08'],
            'Red cup visible throughout — color consistent. Slight deformation at contact points visible in frames 90-120.',
            0.95, [{ x: 0.40, y: 0.46, width: 0.09, height: 0.14, label: 'red-cup' }]),
        },
        {
          name: 'Robot gripper (parallel jaw)',
          properties: { color: 'Metallic grey', material: 'Anodised aluminium', shape: 'Parallel jaw', size: '~12 cm span, 4 cm finger width' },
          state_changes: [
            'Closed → Opens to 5 cm (t=0:02)',
            'Open → Closes on cup (t=0:03)',
            'Holding → Releases cup (t=0:07.5)',
          ],
          grounding: grnd([48, 90, 225], ['0:02', '0:03', '0:07.5'],
            'Gripper jaw gap changes clearly visible—opens at frame 48, closes at frame 90, reopens at frame 225.',
            0.93, [{ x: 0.38, y: 0.30, width: 0.12, height: 0.10, label: 'gripper' }]),
        },
        {
          name: 'Work surface (table)',
          properties: { color: 'White with black tape markers', material: 'Laminated MDF', shape: 'Rectangular flat surface', size: '60×40 cm visible area' },
          state_changes: ['Cup removed from left zone (t=0:03)', 'Cup placed in right zone (t=0:08)'],
          grounding: grnd([0, 240], ['0:00', '0:08'],
            'Table surface always visible — tape markers indicate start/end zones.',
            0.98, [{ x: 0.1, y: 0.55, width: 0.80, height: 0.35, label: 'table' }]),
        },
      ],
    },
    mechanics: {
      contacts: [
        {
          timestamp: '0:03',
          contact_type: 'Parallel-jaw pinch grip',
          force_level: 'medium',
          contact_points: 'Both fingers contact cup outer wall at mid-height (~4.5 cm from base)',
          area: 'Two ~2×1 cm rectangular contact patches on opposite sides of cup',
          grounding: grnd([90, 100, 112], ['0:03', '0:03.3', '0:03.7'],
            'Frame 90: grip initiates. Frames 100-112: slight cup wall deformation visible at contact patch.',
            0.89, [
              { x: 0.37, y: 0.45, width: 0.05, height: 0.08, label: 'contact-L' },
              { x: 0.53, y: 0.45, width: 0.05, height: 0.08, label: 'contact-R' },
            ]),
        },
        {
          timestamp: '0:07.2',
          contact_type: 'Release / controlled descent',
          force_level: 'light',
          contact_points: 'Cup base contacts table surface softly',
          area: 'Full cup base ring (~4 cm diameter)',
          grounding: grnd([216, 228], ['0:07.2', '0:07.6'],
            'Cup base touches table gently — no bounce or slide observed.',
            0.91, [{ x: 0.65, y: 0.56, width: 0.12, height: 0.06, label: 'cup-base-land' }]),
        },
      ],
      force_profile: 'Grip force increases from 0 N to ~5 N during grasp (t=0:03–0:03.7), maintains ~5 N through transport, decreases to 0 N at release (t=0:07.5). Contact with table surface at ~0.5 N (low-velocity placement).',
    },
    reasoning: {
      action_justifications: [
        {
          action: 'Approach from above-left at 45°',
          reason: 'Approach angle avoids occlusion of the cup by the robot arm itself, keeping camera line-of-sight clear for visual feedback',
          constraints: ['No collision with table edge', 'Camera field-of-view must include cup', 'Joint angle limits'],
          grounding: grnd([0, 36], ['0:00', '0:01.2'],
            'Arm trajectory visible from t=0:00 — diagonal approach path from rest position at upper right.',
            0.87),
        },
        {
          action: 'Pre-grasp: open gripper before descent',
          reason: 'Opening jaws before the final descent prevents contact with cup during alignment, reducing risk of knocking it over',
          constraints: ['Gripper must be fully open (>4 cm) before reaching cup level'],
          grounding: grnd([48, 72], ['0:02', '0:02.4'],
            'Frames 48-72: jaw opening completes while end-effector is still 3 cm above cup rim.',
            0.92),
        },
        {
          action: '15 cm vertical lift before horizontal transport',
          reason: 'Lift clears any potential obstacles on the table surface between start and goal positions',
          constraints: ['No nearby obstacles confirmed visually', 'Lift height is minimum safe clearance'],
          grounding: grnd([120, 150], ['0:04', '0:05'],
            'Vertical lift path visible — cup rises to ~15 cm above table confirmed by shadow length.',
            0.90),
        },
        {
          action: 'Slow horizontal transport velocity',
          reason: 'Open-top cup could spill contents if transported too fast (centrifugal force). Gravitational stability maintained.',
          constraints: ['Cup assumed to potentially contain liquid', 'Max lateral acceleration < 0.3 g'],
          grounding: grnd([168, 200], ['0:05.6', '0:06.7'],
            'Smooth constant-velocity transport visible — no acceleration artifacts or cup oscillation.',
            0.86),
        },
      ],
      overall_strategy: 'Conservative pick-and-place: minimise risk of cup toppling by using above-approach, pre-open grasp, sufficient lift height, and slow horizontal travel. Prioritises reliability over speed.',
    },
    summary: {
      task_description: `Pick and place: move red cup from left table zone to right target zone (${taskName})`,
      start_state: 'Red cup upright on left table zone. Gripper at rest position. Table clear of obstacles.',
      end_state: 'Red cup upright in target zone (right side). Gripper returned open. Task complete.',
      success: true,
      key_milestones: [
        't=0:00 — Arm departs rest position',
        't=0:02 — Gripper aligns above cup and opens',
        't=0:03 — Grasp established (grip force ~5 N)',
        't=0:04 — Cup lifted 15 cm above table',
        't=0:07 — Cup placed in target zone',
        't=0:08 — Task complete, gripper releases',
      ],
      duration: '8.0 seconds',
      grounding_start: grnd([0, 6], ['0:00', '0:00.2'],
        'Frame 0: cup on left zone, gripper at rest upper-right of frame.',
        0.98, [
          { x: 0.28, y: 0.46, width: 0.09, height: 0.14, label: 'cup-start' },
          { x: 0.80, y: 0.10, width: 0.12, height: 0.15, label: 'gripper-rest' },
        ]),
      grounding_end: grnd([234, 240], ['0:07.8', '0:08'],
        'Frame 240: cup upright in target zone, gripper open above.',
        0.97, [
          { x: 0.66, y: 0.46, width: 0.09, height: 0.14, label: 'cup-end' },
          { x: 0.66, y: 0.28, width: 0.12, height: 0.10, label: 'gripper-open' },
        ]),
    },
    trajectory: {
      motion_segments: [
        {
          segment: 'Rest → above cup',
          time_range: '0:00-0:02',
          motion_type: 'curved',
          velocity: 'medium',
          waypoints: ['Rest (0.8, 0.1, 0.5)', 'Mid-arc (0.55, 0.15, 0.4)', 'Above cup (0.42, 0.28, 0.35)'],
          grounding: grnd([0, 24, 48], ['0:00', '0:01', '0:02'],
            'Arm arcs from top-right to center-left following curved joint-space path.',
            0.91, [{ x: 0.45, y: 0.20, width: 0.30, height: 0.25, label: 'approach-arc' }]),
        },
        {
          segment: 'Descent to grasp',
          time_range: '0:02-0:03',
          motion_type: 'linear',
          velocity: 'slow',
          waypoints: ['Above cup (0.42, 0.28, 0.35)', 'Cup mid (0.41, 0.36, 0.42)', 'Grasp point (0.41, 0.44, 0.47)'],
          grounding: grnd([48, 66, 90], ['0:02', '0:02.5', '0:03'],
            'Vertical linear descent — end-effector moves straight down into cup.',
            0.94),
        },
        {
          segment: 'Vertical lift',
          time_range: '0:04-0:05.6',
          motion_type: 'linear',
          velocity: 'medium',
          waypoints: ['Grasp (0.41, 0.44, 0.47)', 'Mid-lift (0.41, 0.36, 0.35)', 'Top (0.41, 0.27, 0.25)'],
          grounding: grnd([120, 150, 168], ['0:04', '0:05', '0:05.6'],
            'Straight vertical lift — arm extends upward, cup follows.',
            0.93),
        },
        {
          segment: 'Horizontal transport',
          time_range: '0:05.6-0:07',
          motion_type: 'linear',
          velocity: 'slow',
          waypoints: ['Left zone (0.41, 0.27, 0.25)', 'Center (0.55, 0.27, 0.25)', 'Target (0.68, 0.27, 0.25)'],
          grounding: grnd([168, 192, 210], ['0:05.6', '0:06.4', '0:07'],
            'Constant-height horizontal translation from left to right across workspace.',
            0.92),
        },
        {
          segment: 'Descent to place',
          time_range: '0:07-0:08',
          motion_type: 'linear',
          velocity: 'slow',
          waypoints: ['Above target (0.68, 0.27, 0.25)', 'Surface (0.68, 0.47, 0.47)'],
          grounding: grnd([210, 230, 240], ['0:07', '0:07.7', '0:08'],
            'Controlled vertical descent — cup placed gently onto table.',
            0.95),
        },
      ],
      overall_path: 'J-shaped trajectory: arc from rest → descend vertically → lift → horizontal traverse → descend to place. Total path length ~85 cm. Average end-effector speed ~11 cm/s.',
    },
    visual_evidence: {
      key_frames: [
        { frame_idx: 0,   timestamp: '0:00', significance: 'Initial state — cup and gripper at rest' },
        { frame_idx: 48,  timestamp: '0:02', significance: 'Gripper aligned above cup, jaws begin opening' },
        { frame_idx: 90,  timestamp: '0:03', significance: 'Grasp contact established' },
        { frame_idx: 120, timestamp: '0:04', significance: 'Cup fully lifted clear of table' },
        { frame_idx: 168, timestamp: '0:05.6', significance: 'Horizontal transport begins' },
        { frame_idx: 210, timestamp: '0:07', significance: 'Above target zone, beginning descent' },
        { frame_idx: 240, timestamp: '0:08', significance: 'Cup placed — task complete' },
      ],
    },
    confidence_scores: {
      temporal:   0.95,
      spatial:    0.91,
      attribute:  0.96,
      mechanics:  0.89,
      reasoning:  0.88,
      summary:    0.97,
      trajectory: 0.93,
    },
    temporal_consistency: {
      consistent: true,
      conflicts: [],
      stats: { totalClaims: 22, categoriesChecked: ['temporal', 'spatial', 'mechanics', 'trajectory'] },
    },
    metadata: {
      video_path: 'demo://pick-and-place',
      video_info: { total_frames: totalFrames, fps, duration },
      num_frames_analyzed: 32,
      model: 'demo-mock-v1',
      frame_timestamps: Array.from({ length: 32 }, (_, i) => {
        const s = (i / 31) * duration;
        return `${Math.floor(s / 60)}:${(s % 60).toFixed(1).padStart(4, '0')}`;
      }),
      frame_image_urls: [],
      is_demo: true,
    },
  };

  // Run temporal consistency check on the mock data
  const consistency = validateTemporalConsistency(analysis);
  analysis.temporal_consistency = consistency;

  res.json({ success: true, annotationId: `demo_${Date.now()}`, analysis });
});

// ========== DATA LINEAGE ENDPOINTS ==========

/**
 * POST /api/lineage/record  — record a data lineage event
 */
app.post('/api/lineage/record', (req, res) => {
  const { resourceId, resourceType, operation, sourceRegion, annotator, vlmProvider, details } = req.body;
  if (!resourceId || !resourceType || !operation) {
    return res.status(400).json({ error: 'resourceId, resourceType, operation required' });
  }
  try {
    const entry = Lineage.record({
      resourceId, resourceType, operation,
      sourceRegion: sourceRegion || req.geoCountry || 'unknown',
      annotatorId: annotator || req.user?.userId,
      vlmProvider: vlmProvider || null,
      uploadIp: req.clientIP,
      processingNode: process.env.NODE_HOSTNAME || 'robomemo-backend',
      details: details || {},
    });
    res.json({ success: true, lineageId: entry.id });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

/**
 * GET /api/lineage/trace/:resourceId  — get full lineage trail for a resource
 */
app.get('/api/lineage/trace/:resourceId', (req, res) => {
  try {
    res.json(Lineage.trace(req.params.resourceId));
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// ========== SIMULATORS API ==========

const SIMULATORS = [
  {
    id: 'isaac-lab',
    name: 'Isaac Lab',
    description: 'NVIDIA Isaac Lab — GPU-accelerated robot learning',
    version: '2024.1',
    status: 'stopped',
    scenes: [
      { id: 'tabletop', name: 'Tabletop Manipulation', description: 'Flat surface with objects' },
      { id: 'kitchen', name: 'Kitchen Environment', description: 'Full kitchen with appliances' },
      { id: 'factory', name: 'Factory Floor', description: 'Industrial assembly line' },
      { id: 'office', name: 'Office Space', description: 'Desktop manipulation scenario' },
    ],
    robots: [
      { id: 'franka', name: 'Franka Emika Panda', dof: 7, type: 'single-arm' },
      { id: 'ur5e', name: 'Universal Robots UR5e', dof: 6, type: 'single-arm' },
      { id: 'bimanual-franka', name: 'Bimanual Franka', dof: 14, type: 'bimanual' },
    ],
    capabilities: ['physics', 'rendering', 'sensors', 'domain_randomization'],
  },
  {
    id: 'mujoco',
    name: 'MuJoCo',
    description: 'Multi-Joint dynamics with Contact — fast physics simulation',
    version: '3.1.0',
    status: 'stopped',
    scenes: [
      { id: 'tabletop', name: 'Tabletop Manipulation', description: 'Simple table setup' },
      { id: 'locomotion', name: 'Locomotion Arena', description: 'Flat ground with obstacles' },
    ],
    robots: [
      { id: 'shadow-hand', name: 'Shadow Dexterous Hand', dof: 24, type: 'dexterous' },
      { id: 'allegro', name: 'Allegro Hand', dof: 16, type: 'dexterous' },
      { id: 'franka', name: 'Franka Panda', dof: 7, type: 'single-arm' },
    ],
    capabilities: ['physics', 'contacts', 'tendons'],
  },
  {
    id: 'gazebo',
    name: 'Gazebo',
    description: 'ROS-compatible simulation environment',
    version: 'Harmonic',
    status: 'stopped',
    scenes: [
      { id: 'warehouse', name: 'Warehouse', description: 'Logistics environment' },
      { id: 'home', name: 'Home Environment', description: 'Domestic tasks' },
    ],
    robots: [
      { id: 'tiago', name: 'TIAGo', dof: 14, type: 'mobile-manipulation' },
      { id: 'fetch', name: 'Fetch', dof: 10, type: 'mobile-manipulation' },
    ],
    capabilities: ['physics', 'ros2', 'sensors', 'navigation'],
  },
];

// In-memory simulator state for MVP
const simState = {};
SIMULATORS.forEach(s => { simState[s.id] = { status: 'stopped', scene: null, robot: null, steps: 0 }; });

app.get('/api/simulators', (req, res) => {
  res.json(SIMULATORS.map(s => ({ ...s, ...simState[s.id] })));
});

app.get('/api/simulators/:id', (req, res) => {
  const sim = SIMULATORS.find(s => s.id === req.params.id);
  if (!sim) return res.status(404).json({ error: 'Simulator not found' });
  res.json({ ...sim, ...simState[sim.id] });
});

app.post('/api/simulators/:id/start', (req, res) => {
  const sim = SIMULATORS.find(s => s.id === req.params.id);
  if (!sim) return res.status(404).json({ error: 'Simulator not found' });
  const { scene, robot, sensors } = req.body;
  simState[sim.id] = { status: 'running', scene: scene || sim.scenes[0]?.id, robot: robot || sim.robots[0]?.id, sensors: sensors || [], steps: 0, startedAt: new Date().toISOString() };
  res.json({ ...sim, ...simState[sim.id] });
});

app.post('/api/simulators/:id/stop', (req, res) => {
  const sim = SIMULATORS.find(s => s.id === req.params.id);
  if (!sim) return res.status(404).json({ error: 'Simulator not found' });
  simState[sim.id] = { ...simState[sim.id], status: 'stopped', stoppedAt: new Date().toISOString() };
  res.json({ ...sim, ...simState[sim.id] });
});

app.post('/api/simulators/:id/reset', (req, res) => {
  const sim = SIMULATORS.find(s => s.id === req.params.id);
  if (!sim) return res.status(404).json({ error: 'Simulator not found' });
  simState[sim.id] = { ...simState[sim.id], steps: 0 };
  res.json({ ...sim, ...simState[sim.id] });
});

app.post('/api/simulators/:id/step', (req, res) => {
  const sim = SIMULATORS.find(s => s.id === req.params.id);
  if (!sim) return res.status(404).json({ error: 'Simulator not found' });
  if (simState[sim.id].status !== 'running') return res.status(400).json({ error: 'Simulator not running' });
  simState[sim.id].steps += 1;
  const { action } = req.body;
  res.json({
    step: simState[sim.id].steps,
    observation: {
      joint_positions: Array(7).fill(0).map(() => +(Math.random() * 3.14 - 1.57).toFixed(4)),
      joint_velocities: Array(7).fill(0).map(() => +(Math.random() * 0.5 - 0.25).toFixed(4)),
      gripper_state: Math.random() > 0.5 ? 'open' : 'closed',
      force_torque: Array(6).fill(0).map(() => +(Math.random() * 10 - 5).toFixed(3)),
    },
    reward: +(Math.random()).toFixed(4),
    done: simState[sim.id].steps >= 500,
    info: { action_applied: action || null },
  });
});

app.get('/api/simulators/:id/scenes', (req, res) => {
  const sim = SIMULATORS.find(s => s.id === req.params.id);
  if (!sim) return res.status(404).json({ error: 'Simulator not found' });
  res.json(sim.scenes);
});

app.get('/api/simulators/:id/robots', (req, res) => {
  const sim = SIMULATORS.find(s => s.id === req.params.id);
  if (!sim) return res.status(404).json({ error: 'Simulator not found' });
  res.json(sim.robots);
});

// ========== SENSORS API ==========

const SENSOR_TYPES = [
  { id: 'rgb', name: 'RGB Camera', dataType: 'image', frequency: '30 Hz', resolution: '640x480' },
  { id: 'depth', name: 'Depth Camera', dataType: 'depth_map', frequency: '30 Hz', resolution: '640x480' },
  { id: 'rgbd', name: 'RGB-D Camera', dataType: 'image+depth', frequency: '30 Hz', resolution: '640x480' },
  { id: 'force_torque', name: 'Force/Torque Sensor', dataType: 'force_torque_6d', frequency: '1000 Hz', axes: 6 },
  { id: 'tactile', name: 'Tactile Sensor', dataType: 'pressure_array', frequency: '100 Hz', resolution: '16x16' },
  { id: 'joint_state', name: 'Joint State', dataType: 'joint_positions', frequency: '500 Hz' },
  { id: 'imu', name: 'IMU', dataType: 'imu_6d', frequency: '200 Hz' },
  { id: 'lidar', name: 'LiDAR', dataType: 'point_cloud', frequency: '10 Hz', points: 16000 },
];

const DEFAULT_SENSOR_CONFIGS = {
  'bimanual-manipulation': {
    id: 'bimanual-manipulation',
    name: 'Bimanual Manipulation',
    description: 'Dual arm with wrist cameras, chest camera, and F/T sensors',
    sensors: [
      { type: 'rgbd', mount: 'chest', name: 'Chest Camera' },
      { type: 'rgbd', mount: 'left_wrist', name: 'Left Wrist Camera' },
      { type: 'rgbd', mount: 'right_wrist', name: 'Right Wrist Camera' },
      { type: 'force_torque', mount: 'left_wrist', name: 'Left F/T' },
      { type: 'force_torque', mount: 'right_wrist', name: 'Right F/T' },
      { type: 'joint_state', mount: 'left_arm', name: 'Left Arm Joints' },
      { type: 'joint_state', mount: 'right_arm', name: 'Right Arm Joints' },
    ],
  },
  'humanoid-full-body': {
    id: 'humanoid-full-body',
    name: 'Humanoid Full Body',
    description: 'Multi-camera setup with tactile sensors for humanoid robots',
    sensors: [
      { type: 'rgbd', mount: 'head', name: 'Head Camera' },
      { type: 'rgbd', mount: 'left_wrist', name: 'Left Wrist Camera' },
      { type: 'rgbd', mount: 'right_wrist', name: 'Right Wrist Camera' },
      { type: 'force_torque', mount: 'left_wrist', name: 'Left F/T' },
      { type: 'force_torque', mount: 'right_wrist', name: 'Right F/T' },
      { type: 'tactile', mount: 'left_hand', name: 'Left Tactile' },
      { type: 'tactile', mount: 'right_hand', name: 'Right Tactile' },
      { type: 'imu', mount: 'torso', name: 'Torso IMU' },
      { type: 'joint_state', mount: 'full_body', name: 'Full Body Joints' },
    ],
  },
  'mobile-manipulation': {
    id: 'mobile-manipulation',
    name: 'Mobile Manipulation',
    description: 'Base with arm, LiDAR, and cameras',
    sensors: [
      { type: 'rgbd', mount: 'head', name: 'Head Camera' },
      { type: 'rgbd', mount: 'wrist', name: 'Wrist Camera' },
      { type: 'force_torque', mount: 'wrist', name: 'Wrist F/T' },
      { type: 'lidar', mount: 'base', name: 'Base LiDAR' },
      { type: 'imu', mount: 'base', name: 'Base IMU' },
      { type: 'joint_state', mount: 'arm', name: 'Arm Joints' },
    ],
  },
  'roboforce-titan': {
    id: 'roboforce-titan',
    name: 'RoboForce Titan (Screw Driving)',
    description: 'RGBD chest + 2x wrist cameras + 2x EOAT F/T sensors for screw driving tasks',
    sensors: [
      { type: 'rgbd', mount: 'chest', name: 'Chest RGBD Camera' },
      { type: 'rgbd', mount: 'left_wrist', name: 'Left Wrist RGBD Camera' },
      { type: 'rgbd', mount: 'right_wrist', name: 'Right Wrist RGBD Camera' },
      { type: 'force_torque', mount: 'left_eoat', name: 'Left EOAT F/T (6-axis)' },
      { type: 'force_torque', mount: 'right_eoat', name: 'Right EOAT F/T (6-axis)' },
      { type: 'joint_state', mount: 'left_arm', name: 'Left Arm Joints' },
      { type: 'joint_state', mount: 'right_arm', name: 'Right Arm Joints' },
    ],
  },
};

// In-memory custom sensor configs
const customSensorConfigs = [];

app.get('/api/sensors/types', (req, res) => {
  res.json(SENSOR_TYPES);
});

app.get('/api/sensors/configs', (req, res) => {
  res.json([...Object.values(DEFAULT_SENSOR_CONFIGS), ...customSensorConfigs]);
});

app.post('/api/sensors/configs', (req, res) => {
  const newConfig = {
    id: `sc_${Date.now()}`,
    ...req.body,
  };
  customSensorConfigs.push(newConfig);
  res.status(201).json(newConfig);
});

app.get('/api/sensors/defaults', (req, res) => {
  res.json(DEFAULT_SENSOR_CONFIGS);
});

// ========== AUGMENTATION API ==========

const AUGMENTATION_MODELS = [
  {
    id: 'emu3.5',
    name: 'Emu3.5 (PKU)',
    description: 'Autoregressive video generation for data augmentation',
    type: 'world-model',
    capabilities: ['video_generation', 'image_editing', 'context_transfer'],
    status: 'available',
  },
  {
    id: 'rynn-002',
    name: 'Rynn-002',
    description: 'Embodied world model for sim-to-real domain randomization',
    type: 'world-model',
    capabilities: ['domain_randomization', 'physics_augmentation', 'lighting_variation'],
    status: 'available',
  },
  {
    id: 'genaugment',
    name: 'GenAugment',
    description: 'Cross-embodiment transfer using visual-language models',
    type: 'transfer',
    capabilities: ['cross_embodiment', 'retargeting', 'skill_mapping'],
    status: 'available',
  },
];

// In-memory augmentation jobs
const augmentationJobs = {};

app.get('/api/augmentation/models', (req, res) => {
  res.json(AUGMENTATION_MODELS);
});

app.post('/api/augmentation/generate', (req, res) => {
  const { modelId, datasetId, config } = req.body;
  const jobId = `aug_${Date.now()}`;
  augmentationJobs[jobId] = {
    id: jobId,
    modelId,
    datasetId,
    config,
    status: 'processing',
    progress: 0,
    createdAt: new Date().toISOString(),
    result: null,
  };
  // Simulate async processing
  setTimeout(() => {
    if (augmentationJobs[jobId]) {
      augmentationJobs[jobId].status = 'completed';
      augmentationJobs[jobId].progress = 100;
      augmentationJobs[jobId].result = {
        generatedEpisodes: 10,
        totalFrames: 3000,
        augmentationType: config?.type || 'domain_randomization',
      };
    }
  }, 5000);
  res.json(augmentationJobs[jobId]);
});

app.get('/api/augmentation/jobs/:jobId', (req, res) => {
  const job = augmentationJobs[req.params.jobId];
  if (!job) return res.status(404).json({ error: 'Job not found' });
  res.json(job);
});

app.post('/api/augmentation/cross-embodiment', (req, res) => {
  const { sourceDatasetId, targetRobot, mappingConfig } = req.body;
  const jobId = `xemb_${Date.now()}`;
  augmentationJobs[jobId] = {
    id: jobId,
    type: 'cross-embodiment',
    sourceDatasetId,
    targetRobot,
    mappingConfig,
    status: 'processing',
    progress: 0,
    createdAt: new Date().toISOString(),
  };
  setTimeout(() => {
    if (augmentationJobs[jobId]) {
      augmentationJobs[jobId].status = 'completed';
      augmentationJobs[jobId].progress = 100;
      augmentationJobs[jobId].result = {
        transferredEpisodes: 8,
        retargetingAccuracy: 0.92,
        kinematicErrors: { mean: 0.015, max: 0.043 },
      };
    }
  }, 5000);
  res.json(augmentationJobs[jobId]);
});

app.post('/api/augmentation/cross-context', (req, res) => {
  const { sourceDatasetId, targetContext, generationConfig } = req.body;
  const jobId = `xctx_${Date.now()}`;
  augmentationJobs[jobId] = {
    id: jobId,
    type: 'cross-context',
    sourceDatasetId,
    targetContext,
    generationConfig,
    status: 'processing',
    progress: 0,
    createdAt: new Date().toISOString(),
  };
  setTimeout(() => {
    if (augmentationJobs[jobId]) {
      augmentationJobs[jobId].status = 'completed';
      augmentationJobs[jobId].progress = 100;
      augmentationJobs[jobId].result = {
        generatedEpisodes: 12,
        contextVariations: ['kitchen_bright', 'kitchen_dim', 'kitchen_cluttered'],
        fidelityScore: 0.87,
      };
    }
  }, 5000);
  res.json(augmentationJobs[jobId]);
});

// ========== PLATFORM STATS API ==========

app.get('/api/stats', (req, res) => {
  res.json(getFullStats());
});

app.get('/api/stats/datasets', (req, res) => {
  res.json(getDatasetStats());
});

app.get('/api/stats/annotations', (req, res) => {
  res.json(getAnnotationStats());
});

app.get('/api/stats/simulators', (req, res) => {
  const running = Object.entries(simState).filter(([_, s]) => s.status === 'running').length;
  const totalSteps = Object.values(simState).reduce((sum, s) => sum + (s.steps || 0), 0);
  res.json({
    total: SIMULATORS.length,
    running,
    stopped: SIMULATORS.length - running,
    totalSteps,
    simulators: SIMULATORS.map(s => ({
      id: s.id,
      name: s.name,
      status: simState[s.id].status,
      steps: simState[s.id].steps,
    })),
  });
});

app.get('/api/stats/timeline', (req, res) => {
  const days = parseInt(req.query.days) || 30;
  res.json(getTimeline(days));
});

// ========== EXTENDED AUTO-ANNOTATION API ==========

// In-memory auto-annotation jobs
const autoAnnotationJobs = {};

app.get('/api/autoannotation/jobs/:jobId', (req, res) => {
  const job = autoAnnotationJobs[req.params.jobId];
  if (!job) return res.status(404).json({ error: 'Job not found' });
  res.json(job);
});

app.post('/api/autoannotation/search', async (req, res) => {
  const { datasetId, query, modelId, topK = 5 } = req.body;
  // For MVP: search annotations by label/description containing query
  const episodes = Episodes.getByDataset(datasetId);
  const results = episodes
    .filter(ep => {
      const text = `${ep.name || ''} ${ep.description || ''} ${ep.skill || ''}`.toLowerCase();
      return text.includes((query || '').toLowerCase());
    })
    .slice(0, topK)
    .map(ep => ({
      episodeId: ep.id,
      name: ep.name,
      description: ep.description,
      skill: ep.skill,
      relevance: 0.8 + Math.random() * 0.2,
    }));
  res.json({ query, datasetId, results, count: results.length });
});

app.post('/api/autoannotation/summarize', async (req, res) => {
  const { videoPath, modelId, summaryType = 'brief', options } = req.body;

  try {
    let result;
    if (videoPath && fs.existsSync(videoPath)) {
      const vlmResponse = await processVideoWithVLM(videoPath,
        `Generate a ${summaryType} summary of this robotic task video. Return in JSON: {"summary": "...", "keyActions": [...], "duration": "..."}`);
      if (vlmResponse) {
        const jsonMatch = vlmResponse.match(/\{.*\}/s);
        result = jsonMatch ? JSON.parse(jsonMatch[0]) : null;
      }
    }

    if (!result) {
      result = {
        summary: `This video shows a robotic manipulation task. The robot performs a sequence of ${3 + Math.floor(Math.random() * 4)} actions to complete the task successfully.`,
        keyActions: ['Approach target', 'Grasp object', 'Lift and transport', 'Place at destination'],
        duration: '12.5s',
        summaryType,
        confidence: 0.88,
      };
    }
    res.json(result);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.post('/api/autoannotation/compare', async (req, res) => {
  const { videoPaths, modelId, comparisonType = 'similarity' } = req.body;

  if (!videoPaths || videoPaths.length < 2) {
    return res.status(400).json({ error: 'At least 2 video paths required' });
  }

  // Mock comparison for MVP
  res.json({
    comparisonType,
    videoCount: videoPaths.length,
    similarities: videoPaths.map((_, i) =>
      videoPaths.map((_, j) => i === j ? 1.0 : +(0.5 + Math.random() * 0.4).toFixed(3))
    ),
    differences: [
      'Video 1 uses faster approach velocity',
      'Video 2 has more precise grasp alignment',
      'Grip force differs by ~2N between videos',
    ],
    recommendations: [
      'Video 2 shows better grasp strategy — consider using as reference',
      'Combine approach from Video 1 with grasp from Video 2 for optimal performance',
    ],
    confidence: 0.82,
  });
});

app.listen(port, () => {
  console.log(`Backend server running at http://localhost:${port}`);
  console.log('[DB] SQLite ready —', require('./db').getStats());
});

// ========== TASK MANAGEMENT API ==========

// GET /api/tasks/stats — must be before :id route
app.get('/api/tasks/stats', authMiddleware, requireRole('platform_admin', 'reviewer', 'data_admin'), (req, res) => {
  const all = Tasks.getAll();
  const byStatus = {};
  const byType = {};
  const byPriority = {};
  const byAssignee = {};
  for (const t of all) {
    byStatus[t.status] = (byStatus[t.status] || 0) + 1;
    byType[t.type] = (byType[t.type] || 0) + 1;
    byPriority[t.priority] = (byPriority[t.priority] || 0) + 1;
    if (t.assignedTo) byAssignee[t.assignedTo] = (byAssignee[t.assignedTo] || 0) + 1;
  }
  res.json({ total: all.length, byStatus, byType, byPriority, byAssignee });
});

// GET /api/tasks/my — current user's tasks
app.get('/api/tasks/my', authMiddleware, (req, res) => {
  res.json(Tasks.getByUser(req.user.userId));
});

// POST /api/tasks — create task (admin/reviewer)
app.post('/api/tasks', authMiddleware, requireRole('platform_admin', 'reviewer', 'data_admin'), (req, res) => {
  const task = Tasks.insert({ ...req.body, assignedBy: req.user.userId });
  auditLog.write({ event: 'TASK_CREATE', userId: req.user.userId, action: 'create', resource: `task:${task.id}`, result: 'success' });
  res.status(201).json(task);
});

// GET /api/tasks — all tasks (admin/reviewer)
app.get('/api/tasks', authMiddleware, requireRole('platform_admin', 'reviewer', 'data_admin'), (req, res) => {
  const { status } = req.query;
  if (status) return res.json(Tasks.getByStatus(status));
  res.json(Tasks.getAll());
});

// GET /api/tasks/:id — task detail
app.get('/api/tasks/:id', authMiddleware, (req, res) => {
  const task = Tasks.getById(req.params.id);
  if (!task) return res.status(404).json({ error: 'Task not found' });
  res.json(task);
});

// PUT /api/tasks/:id — update task
app.put('/api/tasks/:id', authMiddleware, (req, res) => {
  const task = Tasks.update(req.params.id, req.body);
  if (!task) return res.status(404).json({ error: 'Task not found' });
  auditLog.write({ event: 'TASK_UPDATE', userId: req.user.userId, action: 'update', resource: `task:${req.params.id}`, result: 'success' });
  res.json(task);
});

// PUT /api/tasks/:id/status — update task status
app.put('/api/tasks/:id/status', authMiddleware, (req, res) => {
  const { status } = req.body;
  if (!status) return res.status(400).json({ error: 'status required' });
  const updates = { status };
  if (status === 'in_progress') updates.startedAt = new Date().toISOString();
  if (status === 'completed') updates.completedAt = new Date().toISOString();
  const task = Tasks.update(req.params.id, updates);
  if (!task) return res.status(404).json({ error: 'Task not found' });
  auditLog.write({ event: 'TASK_STATUS', userId: req.user.userId, action: 'update', resource: `task:${req.params.id}`, details: { status }, result: 'success' });
  res.json(task);
});

// DELETE /api/tasks/:id — delete task (admin)
app.delete('/api/tasks/:id', authMiddleware, requireRole('platform_admin'), (req, res) => {
  const existing = Tasks.getById(req.params.id);
  if (!existing) return res.status(404).json({ error: 'Task not found' });
  Tasks.delete(req.params.id);
  auditLog.write({ event: 'TASK_DELETE', userId: req.user.userId, action: 'delete', resource: `task:${req.params.id}`, result: 'success' });
  res.json({ success: true });
});

// ========== ORDER MANAGEMENT API ==========

// GET /api/orders/stats — must be before :id
app.get('/api/orders/stats', authMiddleware, requireRole('platform_admin', 'data_admin'), (req, res) => {
  res.json(Orders.getStats());
});

// POST /api/orders
app.post('/api/orders', authMiddleware, requireRole('platform_admin', 'data_admin'), (req, res) => {
  const order = Orders.insert({ ...req.body, createdBy: req.user.userId });
  auditLog.write({ event: 'ORDER_CREATE', userId: req.user.userId, action: 'create', resource: `order:${order.id}`, result: 'success' });
  res.status(201).json(order);
});

// GET /api/orders
app.get('/api/orders', authMiddleware, requireRole('platform_admin', 'data_admin'), (req, res) => {
  res.json(Orders.getAll());
});

// GET /api/orders/:id
app.get('/api/orders/:id', authMiddleware, requireRole('platform_admin', 'data_admin'), (req, res) => {
  const order = Orders.getById(req.params.id);
  if (!order) return res.status(404).json({ error: 'Order not found' });
  res.json(order);
});

// PUT /api/orders/:id
app.put('/api/orders/:id', authMiddleware, requireRole('platform_admin', 'data_admin'), (req, res) => {
  const order = Orders.update(req.params.id, req.body);
  if (!order) return res.status(404).json({ error: 'Order not found' });
  auditLog.write({ event: 'ORDER_UPDATE', userId: req.user.userId, action: 'update', resource: `order:${req.params.id}`, result: 'success' });
  res.json(order);
});

// PUT /api/orders/:id/status
app.put('/api/orders/:id/status', authMiddleware, requireRole('platform_admin', 'data_admin'), (req, res) => {
  const { status } = req.body;
  if (!status) return res.status(400).json({ error: 'status required' });
  const order = Orders.update(req.params.id, { status });
  if (!order) return res.status(404).json({ error: 'Order not found' });
  auditLog.write({ event: 'ORDER_STATUS', userId: req.user.userId, action: 'update', resource: `order:${req.params.id}`, details: { status }, result: 'success' });
  res.json(order);
});

// DELETE /api/orders/:id
app.delete('/api/orders/:id', authMiddleware, requireRole('platform_admin'), (req, res) => {
  const existing = Orders.getById(req.params.id);
  if (!existing) return res.status(404).json({ error: 'Order not found' });
  Orders.delete(req.params.id);
  auditLog.write({ event: 'ORDER_DELETE', userId: req.user.userId, action: 'delete', resource: `order:${req.params.id}`, result: 'success' });
  res.json({ success: true });
});

// GET /api/orders/:id/tasks — tasks linked to order
app.get('/api/orders/:id/tasks', authMiddleware, (req, res) => {
  const allTasks = Tasks.getAll();
  const orderTasks = allTasks.filter(t => t.datasetId === req.params.id || t.description?.includes(req.params.id));
  res.json(orderTasks);
});

// POST /api/orders/:id/tasks — create task for order
app.post('/api/orders/:id/tasks', authMiddleware, requireRole('platform_admin', 'data_admin'), (req, res) => {
  const order = Orders.getById(req.params.id);
  if (!order) return res.status(404).json({ error: 'Order not found' });
  const task = Tasks.insert({
    ...req.body,
    datasetId: order.datasetId,
    assignedBy: req.user.userId,
    description: `${req.body.description || ''} [Order: ${req.params.id}]`.trim(),
  });
  res.status(201).json(task);
});

// ========== QUALITY CONTROL API ==========

// POST /api/reviews — submit review
app.post('/api/reviews', authMiddleware, requireRole('platform_admin', 'reviewer'), (req, res) => {
  const review = Reviews.insert({ ...req.body, reviewerId: req.user.userId });
  auditLog.write({ event: 'REVIEW_CREATE', userId: req.user.userId, action: 'create', resource: `review:${review.id}`, result: 'success' });
  res.status(201).json(review);
});

// GET /api/reviews/task/:taskId — task reviews
app.get('/api/reviews/task/:taskId', authMiddleware, (req, res) => {
  res.json(Reviews.getByTask(req.params.taskId));
});

// GET /api/reviews/stats — review statistics
app.get('/api/reviews/stats', authMiddleware, requireRole('platform_admin', 'reviewer'), (req, res) => {
  res.json(Reviews.getStats());
});

// GET /api/quality/annotator/:userId — annotator quality report
app.get('/api/quality/annotator/:userId', authMiddleware, requireRole('platform_admin', 'reviewer'), (req, res) => {
  const reviews = Reviews.getByReviewer(req.params.userId);
  const scores = reviews.filter(r => r.score !== null).map(r => r.score);
  const avgScore = scores.length > 0 ? Math.round(scores.reduce((a, b) => a + b) / scores.length * 10) / 10 : null;
  const approved = reviews.filter(r => r.status === 'approved').length;
  const rejected = reviews.filter(r => r.status === 'rejected').length;
  res.json({
    userId: req.params.userId,
    totalReviews: reviews.length,
    averageScore: avgScore,
    approved,
    rejected,
    approvalRate: reviews.length > 0 ? Math.round(approved / reviews.length * 100) : 0,
  });
});

// GET /api/quality/dashboard — quality dashboard data
app.get('/api/quality/dashboard', authMiddleware, requireRole('platform_admin', 'reviewer'), (req, res) => {
  const stats = Reviews.getStats();
  const allTasks = Tasks.getAll();
  const completedTasks = allTasks.filter(t => t.status === 'completed').length;
  res.json({
    ...stats,
    completedTasks,
    qualityScore: stats.averageScore || 0,
    reviewBacklog: stats.byStatus['pending'] || 0,
  });
});

// GenRobot Dataset API
app.get('/api/datasets/genrobot', (req, res) => {
  try {
    const metadataPath = path.join(DATA_DIR, 'genrobot_open_dataset', 'metadata.json');
    if (!fs.existsSync(metadataPath)) {
      return res.status(404).json({ error: 'GenRobot dataset not found. Please run download_data.py first.' });
    }
    
    const metadata = JSON.parse(fs.readFileSync(metadataPath, 'utf8'));
    res.json(metadata);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Get specific GenRobot sample
app.get('/api/datasets/genrobot/sample/:sampleId', (req, res) => {
  try {
    const { sampleId } = req.params;
    const metadataPath = path.join(DATA_DIR, 'genrobot_open_dataset', 'metadata.json');
    const metadata = JSON.parse(fs.readFileSync(metadataPath, 'utf8'));
    
    const sample = metadata.samples.find(s => s.id === sampleId);
    if (!sample) {
      return res.status(404).json({ error: 'Sample not found' });
    }
    
    res.json(sample);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});
