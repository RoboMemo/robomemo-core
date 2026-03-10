const express = require('express');
const cors = require('cors');
const multer = require('multer');
const path = require('path');
const fs = require('fs');
const morgan = require('morgan');
const dotenv = require('dotenv');
const { GoogleGenerativeAI } = require('@google/generative-ai');
const { spawn } = require('child_process');

dotenv.config();

const app = express();
const port = 3001;

// Middleware
app.use(cors());
app.use(express.json());
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
const { Datasets, Episodes, Annotations, syncFromJSON, getStats } = require('./db');

// VLM Integration (Gemini 1.5 Flash)
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY || 'YOUR_API_KEY_HERE');

async function processVideoWithVLM(videoPath, prompt) {
  // Check if API key is provided
  if (!process.env.GEMINI_API_KEY || process.env.GEMINI_API_KEY === 'YOUR_API_KEY_HERE') {
    console.warn('GEMINI_API_KEY not set, using mock data for demo purposes');
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

app.get('/api/datasets/:id/episodes', (req, res) => {
  res.json(Episodes.getByDataset(req.params.id));
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

// ========== STRUCTURED VQA ANALYSIS API ==========

/**
 * Analyze video with structured 7-category VQA
 * POST /api/vlm/structured-analysis
 * Body: { videoPath, provider, apiKey, numFrames, model }
 */
app.post('/api/vlm/structured-analysis', async (req, res) => {
  const { videoPath, provider = 'gemini', apiKey, numFrames = 32, model } = req.body;

  if (!videoPath) {
    return res.status(400).json({ error: 'videoPath is required' });
  }

  if (!apiKey && provider !== 'local') {
    return res.status(400).json({ error: 'apiKey is required for cloud VLM providers' });
  }

  if (!fs.existsSync(videoPath)) {
    return res.status(404).json({ error: `Video file not found: ${videoPath}` });
  }

  try {
    // Spawn Python process to run VLM analysis
    const pythonScript = path.join(__dirname, 'vlm_video_analyzer.py');
    const args = [provider, apiKey, videoPath, numFrames.toString()];
    if (model) {
      args.push(model);
    }

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

app.listen(port, () => {
  console.log(`Backend server running at http://localhost:${port}`);
  console.log('[DB] SQLite ready —', require('./db').getStats());
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
