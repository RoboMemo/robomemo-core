/**
 * Bilibili API Routes
 * ====================
 * B站视频搜索、预筛选、下载相关API
 * 
 * Routes:
 *   POST /api/bilibili/search      - 搜索B站视频
 *   POST /api/bilibili/hunt        - Agent自动搜索（OpenClaw风格）
 *   POST /api/bilibili/prescreen   - 预筛选视频质量
 *   POST /api/bilibili/download    - 下载视频
 *   POST /api/bilibili/pipeline    - 一键完整流水线
 */

const express = require('express');
const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');

const router = express.Router();

// 服务路径
const BILI_SERVICES_DIR = path.join(__dirname, '../services/bili');
const SFT_SERVICES_DIR = path.join(__dirname, '../services/sft');

// 临时任务存储
const jobs = {};

/**
 * 调用Python脚本
 */
function callPython(scriptPath, args = [], timeout = 60000) {
  return new Promise((resolve, reject) => {
    // 设置环境变量确保 UTF-8 编码
    const env = {
      ...process.env,
      PYTHONIOENCODING: 'utf-8',
      PYTHONUTF8: '1',
    };
    
    const python = spawn('python', [scriptPath, ...args], {
      timeout,
      cwd: path.dirname(scriptPath),
      env,
      // Windows 下需要 shell 来正确处理编码
      shell: true,
    });

    let stdout = '';
    let stderr = '';

    python.stdout.on('data', (data) => {
      stdout += data.toString('utf8');
    });

    python.stderr.on('data', (data) => {
      stderr += data.toString('utf8');
    });

    python.on('close', (code) => {
      if (code === 0) {
        try {
          resolve(JSON.parse(stdout));
        } catch (e) {
          resolve({ raw: stdout, stderr });
        }
      } else {
        reject(new Error(`Python script failed: ${stderr || stdout}`));
      }
    });

    python.on('error', reject);
  });
}

// ═══════════════════════════════════════════════════════════════════════════════
// POST /api/bilibili/search - 搜索B站视频
// ═══════════════════════════════════════════════════════════════════════════════

router.post('/search', async (req, res) => {
  const { keyword, pageSize = 20, page = 1, order = 'totalrank' } = req.body;

  if (!keyword) {
    return res.status(400).json({ error: 'keyword is required' });
  }

  try {
    const scriptPath = path.join(BILI_SERVICES_DIR, 'bili_intel.py');
    const result = await callPython(scriptPath, [
      'search', keyword,
      '--page', String(page),
      '--page-size', String(pageSize),
      '--order', order,
    ], 30000);

    res.json(result);
  } catch (error) {
    console.error('[Bilibili Search Error]', error);
    res.status(500).json({ error: error.message });
  }
});

// ═══════════════════════════════════════════════════════════════════════════════
// POST /api/bilibili/hunt - Agent自动搜索BV号（OpenClaw风格）
// ═══════════════════════════════════════════════════════════════════════════════

router.post('/hunt', async (req, res) => {
  const { intent, minScore = 40, maxResults = 20, keywords } = req.body;

  if (!intent && !keywords) {
    return res.status(400).json({ error: 'intent or keywords is required' });
  }

  try {
    const scriptPath = path.join(BILI_SERVICES_DIR, 'bili_hunter_agent.py');
    const args = [
      intent || '搜索视频',
      '--min-score', String(minScore),
      '--max-results', String(maxResults),
      '--json',
    ];
    
    const result = await callPython(scriptPath, args, 60000);

    res.json(result);
  } catch (error) {
    console.error('[Bilibili Hunt Error]', error);
    res.status(500).json({ error: error.message });
  }
});

// ═══════════════════════════════════════════════════════════════════════════════
// POST /api/bilibili/prescreen - 预筛选视频质量
// ═══════════════════════════════════════════════════════════════════════════════

router.post('/prescreen', async (req, res) => {
  const { bvids, minScore = 50 } = req.body;

  if (!bvids || !Array.isArray(bvids) || bvids.length === 0) {
    return res.status(400).json({ error: 'bvids array is required' });
  }

  try {
    const scriptPath = path.join(BILI_SERVICES_DIR, 'prescreen.py');
    const args = [...bvids, '--min-score', String(minScore), '--json'];
    
    const result = await callPython(scriptPath, args, 60000 * bvids.length);

    res.json(result);
  } catch (error) {
    console.error('[Bilibili Prescreen Error]', error);
    res.status(500).json({ error: error.message });
  }
});

// ═══════════════════════════════════════════════════════════════════════════════
// POST /api/bilibili/download - 下载视频
// ═══════════════════════════════════════════════════════════════════════════════

router.post('/download', async (req, res) => {
  const { bvids, outputDir } = req.body;

  if (!bvids || !Array.isArray(bvids) || bvids.length === 0) {
    return res.status(400).json({ error: 'bvids array is required' });
  }

  const jobId = `dl_${Date.now()}`;
  const targetDir = outputDir || path.join(__dirname, '../uploads/videos');

  // 确保目录存在
  if (!fs.existsSync(targetDir)) {
    fs.mkdirSync(targetDir, { recursive: true });
  }

  // 异步处理
  jobs[jobId] = {
    id: jobId,
    status: 'processing',
    progress: 0,
    results: [],
    createdAt: new Date().toISOString(),
  };

  (async () => {
    const scriptPath = path.join(BILI_SERVICES_DIR, 'video_downloader.py');
    
    for (let i = 0; i < bvids.length; i++) {
      const bvid = bvids[i];
      try {
        const result = await callPython(scriptPath, [bvid, '--output', targetDir], 600000);
        jobs[jobId].results.push({ bvid, ...result });
      } catch (error) {
        jobs[jobId].results.push({ bvid, success: false, error: error.message });
      }
      jobs[jobId].progress = Math.round(((i + 1) / bvids.length) * 100);
    }

    jobs[jobId].status = 'completed';
    jobs[jobId].completedAt = new Date().toISOString();
  })();

  res.json({
    jobId,
    status: 'processing',
    message: `Started downloading ${bvids.length} videos`,
    statusUrl: `/api/bilibili/jobs/${jobId}`,
  });
});

// ═══════════════════════════════════════════════════════════════════════════════
// POST /api/bilibili/pipeline - 一键完整流水线
// ═══════════════════════════════════════════════════════════════════════════════

router.post('/pipeline', async (req, res) => {
  const {
    keyword,
    bvids,
    minScore = 50,
    maxVideos = 10,
    vlmBackend = 'mock',
    apiKey,
    outputDir,
    dryRun = false,
  } = req.body;

  if (!keyword && (!bvids || bvids.length === 0)) {
    return res.status(400).json({ error: 'keyword or bvids is required' });
  }

  const jobId = `pipeline_${Date.now()}`;
  const targetOutputDir = outputDir || path.join(__dirname, '../uploads/sft_output', jobId);

  jobs[jobId] = {
    id: jobId,
    status: 'processing',
    stage: 'initializing',
    progress: 0,
    results: {},
    createdAt: new Date().toISOString(),
  };

  // 异步处理流水线
  (async () => {
    try {
      let targetBvids = bvids || [];

      // Stage 1: 搜索（使用 Hunt API 而不是 Search，避免被封）
      if (keyword && !bvids) {
        jobs[jobId].stage = 'searching';
        jobs[jobId].progress = 5;
        
        // 使用 Hunt API（更好的请求头，避免被封）
        const huntScript = path.join(BILI_SERVICES_DIR, 'bili_hunter_agent.py');
        const huntResult = await callPython(huntScript, [
          keyword,  // 直接使用关键词作为意图
          '--min-score', '30',
          '--max-results', String(maxVideos * 2),
          '--json',
        ], 60000);
        
        if (huntResult.bvids && huntResult.bvids.length > 0) {
          targetBvids = huntResult.bvids;
          jobs[jobId].results.search = {
            total: huntResult.total_found || huntResult.bvids.length,
            found: targetBvids.length,
          };
        } else {
          // 如果 Hunt 失败，使用 mock 数据继续演示
          console.log('[Pipeline] Hunt returned no results, using mock BVIDs for demo');
          targetBvids = ['BV1WLPGzvEjJ', 'BV11Z4y1671T', 'BV16b421h7tg'];
          jobs[jobId].results.search = {
            total: 3,
            found: 3,
            mock: true,
          };
        }
      }

      // Stage 2: 预筛选
      jobs[jobId].stage = 'prescreening';
      jobs[jobId].progress = 15;
      
      const prescreenScript = path.join(BILI_SERVICES_DIR, 'prescreen.py');
      const prescreenResult = await callPython(
        prescreenScript,
        [...targetBvids, '--min-score', String(minScore), '--json'],
        60000 * targetBvids.length
      );
      
      const passedBvids = prescreenResult.recommended?.slice(0, maxVideos) || targetBvids.slice(0, maxVideos);
      jobs[jobId].results.prescreen = {
        total: targetBvids.length,
        passed: passedBvids.length,
      };

      // Stage 3: 下载
      jobs[jobId].stage = 'downloading';
      jobs[jobId].progress = 25;
      
      const videosDir = path.join(targetOutputDir, 'videos');
      const downloadScript = path.join(BILI_SERVICES_DIR, 'video_downloader.py');
      const downloadedVideos = [];

      for (let i = 0; i < passedBvids.length; i++) {
        const bvid = passedBvids[i];
        try {
          const dlResult = await callPython(downloadScript, [bvid, '--output', videosDir], 600000);
          if (dlResult.success && dlResult.filepath) {
            downloadedVideos.push(dlResult.filepath);
          }
        } catch (e) {
          console.error(`Download failed for ${bvid}:`, e.message);
        }
        jobs[jobId].progress = 25 + Math.round((i + 1) / passedBvids.length * 30);
      }
      
      jobs[jobId].results.download = {
        total: passedBvids.length,
        downloaded: downloadedVideos.length,
        videos: downloadedVideos,
      };

      // Stage 4: VLM标注 + LeRobot导出
      jobs[jobId].stage = 'labeling';
      jobs[jobId].progress = 60;

      const sftScript = path.join(SFT_SERVICES_DIR, 'sft_pipeline.py');
      const sftArgs = [
        ...downloadedVideos,  // 视频文件直接作为位置参数
        '--output', targetOutputDir,
        '--vlm', dryRun ? 'mock' : vlmBackend,
        '--task', 'demo_task',
      ];
      
      if (dryRun) {
        sftArgs.push('--dry-run');
      }
      
      sftArgs.push('--json');

      const sftResult = await callPython(sftScript, sftArgs, 600000 * downloadedVideos.length);
      jobs[jobId].results.sft = sftResult;

      // 完成
      jobs[jobId].status = 'completed';
      jobs[jobId].progress = 100;
      jobs[jobId].stage = 'completed';
      jobs[jobId].outputDir = targetOutputDir;
      jobs[jobId].completedAt = new Date().toISOString();

    } catch (error) {
      console.error('[Pipeline Error]', error);
      jobs[jobId].status = 'failed';
      jobs[jobId].error = error.message;
    }
  })();

  res.json({
    jobId,
    status: 'processing',
    message: 'Pipeline started',
    statusUrl: `/api/bilibili/jobs/${jobId}`,
  });
});

// ═══════════════════════════════════════════════════════════════════════════════
// GET /api/bilibili/jobs/:jobId - 查询任务状态
// ═══════════════════════════════════════════════════════════════════════════════

router.get('/jobs/:jobId', (req, res) => {
  const job = jobs[req.params.jobId];
  
  if (!job) {
    return res.status(404).json({ error: 'Job not found' });
  }

  res.json(job);
});

// ═══════════════════════════════════════════════════════════════════════════════
// GET /api/bilibili/video/:bvid - 获取视频信息
// ═══════════════════════════════════════════════════════════════════════════════

router.get('/video/:bvid', async (req, res) => {
  const { bvid } = req.params;

  try {
    const scriptPath = path.join(BILI_SERVICES_DIR, 'bili_intel.py');
    const result = await callPython(scriptPath, ['info', bvid], 15000);
    res.json(result);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

module.exports = router;
