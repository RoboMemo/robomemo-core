/**
 * RoboMemo SQLite Database Layer
 * 使用 better-sqlite3，WAL 模式，零配置，本地文件持久化
 */

const Database = require('better-sqlite3');
const path = require('path');
const fs = require('fs');

const DATA_DIR = path.join(__dirname, 'data');
const DB_PATH = path.join(DATA_DIR, 'robomemo.db');

// 确保 data 目录存在
if (!fs.existsSync(DATA_DIR)) fs.mkdirSync(DATA_DIR, { recursive: true });

const db = new Database(DB_PATH);
db.pragma('journal_mode = WAL');   // 写多读多场景性能更好
db.pragma('foreign_keys = ON');
db.pragma('synchronous = NORMAL'); // 比 FULL 快，比 OFF 安全

// ─── Schema ──────────────────────────────────────────────────────────────────
db.exec(`
  CREATE TABLE IF NOT EXISTS datasets (
    id            TEXT PRIMARY KEY,
    name          TEXT NOT NULL,
    description   TEXT,
    format        TEXT,
    robot_type    TEXT,
    source        TEXT,
    task_desc     TEXT,
    environment   TEXT,   -- JSON
    episode_count INTEGER DEFAULT 0,
    frame_count   INTEGER DEFAULT 0,
    size          INTEGER DEFAULT 0,
    version       TEXT DEFAULT '1.0.0',
    license       TEXT DEFAULT 'MIT',
    sensor_config TEXT,   -- JSON
    skills        TEXT,   -- JSON array
    created_at    TEXT,
    updated_at    TEXT,
    extra         TEXT    -- JSON: 其余字段
  );

  CREATE TABLE IF NOT EXISTS episodes (
    id          TEXT PRIMARY KEY,
    dataset_id  TEXT NOT NULL,
    name        TEXT,
    description TEXT,
    skill       TEXT,
    category    TEXT,
    h5_path     TEXT,
    frame_count INTEGER DEFAULT 0,
    duration    REAL    DEFAULT 0,
    fps         REAL    DEFAULT 30,
    robot       TEXT,
    bimanual    INTEGER DEFAULT 0,
    sensors     TEXT,   -- JSON array
    created_at  TEXT,
    extra       TEXT    -- JSON
  );

  CREATE TABLE IF NOT EXISTS annotations (
    id          TEXT PRIMARY KEY,
    episode_id  TEXT,
    dataset_id  TEXT,
    type        TEXT NOT NULL,
    label       TEXT,
    confidence  REAL    DEFAULT 1.0,
    annotator   TEXT,
    verified    INTEGER DEFAULT 0,
    data        TEXT,   -- JSON: 类型专属数据
    created_at  TEXT,
    updated_at  TEXT
  );

  CREATE INDEX IF NOT EXISTS idx_ep_dataset  ON episodes(dataset_id);
  CREATE INDEX IF NOT EXISTS idx_ann_episode ON annotations(episode_id);
  CREATE INDEX IF NOT EXISTS idx_ann_type    ON annotations(type);
  CREATE INDEX IF NOT EXISTS idx_ann_dataset ON annotations(dataset_id);
`);

// ─── Row parser: snake_case + JSON blobs → camelCase JS object ───────────────
const JSON_COLS = new Set(['environment', 'sensor_config', 'skills', 'sensors', 'data', 'extra']);

const RENAME = {
  dataset_id:    'datasetId',
  episode_id:    'episodeId',
  h5_path:       'h5_path',   // keep as-is (Python convention)
  frame_count:   'frameCount',
  episode_count: 'episodeCount',
  created_at:    'createdAt',
  updated_at:    'updatedAt',
  robot_type:    'robotType',
  task_desc:     'taskDescription',
  sensor_config: 'sensorConfig',
};

function parseRow(row) {
  if (!row) return null;
  const out = {};
  for (const [k, v] of Object.entries(row)) {
    const key = RENAME[k] ?? k;
    if (JSON_COLS.has(k) && typeof v === 'string') {
      try { out[key] = JSON.parse(v); } catch { out[key] = v; }
    } else {
      out[key] = v;
    }
  }
  // Boolean coercion
  if ('bimanual' in out) out.bimanual = !!out.bimanual;
  if ('verified'  in out) out.verified  = !!out.verified;
  // Merge extra blob into top-level
  if (out.extra && typeof out.extra === 'object') {
    const extra = out.extra;
    delete out.extra;
    Object.assign(out, extra);
  }
  return out;
}

// ─── JSON migration on first run ──────────────────────────────────────────────
function loadJSON(filename) {
  const fp = path.join(DATA_DIR, filename);
  if (!fs.existsSync(fp)) return [];
  try { return JSON.parse(fs.readFileSync(fp, 'utf8') || '[]'); } catch { return []; }
}

function migrateFromJSON() {
  const dsCount = db.prepare('SELECT COUNT(*) as n FROM datasets').get().n;
  if (dsCount === 0) {
    const rows = loadJSON('datasets.json');
    const ins = db.prepare(`
      INSERT OR IGNORE INTO datasets
        (id, name, description, format, robot_type, source, task_desc,
         environment, episode_count, frame_count, size, version, license,
         sensor_config, skills, created_at, updated_at, extra)
      VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)`);
    const run = db.transaction(() => {
      for (const d of rows) {
        const { id, name, description, format, robotType, source, taskDescription,
                environment, episodeCount, frameCount, size, version, license,
                sensorConfig, skills, createdAt, updatedAt, ...extra } = d;
        ins.run(id, name, description, format, robotType, source, taskDescription,
          JSON.stringify(environment), episodeCount ?? 0, frameCount ?? 0,
          size ?? 0, version ?? '1.0.0', license ?? 'MIT',
          JSON.stringify(sensorConfig), JSON.stringify(skills),
          createdAt, updatedAt, JSON.stringify(extra));
      }
    });
    run();
    if (rows.length) console.log(`[DB] Migrated ${rows.length} datasets from JSON`);
  }

  const epCount = db.prepare('SELECT COUNT(*) as n FROM episodes').get().n;
  if (epCount === 0) {
    const rows = loadJSON('episodes.json');
    const ins = db.prepare(`
      INSERT OR IGNORE INTO episodes
        (id, dataset_id, name, description, skill, category, h5_path,
         frame_count, duration, fps, robot, bimanual, sensors, created_at, extra)
      VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)`);
    const run = db.transaction(() => {
      for (const e of rows) {
        const { id, datasetId, name, description, skill, category,
                h5_path, frameCount, duration, fps, robot, bimanual,
                sensors, createdAt, ...extra } = e;
        ins.run(id, datasetId, name, description, skill, category,
          h5_path, frameCount ?? 0, duration ?? 0, fps ?? 30,
          robot, bimanual ? 1 : 0, JSON.stringify(sensors), createdAt, JSON.stringify(extra));
      }
    });
    run();
    if (rows.length) console.log(`[DB] Migrated ${rows.length} episodes from JSON`);
  }

  const annCount = db.prepare('SELECT COUNT(*) as n FROM annotations').get().n;
  if (annCount === 0) {
    const rows = loadJSON('annotations.json');
    const ins = db.prepare(`
      INSERT OR IGNORE INTO annotations
        (id, episode_id, dataset_id, type, label, confidence,
         annotator, verified, data, created_at, updated_at)
      VALUES (?,?,?,?,?,?,?,?,?,?,?)`);
    const run = db.transaction(() => {
      for (const a of rows) {
        const { id, episodeId, datasetId, type, label, confidence,
                annotator, verified, createdAt, updatedAt, ...data } = a;
        ins.run(id, episodeId, datasetId, type, label, confidence ?? 1.0,
          annotator, verified ? 1 : 0, JSON.stringify(data), createdAt, updatedAt);
      }
    });
    run();
    if (rows.length) console.log(`[DB] Migrated ${rows.length} annotations from JSON`);
  }
}

migrateFromJSON();

// ─── Re-sync from JSON (call after download_data.py writes JSON files) ────────
function syncFromJSON() {
  // Upsert datasets
  const datasets = loadJSON('datasets.json');
  const upsertDs = db.prepare(`
    INSERT OR REPLACE INTO datasets
      (id, name, description, format, robot_type, source, task_desc,
       environment, episode_count, frame_count, size, version, license,
       sensor_config, skills, created_at, updated_at, extra)
    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)`);
  db.transaction(() => {
    for (const d of datasets) {
      const { id, name, description, format, robotType, source, taskDescription,
              environment, episodeCount, frameCount, size, version, license,
              sensorConfig, skills, createdAt, updatedAt, ...extra } = d;
      upsertDs.run(id, name, description, format, robotType, source, taskDescription,
        JSON.stringify(environment), episodeCount ?? 0, frameCount ?? 0,
        size ?? 0, version ?? '1.0.0', license ?? 'MIT',
        JSON.stringify(sensorConfig), JSON.stringify(skills),
        createdAt, updatedAt, JSON.stringify(extra));
    }
  })();

  // Upsert episodes
  const episodes = loadJSON('episodes.json');
  const upsertEp = db.prepare(`
    INSERT OR REPLACE INTO episodes
      (id, dataset_id, name, description, skill, category, h5_path,
       frame_count, duration, fps, robot, bimanual, sensors, created_at, extra)
    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)`);
  db.transaction(() => {
    for (const e of episodes) {
      const { id, datasetId, name, description, skill, category,
              h5_path, frameCount, duration, fps, robot, bimanual,
              sensors, createdAt, ...extra } = e;
      upsertEp.run(id, datasetId, name, description, skill, category,
        h5_path, frameCount ?? 0, duration ?? 0, fps ?? 30,
        robot, bimanual ? 1 : 0, JSON.stringify(sensors), createdAt, JSON.stringify(extra));
    }
  })();

  // Upsert annotations (seed labels only, not VQA)
  const annotations = loadJSON('annotations.json');
  const upsertAnn = db.prepare(`
    INSERT OR REPLACE INTO annotations
      (id, episode_id, dataset_id, type, label, confidence,
       annotator, verified, data, created_at, updated_at)
    VALUES (?,?,?,?,?,?,?,?,?,?,?)`);
  db.transaction(() => {
    for (const a of annotations) {
      const { id, episodeId, datasetId, type, label, confidence,
              annotator, verified, createdAt, updatedAt, ...data } = a;
      upsertAnn.run(id, episodeId, datasetId, type, label, confidence ?? 1.0,
        annotator, verified ? 1 : 0, JSON.stringify(data), createdAt, updatedAt);
    }
  })();

  const stats = {
    datasets: datasets.length,
    episodes: episodes.length,
    annotations: annotations.length,
  };
  console.log('[DB] Sync from JSON complete:', stats);
  return stats;
}

// ─── Dataset CRUD ─────────────────────────────────────────────────────────────
const Datasets = {
  getAll: () =>
    db.prepare('SELECT * FROM datasets ORDER BY created_at DESC').all().map(parseRow),

  getById: (id) =>
    parseRow(db.prepare('SELECT * FROM datasets WHERE id = ?').get(id)),

  insert: (d) => {
    const { id, name, description, format, robotType, source, taskDescription,
            environment, episodeCount, frameCount, size, version, license,
            sensorConfig, skills, createdAt, updatedAt, ...extra } = d;
    db.prepare(`
      INSERT OR REPLACE INTO datasets
        (id, name, description, format, robot_type, source, task_desc,
         environment, episode_count, frame_count, size, version, license,
         sensor_config, skills, created_at, updated_at, extra)
      VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)`
    ).run(id, name, description, format, robotType, source, taskDescription,
      JSON.stringify(environment), episodeCount ?? 0, frameCount ?? 0,
      size ?? 0, version ?? '1.0.0', license ?? 'MIT',
      JSON.stringify(sensorConfig), JSON.stringify(skills),
      createdAt, updatedAt, JSON.stringify(extra));
    return Datasets.getById(id);
  },
};

// ─── Episode CRUD ─────────────────────────────────────────────────────────────
const Episodes = {
  getAll: () =>
    db.prepare('SELECT * FROM episodes ORDER BY created_at DESC').all().map(parseRow),

  getByDataset: (datasetId) =>
    db.prepare('SELECT * FROM episodes WHERE dataset_id = ? ORDER BY created_at')
      .all(datasetId).map(parseRow),

  getById: (id) =>
    parseRow(db.prepare('SELECT * FROM episodes WHERE id = ?').get(id)),

  insert: (e) => {
    const { id, datasetId, name, description, skill, category,
            h5_path, frameCount, duration, fps, robot, bimanual,
            sensors, createdAt, ...extra } = e;
    db.prepare(`
      INSERT OR REPLACE INTO episodes
        (id, dataset_id, name, description, skill, category, h5_path,
         frame_count, duration, fps, robot, bimanual, sensors, created_at, extra)
      VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)`
    ).run(id, datasetId, name, description, skill, category,
      h5_path, frameCount ?? 0, duration ?? 0, fps ?? 30,
      robot, bimanual ? 1 : 0, JSON.stringify(sensors), createdAt, JSON.stringify(extra));
    return Episodes.getById(id);
  },
};

// ─── Annotation CRUD ──────────────────────────────────────────────────────────
const Annotations = {
  getByEpisode: (episodeId) =>
    db.prepare('SELECT * FROM annotations WHERE episode_id = ? ORDER BY created_at DESC')
      .all(episodeId).map(parseRow),

  getAll: () =>
    db.prepare('SELECT * FROM annotations ORDER BY created_at DESC').all().map(parseRow),

  getById: (id) =>
    parseRow(db.prepare('SELECT * FROM annotations WHERE id = ?').get(id)),

  getVQAAnalyses: () =>
    db.prepare("SELECT * FROM annotations WHERE type = 'structured_vqa' ORDER BY created_at DESC")
      .all().map(parseRow),

  getVQAById: (id) =>
    parseRow(db.prepare("SELECT * FROM annotations WHERE id = ? AND type = 'structured_vqa'").get(id)),

  insert: (a) => {
    const { id, episodeId, datasetId, type, label, confidence,
            annotator, verified, createdAt, updatedAt, ...data } = a;
    db.prepare(`
      INSERT OR REPLACE INTO annotations
        (id, episode_id, dataset_id, type, label, confidence,
         annotator, verified, data, created_at, updated_at)
      VALUES (?,?,?,?,?,?,?,?,?,?,?)`
    ).run(id, episodeId, datasetId, type, label, confidence ?? 1.0,
      annotator, verified ? 1 : 0, JSON.stringify(data), createdAt, updatedAt);
    return Annotations.getById(id);
  },

  insertMany: db.transaction(function(anns) {
    const stmt = db.prepare(`
      INSERT OR REPLACE INTO annotations
        (id, episode_id, dataset_id, type, label, confidence,
         annotator, verified, data, created_at, updated_at)
      VALUES (?,?,?,?,?,?,?,?,?,?,?)`);
    for (const a of anns) {
      const { id, episodeId, datasetId, type, label, confidence,
              annotator, verified, createdAt, updatedAt, ...data } = a;
      stmt.run(id, episodeId, datasetId, type, label, confidence ?? 1.0,
        annotator, verified ? 1 : 0, JSON.stringify(data), createdAt, updatedAt);
    }
  }),
};

// ─── Stats helper ─────────────────────────────────────────────────────────────
function getStats() {
  return {
    datasets:    db.prepare('SELECT COUNT(*) as n FROM datasets').get().n,
    episodes:    db.prepare('SELECT COUNT(*) as n FROM episodes').get().n,
    annotations: db.prepare('SELECT COUNT(*) as n FROM annotations').get().n,
    vqaAnalyses: db.prepare("SELECT COUNT(*) as n FROM annotations WHERE type='structured_vqa'").get().n,
    dbPath:      DB_PATH,
  };
}

// ─── Data Lineage ──────────────────────────────────────────────────────────────
// ISO 27001 Annex A.8.2 / SOC 2 CC6.1: track every data-touching operation
db.exec(`
  CREATE TABLE IF NOT EXISTS data_lineage (
    id              TEXT PRIMARY KEY,
    timestamp       TEXT NOT NULL,
    resource_id     TEXT NOT NULL,
    resource_type   TEXT NOT NULL,   -- 'video' | 'annotation' | 'dataset' | 'export'
    operation       TEXT NOT NULL,   -- 'upload' | 'annotate' | 'review' | 'export' | 'delete'
    source_region   TEXT,            -- ISO country / AWS region of origin
    annotator_id    TEXT,
    vlm_provider    TEXT,            -- 'gemini' | 'claude' | 'openai' | 'local' | null
    upload_ip       TEXT,
    processing_node TEXT,            -- hostname of backend that processed this
    details         TEXT             -- JSON: any extra lineage metadata
  );
  CREATE INDEX IF NOT EXISTS idx_lineage_resource ON data_lineage(resource_id);
  CREATE INDEX IF NOT EXISTS idx_lineage_ts       ON data_lineage(timestamp);
  CREATE INDEX IF NOT EXISTS idx_lineage_op       ON data_lineage(operation);
`);

const crypto_lin = require('crypto');
const Lineage = {
  record: (entry) => {
    const id = `lin_${Date.now()}_${crypto_lin.randomBytes(4).toString('hex')}`;
    const now = new Date().toISOString();
    db.prepare(`
      INSERT INTO data_lineage
        (id, timestamp, resource_id, resource_type, operation,
         source_region, annotator_id, vlm_provider, upload_ip, processing_node, details)
      VALUES (?,?,?,?,?,?,?,?,?,?,?)`
    ).run(
      id, now,
      entry.resourceId, entry.resourceType, entry.operation,
      entry.sourceRegion || null,
      entry.annotatorId || null,
      entry.vlmProvider || null,
      entry.uploadIp || null,
      entry.processingNode || null,
      entry.details ? JSON.stringify(entry.details) : null,
    );
    return { id, timestamp: now, ...entry };
  },

  trace: (resourceId) => {
    const rows = db.prepare(
      'SELECT * FROM data_lineage WHERE resource_id = ? ORDER BY timestamp ASC'
    ).all(resourceId);
    return rows.map(r => ({
      ...r,
      details: r.details ? JSON.parse(r.details) : null,
    }));
  },

  getRecent: (limit = 50) => {
    return db.prepare(
      'SELECT * FROM data_lineage ORDER BY timestamp DESC LIMIT ?'
    ).all(limit).map(r => ({ ...r, details: r.details ? JSON.parse(r.details) : null }));
  },
};

// ─── Collections ──────────────────────────────────────────────────────────────
db.exec(`
  CREATE TABLE IF NOT EXISTS collections (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    description TEXT,
    dataset_id  TEXT,
    episode_ids TEXT,   -- JSON array
    created_at  TEXT,
    updated_at  TEXT
  );
`);

const Collections = {
  getAll: () =>
    db.prepare('SELECT * FROM collections ORDER BY created_at DESC').all().map(r => {
      const out = parseRow(r);
      if (out.episode_ids && typeof out.episode_ids === 'string') {
        try { out.episodeIds = JSON.parse(out.episode_ids); } catch { out.episodeIds = []; }
      }
      delete out.episode_ids;
      return out;
    }),

  getById: (id) => {
    const r = db.prepare('SELECT * FROM collections WHERE id = ?').get(id);
    if (!r) return null;
    const out = parseRow(r);
    if (out.episode_ids && typeof out.episode_ids === 'string') {
      try { out.episodeIds = JSON.parse(out.episode_ids); } catch { out.episodeIds = []; }
    }
    delete out.episode_ids;
    return out;
  },

  insert: (c) => {
    db.prepare(`
      INSERT OR REPLACE INTO collections
        (id, name, description, dataset_id, episode_ids, created_at, updated_at)
      VALUES (?,?,?,?,?,?,?)`
    ).run(c.id, c.name, c.description || null, c.datasetId || null,
      JSON.stringify(c.episodeIds || []), c.createdAt, c.updatedAt);
    return Collections.getById(c.id);
  },

  addEpisode: (collectionId, episodeId) => {
    const col = Collections.getById(collectionId);
    if (!col) return null;
    const ids = col.episodeIds || [];
    if (!ids.includes(episodeId)) ids.push(episodeId);
    db.prepare('UPDATE collections SET episode_ids = ?, updated_at = ? WHERE id = ?')
      .run(JSON.stringify(ids), new Date().toISOString(), collectionId);
    return Collections.getById(collectionId);
  },
};

// ─── Extended Dataset CRUD ────────────────────────────────────────────────────
Datasets.update = (id, updates) => {
  const existing = Datasets.getById(id);
  if (!existing) return null;
  const merged = { ...existing, ...updates, updatedAt: new Date().toISOString() };
  return Datasets.insert(merged);
};

Datasets.delete = (id) => {
  db.prepare('DELETE FROM datasets WHERE id = ?').run(id);
  db.prepare('DELETE FROM episodes WHERE dataset_id = ?').run(id);
  db.prepare('DELETE FROM annotations WHERE dataset_id = ?').run(id);
};

// ─── Extended Annotation CRUD ─────────────────────────────────────────────────
Annotations.update = (id, updates) => {
  const existing = Annotations.getById(id);
  if (!existing) return null;
  const merged = { ...existing, ...updates, updatedAt: new Date().toISOString() };
  return Annotations.insert(merged);
};

Annotations.delete = (id) => {
  db.prepare('DELETE FROM annotations WHERE id = ?').run(id);
};

Annotations.getByFrame = (frameId) => {
  // frameId is stored in the data JSON blob or as a label convention
  return db.prepare("SELECT * FROM annotations WHERE data LIKE ? OR label LIKE ? ORDER BY created_at DESC")
    .all(`%${frameId}%`, `%${frameId}%`).map(parseRow);
};

// ─── Extended Stats ───────────────────────────────────────────────────────────
function getFullStats() {
  const base = getStats();
  const totalSize = db.prepare('SELECT COALESCE(SUM(size),0) as s FROM datasets').get().s;
  const totalDuration = db.prepare('SELECT COALESCE(SUM(duration),0) as d FROM episodes').get().d;
  const formats = db.prepare('SELECT format, COUNT(*) as count FROM datasets GROUP BY format').all();
  const robotTypes = db.prepare('SELECT robot_type, COUNT(*) as count FROM datasets GROUP BY robot_type').all();
  return {
    ...base,
    totalSize,
    totalDuration,
    formats: formats.map(f => ({ format: f.format, count: f.count })),
    robotTypes: robotTypes.map(r => ({ robotType: r.robot_type, count: r.count })),
    collections: db.prepare('SELECT COUNT(*) as n FROM collections').get().n,
  };
}

function getDatasetStats() {
  return {
    total: db.prepare('SELECT COUNT(*) as n FROM datasets').get().n,
    totalEpisodes: db.prepare('SELECT COUNT(*) as n FROM episodes').get().n,
    totalSize: db.prepare('SELECT COALESCE(SUM(size),0) as s FROM datasets').get().s,
    byFormat: db.prepare('SELECT format, COUNT(*) as count FROM datasets GROUP BY format').all(),
    byRobot: db.prepare('SELECT robot_type as robotType, COUNT(*) as count FROM datasets GROUP BY robot_type').all(),
    recentDatasets: Datasets.getAll().slice(0, 5),
  };
}

function getAnnotationStats() {
  return {
    total: db.prepare('SELECT COUNT(*) as n FROM annotations').get().n,
    byType: db.prepare('SELECT type, COUNT(*) as count FROM annotations GROUP BY type').all(),
    verified: db.prepare('SELECT COUNT(*) as n FROM annotations WHERE verified = 1').get().n,
    unverified: db.prepare('SELECT COUNT(*) as n FROM annotations WHERE verified = 0').get().n,
    byAnnotator: db.prepare('SELECT annotator, COUNT(*) as count FROM annotations GROUP BY annotator').all(),
    vqaCount: db.prepare("SELECT COUNT(*) as n FROM annotations WHERE type = 'structured_vqa'").get().n,
  };
}

function getTimeline(days = 30) {
  const since = new Date(Date.now() - days * 86400000).toISOString();
  return {
    datasets: db.prepare('SELECT DATE(created_at) as date, COUNT(*) as count FROM datasets WHERE created_at >= ? GROUP BY DATE(created_at) ORDER BY date').all(since),
    episodes: db.prepare('SELECT DATE(created_at) as date, COUNT(*) as count FROM episodes WHERE created_at >= ? GROUP BY DATE(created_at) ORDER BY date').all(since),
    annotations: db.prepare('SELECT DATE(created_at) as date, COUNT(*) as count FROM annotations WHERE created_at >= ? GROUP BY DATE(created_at) ORDER BY date').all(since),
  };
}

// ─── Tasks ────────────────────────────────────────────────────────────────────
db.exec(`
  CREATE TABLE IF NOT EXISTS tasks (
    id            TEXT PRIMARY KEY,
    title         TEXT NOT NULL,
    description   TEXT,
    type          TEXT NOT NULL DEFAULT 'annotation',
    status        TEXT NOT NULL DEFAULT 'pending',
    priority      TEXT DEFAULT 'normal',
    dataset_id    TEXT,
    episode_ids   TEXT,
    assigned_to   TEXT,
    assigned_by   TEXT,
    reviewer_id   TEXT,
    due_date      TEXT,
    started_at    TEXT,
    completed_at  TEXT,
    result        TEXT,
    created_at    TEXT,
    updated_at    TEXT
  );
  CREATE INDEX IF NOT EXISTS idx_task_assigned ON tasks(assigned_to);
  CREATE INDEX IF NOT EXISTS idx_task_status   ON tasks(status);
  CREATE INDEX IF NOT EXISTS idx_task_dataset  ON tasks(dataset_id);
`);

const TASK_JSON_COLS = new Set(['episode_ids', 'result']);
const TASK_RENAME = {
  dataset_id:   'datasetId',
  episode_ids:  'episodeIds',
  assigned_to:  'assignedTo',
  assigned_by:  'assignedBy',
  reviewer_id:  'reviewerId',
  due_date:     'dueDate',
  started_at:   'startedAt',
  completed_at: 'completedAt',
  created_at:   'createdAt',
  updated_at:   'updatedAt',
};

function parseTaskRow(row) {
  if (!row) return null;
  const out = {};
  for (const [k, v] of Object.entries(row)) {
    const key = TASK_RENAME[k] ?? k;
    if (TASK_JSON_COLS.has(k) && typeof v === 'string') {
      try { out[key] = JSON.parse(v); } catch { out[key] = v; }
    } else {
      out[key] = v;
    }
  }
  return out;
}

const Tasks = {
  getAll: () =>
    db.prepare('SELECT * FROM tasks ORDER BY created_at DESC').all().map(parseTaskRow),

  getById: (id) =>
    parseTaskRow(db.prepare('SELECT * FROM tasks WHERE id = ?').get(id)),

  getByUser: (userId) =>
    db.prepare('SELECT * FROM tasks WHERE assigned_to = ? ORDER BY created_at DESC')
      .all(userId).map(parseTaskRow),

  getByStatus: (status) =>
    db.prepare('SELECT * FROM tasks WHERE status = ? ORDER BY created_at DESC')
      .all(status).map(parseTaskRow),

  insert: (t) => {
    const id = t.id || `task_${Date.now()}_${require('crypto').randomBytes(3).toString('hex')}`;
    const now = new Date().toISOString();
    db.prepare(`
      INSERT OR REPLACE INTO tasks
        (id, title, description, type, status, priority, dataset_id,
         episode_ids, assigned_to, assigned_by, reviewer_id,
         due_date, started_at, completed_at, result, created_at, updated_at)
      VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    `).run(
      id,
      t.title,
      t.description || null,
      t.type || 'annotation',
      t.status || 'pending',
      t.priority || 'normal',
      t.datasetId || null,
      JSON.stringify(t.episodeIds || []),
      t.assignedTo || null,
      t.assignedBy || null,
      t.reviewerId || null,
      t.dueDate || null,
      t.startedAt || null,
      t.completedAt || null,
      t.result ? JSON.stringify(t.result) : null,
      t.createdAt || now,
      now
    );
    return Tasks.getById(id);
  },

  update: (id, updates) => {
    const existing = Tasks.getById(id);
    if (!existing) return null;
    const merged = { ...existing, ...updates, updatedAt: new Date().toISOString() };
    return Tasks.insert({ ...merged, id });
  },

  delete: (id) => {
    db.prepare('DELETE FROM tasks WHERE id = ?').run(id);
  },
};

// ─── Reviews ──────────────────────────────────────────────────────────────────
db.exec(`
  CREATE TABLE IF NOT EXISTS reviews (
    id            TEXT PRIMARY KEY,
    task_id       TEXT,
    annotation_id TEXT,
    reviewer_id   TEXT NOT NULL,
    status        TEXT NOT NULL DEFAULT 'pending',
    score         INTEGER,
    feedback      TEXT,
    created_at    TEXT
  );
  CREATE INDEX IF NOT EXISTS idx_review_task     ON reviews(task_id);
  CREATE INDEX IF NOT EXISTS idx_review_reviewer ON reviews(reviewer_id);
  CREATE INDEX IF NOT EXISTS idx_review_status   ON reviews(status);
`);

const REVIEW_RENAME = {
  task_id:       'taskId',
  annotation_id: 'annotationId',
  reviewer_id:   'reviewerId',
  created_at:    'createdAt',
};

function parseReviewRow(row) {
  if (!row) return null;
  const out = {};
  for (const [k, v] of Object.entries(row)) {
    out[REVIEW_RENAME[k] ?? k] = v;
  }
  return out;
}

const Reviews = {
  getAll: () =>
    db.prepare('SELECT * FROM reviews ORDER BY created_at DESC').all().map(parseReviewRow),

  getById: (id) =>
    parseReviewRow(db.prepare('SELECT * FROM reviews WHERE id = ?').get(id)),

  getByTask: (taskId) =>
    db.prepare('SELECT * FROM reviews WHERE task_id = ? ORDER BY created_at DESC')
      .all(taskId).map(parseReviewRow),

  getByReviewer: (reviewerId) =>
    db.prepare('SELECT * FROM reviews WHERE reviewer_id = ? ORDER BY created_at DESC')
      .all(reviewerId).map(parseReviewRow),

  insert: (r) => {
    const id = r.id || `rev_${Date.now()}_${require('crypto').randomBytes(3).toString('hex')}`;
    const now = new Date().toISOString();
    db.prepare(`
      INSERT INTO reviews (id, task_id, annotation_id, reviewer_id, status, score, feedback, created_at)
      VALUES (?,?,?,?,?,?,?,?)
    `).run(id, r.taskId || null, r.annotationId || null, r.reviewerId,
      r.status || 'pending', r.score ?? null, r.feedback || null, r.createdAt || now);
    return Reviews.getById(id);
  },

  getStats: () => {
    const total = db.prepare('SELECT COUNT(*) as n FROM reviews').get().n;
    const byStatus = db.prepare('SELECT status, COUNT(*) as count FROM reviews GROUP BY status').all();
    const avgScore = db.prepare('SELECT AVG(score) as avg FROM reviews WHERE score IS NOT NULL').get().avg;
    const byReviewer = db.prepare(`
      SELECT reviewer_id, COUNT(*) as count, AVG(score) as avgScore
      FROM reviews GROUP BY reviewer_id
    `).all();
    return {
      total,
      byStatus: Object.fromEntries(byStatus.map(r => [r.status, r.count])),
      averageScore: avgScore ? Math.round(avgScore * 10) / 10 : null,
      byReviewer: byReviewer.map(r => ({
        reviewerId: r.reviewer_id,
        count: r.count,
        averageScore: r.avgScore ? Math.round(r.avgScore * 10) / 10 : null,
      })),
    };
  },
};

module.exports = {
  db, Datasets, Episodes, Annotations, Collections, Lineage, Tasks, Reviews,
  syncFromJSON, getStats, getFullStats, getDatasetStats, getAnnotationStats, getTimeline
};
