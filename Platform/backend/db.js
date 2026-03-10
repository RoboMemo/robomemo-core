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

module.exports = { db, Datasets, Episodes, Annotations, syncFromJSON, getStats };
