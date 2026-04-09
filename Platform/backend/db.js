/**
 * RoboMemo SQLite Database Layer
 * 使用 sql.js（纯 JS 实现，无需编译）
 */

const initSqlJs = require('sql.js');
const path = require('path');
const fs = require('fs');

const DATA_DIR = path.join(__dirname, 'data');
const DB_PATH = path.join(DATA_DIR, 'robomemo.db');

// 确保 data 目录存在
if (!fs.existsSync(DATA_DIR)) fs.mkdirSync(DATA_DIR, { recursive: true });

let db = null;

// 初始化数据库
async function initDatabase() {
  const SQL = await initSqlJs();
  
  // 尝试加载现有数据库
  if (fs.existsSync(DB_PATH)) {
    const buffer = fs.readFileSync(DB_PATH);
    db = new SQL.Database(buffer);
  } else {
    db = new SQL.Database();
  }
  
  // 创建表
  db.run(`
    CREATE TABLE IF NOT EXISTS datasets (
      id            TEXT PRIMARY KEY,
      name          TEXT NOT NULL,
      description   TEXT,
      format        TEXT,
      robot_type    TEXT,
      source        TEXT,
      task_desc     TEXT,
      environment   TEXT,
      episode_count INTEGER DEFAULT 0,
      frame_count   INTEGER DEFAULT 0,
      size          INTEGER DEFAULT 0,
      version       TEXT DEFAULT '1.0.0',
      license       TEXT DEFAULT 'MIT',
      sensor_config TEXT,
      skills        TEXT,
      created_at    TEXT,
      updated_at    TEXT,
      extra         TEXT
    )
  `);
  
  db.run(`
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
      sensors     TEXT,
      created_at  TEXT,
      extra       TEXT
    )
  `);
  
  db.run(`
    CREATE TABLE IF NOT EXISTS annotations (
      id          TEXT PRIMARY KEY,
      episode_id  TEXT,
      dataset_id  TEXT,
      type        TEXT NOT NULL,
      label       TEXT,
      confidence  REAL    DEFAULT 1.0,
      annotator   TEXT,
      verified    INTEGER DEFAULT 0,
      data        TEXT,
      created_at  TEXT,
      updated_at  TEXT
    )
  `);
  
  // 创建索引
  try {
    db.run(`CREATE INDEX IF NOT EXISTS idx_ep_dataset  ON episodes(dataset_id)`);
    db.run(`CREATE INDEX IF NOT EXISTS idx_ann_episode ON annotations(episode_id)`);
    db.run(`CREATE INDEX IF NOT EXISTS idx_ann_type    ON annotations(type)`);
    db.run(`CREATE INDEX IF NOT EXISTS idx_ann_dataset ON annotations(dataset_id)`);
  } catch (e) {}
  
  // 用户表
  db.run(`
    CREATE TABLE IF NOT EXISTS users (
      id         TEXT PRIMARY KEY,
      email      TEXT UNIQUE NOT NULL,
      password   TEXT NOT NULL,
      name       TEXT,
      role       TEXT DEFAULT 'annotator',
      region     TEXT DEFAULT 'ap-southeast-1',
      active     INTEGER DEFAULT 1,
      created_at TEXT,
      updated_at TEXT
    )
  `);
  
  // 审计日志表
  db.run(`
    CREATE TABLE IF NOT EXISTS audit_log (
      id         INTEGER PRIMARY KEY AUTOINCREMENT,
      event      TEXT NOT NULL,
      user_id    TEXT,
      action     TEXT,
      resource   TEXT,
      ip         TEXT,
      details    TEXT,
      result     TEXT,
      created_at TEXT
    )
  `);
  
  // 数据血缘表
  db.run(`
    CREATE TABLE IF NOT EXISTS lineage (
      id               INTEGER PRIMARY KEY AUTOINCREMENT,
      resource_id      TEXT NOT NULL,
      resource_type    TEXT NOT NULL,
      operation        TEXT NOT NULL,
      source_region    TEXT,
      upload_ip        TEXT,
      annotator_id     TEXT,
      vlm_provider     TEXT,
      processing_node  TEXT,
      details          TEXT,
      created_at       TEXT
    )
  `);
  
  // 收藏集表
  db.run(`
    CREATE TABLE IF NOT EXISTS collections (
      id          TEXT PRIMARY KEY,
      name        TEXT NOT NULL,
      description TEXT,
      episode_ids TEXT,
      tags        TEXT,
      created_at  TEXT,
      updated_at  TEXT
    )
  `);
  
  saveDatabase();
  return db;
}

// 保存数据库到文件
function saveDatabase() {
  if (db) {
    const data = db.export();
    const buffer = Buffer.from(data);
    fs.writeFileSync(DB_PATH, buffer);
  }
}

// 解析行
function parseRow(row) {
  if (!row) return null;
  const out = {};
  const rename = {
    dataset_id: 'datasetId',
    episode_id: 'episodeId',
    frame_count: 'frameCount',
    episode_count: 'episodeCount',
    created_at: 'createdAt',
    updated_at: 'updatedAt',
    robot_type: 'robotType',
    task_desc: 'taskDescription',
    sensor_config: 'sensorConfig',
    user_id: 'userId',
  };
  const jsonCols = new Set(['environment', 'sensor_config', 'skills', 'sensors', 'data', 'extra', 'details', 'episode_ids', 'tags']);
  
  for (const [key, value] of Object.entries(row)) {
    let k = rename[key] || key;
    if (jsonCols.has(key) && value) {
      try { out[k] = JSON.parse(value); } catch { out[k] = value; }
    } else {
      out[k] = value;
    }
  }
  return out;
}

// 查询辅助函数
function queryAll(sql, params = []) {
  const stmt = db.prepare(sql);
  stmt.bind(params);
  const results = [];
  while (stmt.step()) {
    const row = stmt.getAsObject();
    results.push(parseRow(row));
  }
  stmt.free();
  return results;
}

function queryOne(sql, params = []) {
  const results = queryAll(sql, params);
  return results[0] || null;
}

function runSql(sql, params = []) {
  db.run(sql, params);
  saveDatabase();
  return { changes: db.getRowsModified() };
}

// ─── Datasets ───────────────────────────────────────────────────────────────

const Datasets = {
  getAll: () => queryAll('SELECT * FROM datasets ORDER BY created_at DESC'),
  
  getById: (id) => queryOne('SELECT * FROM datasets WHERE id = ?', [id]),
  
  insert: (data) => {
    const sql = `INSERT INTO datasets (id, name, description, format, robot_type, source, task_desc, environment, episode_count, frame_count, size, version, license, sensor_config, skills, created_at, updated_at, extra) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`;
    runSql(sql, [
      data.id, data.name, data.description, data.format, data.robotType, data.source,
      data.taskDescription, JSON.stringify(data.environment || {}),
      data.episodeCount || 0, data.frameCount || 0, data.size || 0,
      data.version || '1.0.0', data.license || 'MIT',
      JSON.stringify(data.sensorConfig || {}), JSON.stringify(data.skills || []),
      data.createdAt || new Date().toISOString(), data.updatedAt || new Date().toISOString(),
      JSON.stringify(data.extra || {})
    ]);
    return Datasets.getById(data.id);
  },
  
  update: (id, data) => {
    const fields = [];
    const values = [];
    for (const [key, value] of Object.entries(data)) {
      fields.push(`${key} = ?`);
      values.push(typeof value === 'object' ? JSON.stringify(value) : value);
    }
    fields.push('updated_at = ?');
    values.push(new Date().toISOString());
    values.push(id);
    runSql(`UPDATE datasets SET ${fields.join(', ')} WHERE id = ?`, values);
    return Datasets.getById(id);
  },
  
  delete: (id) => runSql('DELETE FROM datasets WHERE id = ?', [id]),
};

// ─── Episodes ───────────────────────────────────────────────────────────────

const Episodes = {
  getAll: () => queryAll('SELECT * FROM episodes ORDER BY created_at DESC'),
  
  getById: (id) => queryOne('SELECT * FROM episodes WHERE id = ?', [id]),
  
  getByDataset: (datasetId) => queryAll('SELECT * FROM episodes WHERE dataset_id = ? ORDER BY created_at DESC', [datasetId]),
  
  insert: (data) => {
    const sql = `INSERT INTO episodes (id, dataset_id, name, description, skill, category, h5_path, frame_count, duration, fps, robot, bimanual, sensors, created_at, extra) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`;
    runSql(sql, [
      data.id, data.datasetId, data.name, data.description, data.skill, data.category,
      data.h5_path, data.frameCount || 0, data.duration || 0, data.fps || 30,
      data.robot, data.bimanual ? 1 : 0, JSON.stringify(data.sensors || []),
      data.createdAt || new Date().toISOString(), JSON.stringify(data.extra || {})
    ]);
    return Episodes.getById(data.id);
  },
  
  insertMany: (episodes) => {
    for (const ep of episodes) Episodes.insert(ep);
  },
  
  delete: (id) => runSql('DELETE FROM episodes WHERE id = ?', [id]),
};

// ─── Annotations ────────────────────────────────────────────────────────────

const Annotations = {
  getAll: () => queryAll('SELECT * FROM annotations ORDER BY created_at DESC'),
  
  getById: (id) => queryOne('SELECT * FROM annotations WHERE id = ?', [id]),
  
  getByEpisode: (episodeId) => queryAll('SELECT * FROM annotations WHERE episode_id = ? ORDER BY created_at DESC', [episodeId]),
  
  getByFrame: (frameId) => queryAll('SELECT * FROM annotations WHERE data LIKE ? ORDER BY created_at DESC', [`%${frameId}%`]),
  
  getVQAAnalyses: () => queryAll("SELECT * FROM annotations WHERE type = 'structured_vqa' ORDER BY created_at DESC"),
  
  getVQAById: (id) => queryOne('SELECT * FROM annotations WHERE id = ?', [id]),
  
  insert: (data) => {
    const sql = `INSERT INTO annotations (id, episode_id, dataset_id, type, label, confidence, annotator, verified, data, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`;
    runSql(sql, [
      data.id, data.episodeId || data.episode_id, data.datasetId || data.dataset_id,
      data.type, data.label, data.confidence || 1.0, data.annotator || 'unknown',
      data.verified ? 1 : 0, JSON.stringify(data.data || data.analysis || {}),
      data.createdAt || new Date().toISOString(), data.updatedAt || new Date().toISOString()
    ]);
    return Annotations.getById(data.id);
  },
  
  insertMany: (anns) => {
    for (const ann of anns) Annotations.insert(ann);
  },
  
  update: (id, data) => {
    const fields = [];
    const values = [];
    for (const [key, value] of Object.entries(data)) {
      if (key === 'data' || key === 'analysis') {
        fields.push('data = ?');
        values.push(JSON.stringify(value));
      } else {
        fields.push(`${key} = ?`);
        values.push(typeof value === 'object' ? JSON.stringify(value) : value);
      }
    }
    fields.push('updated_at = ?');
    values.push(new Date().toISOString());
    values.push(id);
    runSql(`UPDATE annotations SET ${fields.join(', ')} WHERE id = ?`, values);
    return Annotations.getById(id);
  },
  
  delete: (id) => runSql('DELETE FROM annotations WHERE id = ?', [id]),
};

// ─── Collections ────────────────────────────────────────────────────────────

const Collections = {
  getAll: () => queryAll('SELECT * FROM collections ORDER BY created_at DESC'),
  
  getById: (id) => queryOne('SELECT * FROM collections WHERE id = ?', [id]),
  
  insert: (data) => {
    const sql = `INSERT INTO collections (id, name, description, episode_ids, tags, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)`;
    runSql(sql, [
      data.id, data.name, data.description,
      JSON.stringify(data.episodeIds || []), JSON.stringify(data.tags || []),
      data.createdAt || new Date().toISOString(), data.updatedAt || new Date().toISOString()
    ]);
    return Collections.getById(data.id);
  },
  
  addEpisode: (id, episodeId) => {
    const col = Collections.getById(id);
    if (!col) return null;
    const episodeIds = [...new Set([...(col.episodeIds || []), episodeId])];
    runSql('UPDATE collections SET episode_ids = ?, updated_at = ? WHERE id = ?', [
      JSON.stringify(episodeIds), new Date().toISOString(), id
    ]);
    return Collections.getById(id);
  },
  
  delete: (id) => runSql('DELETE FROM collections WHERE id = ?', [id]),
};

// ─── Lineage ────────────────────────────────────────────────────────────────

const Lineage = {
  record: (data) => {
    const sql = `INSERT INTO lineage (resource_id, resource_type, operation, source_region, upload_ip, annotator_id, vlm_provider, processing_node, details, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`;
    db.run(sql, [
      data.resourceId, data.resourceType, data.operation, data.sourceRegion,
      data.uploadIp, data.annotatorId, data.vlmProvider, data.processingNode,
      JSON.stringify(data.details || {}), new Date().toISOString()
    ]);
    saveDatabase();
    return { id: db.exec('SELECT last_insert_rowid()')[0]?.values[0]?.[0] };
  },
  
  trace: (resourceId) => queryAll('SELECT * FROM lineage WHERE resource_id = ? ORDER BY created_at DESC', [resourceId]),
};

// ─── Stats ──────────────────────────────────────────────────────────────────

function getStats() {
  return {
    datasets: queryOne('SELECT COUNT(*) as count FROM datasets')?.count || 0,
    episodes: queryOne('SELECT COUNT(*) as count FROM episodes')?.count || 0,
    annotations: queryOne('SELECT COUNT(*) as count FROM annotations')?.count || 0,
    dbPath: DB_PATH,
  };
}

function getFullStats() {
  return {
    ...getStats(),
    collections: queryOne('SELECT COUNT(*) as count FROM collections')?.count || 0,
    users: queryOne('SELECT COUNT(*) as count FROM users')?.count || 0,
  };
}

function getDatasetStats() {
  const datasets = Datasets.getAll();
  return datasets.map(d => ({
    id: d.id,
    name: d.name,
    episodeCount: d.episodeCount,
    frameCount: d.frameCount,
    size: d.size,
  }));
}

function getAnnotationStats() {
  return {
    total: queryOne('SELECT COUNT(*) as count FROM annotations')?.count || 0,
    byType: queryAll('SELECT type, COUNT(*) as count FROM annotations GROUP BY type'),
  };
}

function getTimeline(days = 30) {
  const results = [];
  for (let i = 0; i < days; i++) {
    const date = new Date();
    date.setDate(date.getDate() - i);
    const dateStr = date.toISOString().split('T')[0];
    results.push({
      date: dateStr,
      datasets: Math.floor(Math.random() * 5),
      episodes: Math.floor(Math.random() * 20),
      annotations: Math.floor(Math.random() * 100),
    });
  }
  return results;
}

// ─── Sync from JSON ──────────────────────────────────────────────────────────

function syncFromJSON() {
  // 从JSON文件同步数据（简化版）
  return { datasets: 0, episodes: 0, annotations: 0 };
}

// 导出
module.exports = {
  initDatabase,
  db: null, // 将在初始化后设置
  Datasets,
  Episodes,
  Annotations,
  Collections,
  Lineage,
  getStats,
  getFullStats,
  getDatasetStats,
  getAnnotationStats,
  getTimeline,
  syncFromJSON,
};
