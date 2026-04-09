/**
 * ISO 27001/27701 Compliant Audit Logger
 * 
 * Append-only audit log for all security-relevant events.
 * Compatible with sql.js (no better-sqlite3 dependency)
 */

const crypto = require('crypto');

let dbModule = null;

function initAuditLog(module) {
  dbModule = module;
  // audit_log 表已在 db.js 中创建
}

// 查询辅助函数
function queryAll(sql, params = []) {
  if (!dbModule || !dbModule.db) return [];
  const stmt = dbModule.db.prepare(sql);
  stmt.bind(params);
  const results = [];
  while (stmt.step()) {
    results.push(stmt.getAsObject());
  }
  stmt.free();
  return results;
}

function runSql(sql, params = []) {
  if (!dbModule || !dbModule.db) return { changes: 0 };
  dbModule.db.run(sql, params);
  // 保存数据库
  if (dbModule.saveDatabase) {
    dbModule.saveDatabase();
  }
  return { changes: dbModule.db.getRowsModified() };
}

/**
 * Write an audit event (append-only)
 */
function write(event) {
  if (!dbModule || !dbModule.db) {
    console.warn('[AuditLog] Not initialized — call initAuditLog(db) first');
    return null;
  }

  const record = {
    id: `audit_${Date.now()}_${crypto.randomBytes(4).toString('hex')}`,
    timestamp: new Date().toISOString(),
    event: event.event || 'UNKNOWN',
    userId: event.userId || null,
    role: event.role || null,
    ip: event.ip || null,
    country: event.country || null,
    resource: event.resource || null,
    action: event.action || 'unknown',
    details: event.details ? JSON.stringify(event.details) : null,
    result: event.result || 'success',
    sessionId: event.sessionId || null,
  };

  try {
    runSql(
      `INSERT INTO audit_log (id, timestamp, event, user_id, role, ip, country, resource, action, details, result, session_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
      [record.id, record.timestamp, record.event, record.userId, record.role,
       record.ip, record.country, record.resource, record.action,
       record.details, record.result, record.sessionId]
    );
    return record;
  } catch (err) {
    console.error('[AuditLog] Write failed:', err.message);
    return null;
  }
}

/**
 * Query audit logs with filters (for compliance reporting & GDPR)
 */
function query({ startDate, endDate, event, userId, result, limit = 100 }) {
  if (!dbModule || !dbModule.db) return [];
  
  let sql = 'SELECT * FROM audit_log WHERE 1=1';
  const params = [];

  if (startDate) {
    sql += ' AND timestamp >= ?';
    params.push(startDate);
  }
  if (endDate) {
    sql += ' AND timestamp <= ?';
    params.push(endDate);
  }
  if (event) {
    sql += ' AND event = ?';
    params.push(event);
  }
  if (userId) {
    sql += ' AND user_id = ?';
    params.push(userId);
  }
  if (result) {
    sql += ' AND result = ?';
    params.push(result);
  }

  sql += ' ORDER BY timestamp DESC LIMIT ?';
  params.push(limit);

  return queryAll(sql, params);
}

/**
 * Get audit statistics
 */
function getStats() {
  if (!dbModule || !dbModule.db) return { total: 0 };
  
  const total = queryOne('SELECT COUNT(*) as count FROM audit_log')?.count || 0;
  const denied = queryOne("SELECT COUNT(*) as count FROM audit_log WHERE result = 'denied'")?.count || 0;
  const errors = queryOne("SELECT COUNT(*) as count FROM audit_log WHERE result = 'error'")?.count || 0;
  const geoBlocks = queryOne("SELECT COUNT(*) as count FROM audit_log WHERE event = 'GEO_BLOCK'")?.count || 0;
  
  // Last 24h
  const yesterday = new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString();
  const last24h = queryOne('SELECT COUNT(*) as count FROM audit_log WHERE timestamp >= ?', [yesterday])?.count || 0;

  return { total, denied, errors, geoBlocks, last24h };
}

function queryOne(sql, params = []) {
  const results = queryAll(sql, params);
  return results[0] || null;
}

/**
 * Express middleware: extract client IP for audit logging
 */
function auditMiddleware(req, res, next) {
  req.clientIP = req.headers['x-forwarded-for']?.split(',')[0]?.trim() ||
                 req.headers['x-real-ip'] ||
                 req.socket?.remoteAddress ||
                 'unknown';
  req.geoCountry = req.headers['cf-ipcountry'] || 'unknown';
  next();
}

module.exports = {
  initAuditLog,
  write,
  query,
  getStats,
  auditMiddleware,
};
