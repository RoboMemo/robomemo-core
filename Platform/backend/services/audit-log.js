/**
 * ISO 27001/27701 Compliant Audit Logger
 * 
 * Append-only audit log for all security-relevant events.
 * Stored in SQLite audit_log table (append-only: no UPDATE/DELETE exposed).
 * In production, migrate to PostgreSQL with partitioning.
 */

const crypto = require('crypto');

let db = null;

function initAuditLog(database) {
  db = database;
  db.exec(`
    CREATE TABLE IF NOT EXISTS audit_log (
      id          TEXT PRIMARY KEY,
      timestamp   TEXT NOT NULL,
      event       TEXT NOT NULL,
      user_id     TEXT,
      role        TEXT,
      ip          TEXT,
      country     TEXT,
      resource    TEXT,
      action      TEXT NOT NULL,
      details     TEXT,
      result      TEXT NOT NULL CHECK (result IN ('success', 'denied', 'error')),
      session_id  TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_audit_ts    ON audit_log(timestamp);
    CREATE INDEX IF NOT EXISTS idx_audit_event ON audit_log(event);
    CREATE INDEX IF NOT EXISTS idx_audit_user  ON audit_log(user_id);
  `);
}

const insertStmt = () => db.prepare(`
  INSERT INTO audit_log (id, timestamp, event, user_id, role, ip, country,
    resource, action, details, result, session_id)
  VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
`);

/**
 * Write an audit event (append-only)
 */
function write(event) {
  if (!db) {
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
    insertStmt().run(
      record.id, record.timestamp, record.event, record.userId, record.role,
      record.ip, record.country, record.resource, record.action,
      record.details, record.result, record.sessionId
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
function query(filters = {}) {
  if (!db) return [];

  let sql = 'SELECT * FROM audit_log WHERE 1=1';
  const params = [];

  if (filters.startDate) {
    sql += ' AND timestamp >= ?';
    params.push(filters.startDate);
  }
  if (filters.endDate) {
    sql += ' AND timestamp <= ?';
    params.push(filters.endDate);
  }
  if (filters.event) {
    sql += ' AND event = ?';
    params.push(filters.event);
  }
  if (filters.userId) {
    sql += ' AND user_id = ?';
    params.push(filters.userId);
  }
  if (filters.result) {
    sql += ' AND result = ?';
    params.push(filters.result);
  }
  if (filters.resource) {
    sql += ' AND resource = ?';
    params.push(filters.resource);
  }

  sql += ' ORDER BY timestamp DESC';

  if (filters.limit) {
    sql += ' LIMIT ?';
    params.push(filters.limit);
  }

  return db.prepare(sql).all(...params);
}

/**
 * Get audit stats for dashboard
 */
function getStats() {
  if (!db) return {};
  return {
    total: db.prepare('SELECT COUNT(*) as n FROM audit_log').get().n,
    denied: db.prepare("SELECT COUNT(*) as n FROM audit_log WHERE result='denied'").get().n,
    errors: db.prepare("SELECT COUNT(*) as n FROM audit_log WHERE result='error'").get().n,
    geoBlocks: db.prepare("SELECT COUNT(*) as n FROM audit_log WHERE event='GEO_FENCE_BLOCK'").get().n,
    last24h: db.prepare("SELECT COUNT(*) as n FROM audit_log WHERE timestamp > datetime('now', '-1 day')").get().n,
  };
}

/**
 * Express middleware: auto-log every request
 */
function auditMiddleware(req, res, next) {
  const start = Date.now();

  res.on('finish', () => {
    // Skip health checks and static assets from logging
    if (req.path === '/api/health' || req.path.startsWith('/uploads/frames')) return;

    const duration = Date.now() - start;
    const action = methodToAction(req.method);

    write({
      event: `HTTP_${req.method}`,
      userId: req.user?.userId || null,
      role: req.user?.role || null,
      ip: req.headers['x-real-ip'] || req.headers['x-forwarded-for'] || req.ip,
      country: req.headers['x-geo-country'] || null,
      resource: req.path,
      action,
      details: {
        method: req.method,
        path: req.path,
        statusCode: res.statusCode,
        duration,
        userAgent: req.headers['user-agent'],
      },
      result: res.statusCode < 400 ? 'success' : (res.statusCode < 500 ? 'denied' : 'error'),
    });
  });

  next();
}

function methodToAction(method) {
  switch (method) {
    case 'GET': return 'read';
    case 'POST': return 'create';
    case 'PUT': case 'PATCH': return 'update';
    case 'DELETE': return 'delete';
    default: return 'unknown';
  }
}

module.exports = { initAuditLog, write, query, getStats, auditMiddleware };
