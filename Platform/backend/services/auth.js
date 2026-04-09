/**
 * JWT Authentication & Session Management
 * 
 * - Simple password hashing (crypto-based)
 * - JWT token issue/verify
 * - User CRUD on SQLite users table
 * 
 * Compatible with sql.js (no better-sqlite3 dependency)
 */

const crypto = require('crypto');
const jwt = require('jsonwebtoken');

const JWT_SECRET = () => process.env.JWT_SECRET || 'robomemo-dev-secret-change-in-production';
const JWT_EXPIRES_IN = '24h';

let db = null;
let dbModule = null;

// 简单的密码哈希（不需要 bcrypt）
function hashPasswordSync(password) {
  const salt = crypto.randomBytes(16).toString('hex');
  const hash = crypto.pbkdf2Sync(password, salt, 100000, 64, 'sha512').toString('hex');
  return `${salt}:${hash}`;
}

function verifyPasswordSync(password, stored) {
  const [salt, hash] = stored.split(':');
  const verifyHash = crypto.pbkdf2Sync(password, salt, 100000, 64, 'sha512').toString('hex');
  return hash === verifyHash;
}

function initAuth(module) {
  dbModule = module;
  
  // 用户表已在 db.js 中创建
  // Seed a default platform admin if no users exist
  const users = queryAll('SELECT COUNT(*) as n FROM users');
  if (users && users[0] && users[0].n === 0) {
    const now = new Date().toISOString();
    const hashedPw = hashPasswordSync('admin123');
    runSql(`
      INSERT INTO users (id, email, name, password, role, region, created_at, updated_at)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    `, ['user_admin', 'admin@robomemo.io', 'Platform Admin', hashedPw, 'platform_admin', 'SG', now, now]);
    console.log('[Auth] Default admin created: admin@robomemo.io / admin123');
  }
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

function queryOne(sql, params = []) {
  const results = queryAll(sql, params);
  return results[0] || null;
}

function runSql(sql, params = []) {
  if (!dbModule || !dbModule.db) return { changes: 0 };
  dbModule.db.run(sql, params);
  return { changes: dbModule.db.getRowsModified() };
}

function signToken(payload) {
  return jwt.sign(payload, JWT_SECRET(), { expiresIn: JWT_EXPIRES_IN });
}

function verifyToken(token) {
  return jwt.verify(token, JWT_SECRET());
}

const Users = {
  getAll: () => queryAll('SELECT id, email, name, role, region, active, created_at, updated_at FROM users'),
  
  getById: (id) => queryOne('SELECT * FROM users WHERE id = ?', [id]),
  
  getByEmail: (email) => queryOne('SELECT * FROM users WHERE email = ?', [email]),
  
  create: ({ email, password, name, role = 'annotator', region = 'SG' }) => {
    const id = `user_${Date.now()}`;
    const now = new Date().toISOString();
    const hashedPw = hashPasswordSync(password);
    runSql(
      'INSERT INTO users (id, email, name, password, role, region, active, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?)',
      [id, email, name, hashedPw, role, region, now, now]
    );
    return Users.getById(id);
  },
  
  update: (id, data) => {
    const fields = [];
    const values = [];
    const allowed = ['name', 'role', 'region', 'active'];
    for (const key of allowed) {
      if (data[key] !== undefined) {
        fields.push(`${key} = ?`);
        values.push(data[key]);
      }
    }
    if (fields.length === 0) return Users.getById(id);
    fields.push('updated_at = ?');
    values.push(new Date().toISOString());
    values.push(id);
    runSql(`UPDATE users SET ${fields.join(', ')} WHERE id = ?`, values);
    return Users.getById(id);
  },
  
  delete: (id) => runSql('DELETE FROM users WHERE id = ?', [id]),
  
  getAllData: (userId) => {
    const user = Users.getById(userId);
    const annotations = queryAll('SELECT * FROM annotations WHERE annotator = ?', [userId]);
    const auditLogs = queryAll('SELECT * FROM audit_log WHERE user_id = ?', [userId]);
    return { user, annotations, auditLogs };
  },
  
  eraseAllData: (userId) => {
    const deleted = {
      annotations: runSql('DELETE FROM annotations WHERE annotator = ?', [userId]).changes,
      auditLogs: runSql("UPDATE audit_log SET user_id = '[ERASED]', details = NULL WHERE user_id = ?", [userId]).changes,
      user: runSql('DELETE FROM users WHERE id = ?', [userId]).changes,
    };
    return deleted;
  },
};

function authMiddleware(req, res, next) {
  const PUBLIC_PATHS = ['/api/auth/login', '/api/auth/register', '/api/health', '/uploads/'];
  if (PUBLIC_PATHS.some(p => req.path.startsWith(p))) return next();

  const authHeader = req.headers.authorization;
  if (!authHeader || !authHeader.startsWith('Bearer ')) {
    return res.status(401).json({ error: 'Authentication required', code: 'AUTH_REQUIRED' });
  }

  try {
    const token = authHeader.slice(7);
    const payload = verifyToken(token);
    req.user = payload;
    next();
  } catch (err) {
    if (err.name === 'TokenExpiredError') {
      return res.status(401).json({ error: 'Token expired', code: 'TOKEN_EXPIRED' });
    }
    return res.status(401).json({ error: 'Invalid token', code: 'INVALID_TOKEN' });
  }
}

function requireRole(...roles) {
  return (req, res, next) => {
    if (!req.user) {
      return res.status(401).json({ error: 'Authentication required' });
    }
    if (!roles.includes(req.user.role)) {
      return res.status(403).json({
        error: 'Insufficient permissions',
        required: roles,
        current: req.user.role,
      });
    }
    next();
  };
}

module.exports = {
  initAuth,
  hashPasswordSync,
  verifyPasswordSync,
  signToken,
  verifyToken,
  Users,
  authMiddleware,
  requireRole,
};
