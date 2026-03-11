/**
 * JWT Authentication & Session Management
 * 
 * - bcrypt password hashing
 * - JWT token issue/verify
 * - User CRUD on SQLite users table
 */

const crypto = require('crypto');
const jwt = require('jsonwebtoken');

const JWT_SECRET = () => process.env.JWT_SECRET || 'robomemo-dev-secret-change-in-production';
const JWT_EXPIRES_IN = '24h';
const SALT_ROUNDS = 12;

let db = null;

function initAuth(database) {
  db = database;
  db.exec(`
    CREATE TABLE IF NOT EXISTS users (
      id          TEXT PRIMARY KEY,
      email       TEXT UNIQUE NOT NULL,
      name        TEXT NOT NULL,
      password    TEXT NOT NULL,
      role        TEXT NOT NULL DEFAULT 'annotator'
                  CHECK (role IN ('annotator','reviewer','data_admin','platform_admin','auditor')),
      region      TEXT DEFAULT 'SG',
      mfa_enabled INTEGER DEFAULT 0,
      mfa_secret  TEXT,
      active      INTEGER DEFAULT 1,
      created_at  TEXT NOT NULL,
      updated_at  TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_user_email ON users(email);
    CREATE INDEX IF NOT EXISTS idx_user_role  ON users(role);
  `);

  // Seed a default platform admin if no users exist
  const count = db.prepare('SELECT COUNT(*) as n FROM users').get().n;
  if (count === 0) {
    const now = new Date().toISOString();
    const hashedPw = hashPasswordSync('admin123');
    db.prepare(`
      INSERT INTO users (id, email, name, password, role, region, created_at, updated_at)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    `).run('user_admin', 'admin@robomemo.io', 'Platform Admin', hashedPw, 'platform_admin', 'SG', now, now);
    console.log('[Auth] Default admin created: admin@robomemo.io / admin123');
  }
}

// --- Password hashing (PBKDF2 — no native bcrypt in Node without C addon) ---

function hashPasswordSync(password) {
  const salt = crypto.randomBytes(16).toString('hex');
  const hash = crypto.pbkdf2Sync(password, salt, 100000, 64, 'sha512').toString('hex');
  return `${salt}:${hash}`;
}

function verifyPasswordSync(password, stored) {
  const [salt, hash] = stored.split(':');
  const verify = crypto.pbkdf2Sync(password, salt, 100000, 64, 'sha512').toString('hex');
  return crypto.timingSafeEqual(Buffer.from(hash, 'hex'), Buffer.from(verify, 'hex'));
}

// --- JWT ---

function signToken(user) {
  const payload = {
    userId: user.id,
    email: user.email,
    name: user.name,
    role: user.role,
    region: user.region,
    mfaVerified: !!user.mfa_enabled,
    permissions: rolePermissions(user.role),
  };
  return jwt.sign(payload, JWT_SECRET(), { expiresIn: JWT_EXPIRES_IN });
}

function verifyToken(token) {
  return jwt.verify(token, JWT_SECRET());
}

function rolePermissions(role) {
  const perms = {
    annotator:      ['datasets:read', 'episodes:read', 'annotations:create', 'annotations:read'],
    reviewer:       ['datasets:read', 'episodes:read', 'annotations:*', 'review:*'],
    data_admin:     ['datasets:*', 'episodes:*', 'annotations:*', 'export:*'],
    platform_admin: ['*'],
    auditor:        ['audit:read', 'datasets:read', 'annotations:read'],
  };
  return perms[role] || [];
}

// --- User CRUD ---

const Users = {
  getById: (id) => {
    const row = db.prepare('SELECT * FROM users WHERE id = ?').get(id);
    if (row) delete row.password;
    return row;
  },

  getByEmail: (email) => {
    return db.prepare('SELECT * FROM users WHERE email = ?').get(email);
  },

  getAll: () => {
    return db.prepare('SELECT id, email, name, role, region, active, created_at, updated_at FROM users ORDER BY created_at DESC').all();
  },

  create: (data) => {
    const id = `user_${Date.now()}_${crypto.randomBytes(3).toString('hex')}`;
    const now = new Date().toISOString();
    const hashedPw = hashPasswordSync(data.password);
    db.prepare(`
      INSERT INTO users (id, email, name, password, role, region, created_at, updated_at)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    `).run(id, data.email, data.name, hashedPw, data.role || 'annotator', data.region || 'SG', now, now);
    return Users.getById(id);
  },

  update: (id, data) => {
    const now = new Date().toISOString();
    const sets = [];
    const params = [];
    if (data.name) { sets.push('name = ?'); params.push(data.name); }
    if (data.role) { sets.push('role = ?'); params.push(data.role); }
    if (data.region) { sets.push('region = ?'); params.push(data.region); }
    if (data.active !== undefined) { sets.push('active = ?'); params.push(data.active ? 1 : 0); }
    if (data.password) { sets.push('password = ?'); params.push(hashPasswordSync(data.password)); }
    sets.push('updated_at = ?');
    params.push(now);
    params.push(id);
    db.prepare(`UPDATE users SET ${sets.join(', ')} WHERE id = ?`).run(...params);
    return Users.getById(id);
  },

  delete: (id) => {
    db.prepare('DELETE FROM users WHERE id = ?').run(id);
  },

  /** Get all data associated with a user (for GDPR Art.15) */
  getAllData: (userId) => {
    const user = Users.getById(userId);
    const annotations = db.prepare("SELECT * FROM annotations WHERE annotator = ? OR data LIKE ?")
      .all(userId, `%${userId}%`);
    const auditLogs = db.prepare("SELECT * FROM audit_log WHERE user_id = ?").all(userId);
    return { user, annotations, auditLogs };
  },

  /** Erase all user data (for GDPR Art.17) */
  eraseAllData: (userId) => {
    const deleted = {
      annotations: db.prepare("DELETE FROM annotations WHERE annotator = ?").run(userId).changes,
      auditLogs: db.prepare("UPDATE audit_log SET user_id = '[ERASED]', details = NULL WHERE user_id = ?").run(userId).changes,
      user: db.prepare("DELETE FROM users WHERE id = ?").run(userId).changes,
    };
    return deleted;
  },
};

// --- Middleware ---

/**
 * Express middleware: require JWT authentication.
 * Sets req.user from token payload.
 */
function authMiddleware(req, res, next) {
  // Skip auth for login/register and health check
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

/**
 * RBAC middleware factory: require specific roles.
 * Usage: requireRole('platform_admin', 'data_admin')
 */
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
