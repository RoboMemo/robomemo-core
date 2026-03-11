/**
 * Rate Limiter Middleware
 * 
 * Token-bucket rate limiting per IP for VLM endpoints.
 * Prevents API abuse and controls cost.
 */

const buckets = new Map();

const DEFAULTS = {
  windowMs: 60 * 1000,   // 1 minute window
  maxRequests: 10,        // max 10 requests per window (VLM endpoints)
};

/**
 * Create a rate limiter middleware with configurable options.
 */
function rateLimiter(opts = {}) {
  const windowMs = opts.windowMs || DEFAULTS.windowMs;
  const maxRequests = opts.maxRequests || DEFAULTS.maxRequests;
  const keyFn = opts.keyFn || ((req) => req.headers['x-real-ip'] || req.ip || 'unknown');

  // Cleanup stale buckets every 5 minutes
  setInterval(() => {
    const now = Date.now();
    for (const [key, bucket] of buckets) {
      if (now - bucket.windowStart > windowMs * 2) {
        buckets.delete(key);
      }
    }
  }, 5 * 60 * 1000).unref();

  return (req, res, next) => {
    const key = keyFn(req);
    const now = Date.now();

    let bucket = buckets.get(key);
    if (!bucket || now - bucket.windowStart > windowMs) {
      bucket = { windowStart: now, count: 0 };
      buckets.set(key, bucket);
    }

    bucket.count++;

    // Set rate limit headers
    res.set('X-RateLimit-Limit', String(maxRequests));
    res.set('X-RateLimit-Remaining', String(Math.max(0, maxRequests - bucket.count)));
    res.set('X-RateLimit-Reset', String(Math.ceil((bucket.windowStart + windowMs) / 1000)));

    if (bucket.count > maxRequests) {
      return res.status(429).json({
        error: 'Too many requests',
        retryAfter: Math.ceil((bucket.windowStart + windowMs - now) / 1000),
      });
    }

    next();
  };
}

// Pre-configured limiters
const vlmLimiter = rateLimiter({ windowMs: 60000, maxRequests: 10 });
const authLimiter = rateLimiter({ windowMs: 60000, maxRequests: 20 });
const exportLimiter = rateLimiter({ windowMs: 300000, maxRequests: 5 });

module.exports = { rateLimiter, vlmLimiter, authLimiter, exportLimiter };
