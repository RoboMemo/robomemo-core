/**
 * Geo-Fencing Middleware
 * 
 * Blocks requests from China (CN) and Hong Kong (HK).
 * Works with:
 *   1. NGINX GeoIP headers (X-Geo-Country) in production
 *   2. Fallback IP-based check using geoip-lite in development
 */

const auditLog = require('../services/audit-log');

// Blocked country codes (ISO 3166-1 alpha-2)
const BLOCKED_COUNTRIES = new Set(['CN', 'HK']);

// Known China IP ranges (APNIC — sample prefixes for dev fallback)
// In production, use MaxMind GeoIP2 via NGINX
const CHINA_CIDR_PREFIXES = [
  '1.0.1.', '1.0.2.', '1.1.', '1.2.', '14.', '27.', '36.', '39.',
  '42.', '49.', '58.', '59.', '60.', '61.', '101.', '103.', '106.',
  '110.', '111.', '112.', '113.', '114.', '115.', '116.', '117.',
  '118.', '119.', '120.', '121.', '122.', '123.', '124.', '125.',
  '139.', '140.', '144.', '150.', '153.', '157.', '159.', '163.',
  '171.', '175.', '180.', '182.', '183.', '202.', '203.', '210.',
  '211.', '218.', '219.', '220.', '221.', '222.', '223.',
];

/**
 * Get client IP from request
 */
function getClientIP(req) {
  return req.headers['x-real-ip']
    || (req.headers['x-forwarded-for'] || '').split(',')[0].trim()
    || req.ip
    || req.connection?.remoteAddress
    || '';
}

/**
 * Dev fallback: rough IP prefix check (NOT production-ready)
 */
function isLikelyChinaIP(ip) {
  if (!ip || ip === '127.0.0.1' || ip === '::1' || ip.startsWith('192.168.') || ip.startsWith('10.')) {
    return false; // local dev — allow
  }
  return CHINA_CIDR_PREFIXES.some(prefix => ip.startsWith(prefix));
}

/**
 * Main geo-fence middleware
 */
function geoFenceMiddleware(req, res, next) {
  // Skip for health checks
  if (req.path === '/api/health') return next();

  const country = (req.headers['x-geo-country'] || '').toUpperCase();
  const clientIP = getClientIP(req);

  let blocked = false;
  let source = '';

  // Check 1: NGINX GeoIP header
  if (country && BLOCKED_COUNTRIES.has(country)) {
    blocked = true;
    source = 'geoip_header';
  }

  // Check 2: Dev fallback IP prefix check (only if no GeoIP header)
  if (!country && isLikelyChinaIP(clientIP)) {
    blocked = true;
    source = 'ip_prefix_check';
  }

  if (blocked) {
    auditLog.write({
      event: 'GEO_FENCE_BLOCK',
      ip: clientIP,
      country: country || 'CN_SUSPECTED',
      resource: req.path,
      action: 'read',
      details: { source, method: req.method, userAgent: req.headers['user-agent'] },
      result: 'denied',
    });

    return res.status(403).json({
      error: 'Access denied',
      reason: 'GEOGRAPHIC_RESTRICTION',
      message: 'This service is not available in your region.',
    });
  }

  // Attach geo info to request for downstream use
  req.geoCountry = country || null;
  req.clientIP = clientIP;

  next();
}

/**
 * VLM API egress region validator.
 * Ensures outbound VLM API calls don't route through China.
 */
const ALLOWED_EGRESS_REGIONS = ['ap-southeast-1', 'me-south-1', 'eu-west-1', 'us-east-1'];

function validateEgressRegion(region) {
  if (!region) return true; // no region specified = allow (cloud provider routing)
  return ALLOWED_EGRESS_REGIONS.includes(region);
}

module.exports = { geoFenceMiddleware, getClientIP, validateEgressRegion, BLOCKED_COUNTRIES };
