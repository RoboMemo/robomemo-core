/**
 * Temporal Consistency Validator
 * 
 * Cross-category conflict detection for structured VQA results.
 * Ensures timestamps/claims across temporal, spatial, mechanics, trajectory
 * categories are mutually consistent.
 */

/**
 * Parse timestamp string → seconds (supports "MM:SS", "SS.s", "Xs", "X.Ys")
 */
function parseTimestamp(ts) {
  if (typeof ts !== 'string') return NaN;
  ts = ts.trim();
  // "MM:SS" or "M:SS"
  const mmss = ts.match(/^(\d+):(\d+(?:\.\d+)?)$/);
  if (mmss) return parseFloat(mmss[1]) * 60 + parseFloat(mmss[2]);
  // "12.5s" or "12s"
  const secs = ts.match(/^([\d.]+)\s*s$/i);
  if (secs) return parseFloat(secs[1]);
  // plain number
  const n = parseFloat(ts);
  return isNaN(n) ? NaN : n;
}

/**
 * Extract all timestamped claims from a structured VQA analysis result.
 * Returns array of { category, claim, timestamp (seconds), raw }
 */
function extractTimestampedClaims(analysis) {
  const claims = [];

  // Temporal: action_sequence
  if (analysis.temporal?.action_sequence) {
    for (const action of analysis.temporal.action_sequence) {
      const t = parseTimestamp(action.timestamp);
      if (!isNaN(t)) {
        claims.push({
          category: 'temporal',
          claim: action.action || action.description,
          timestamp: t,
          rawTimestamp: action.timestamp,
          frameRange: action.frame_range,
        });
      }
    }
  }

  // Spatial: key_relationships
  if (analysis.spatial?.key_relationships) {
    for (const rel of analysis.spatial.key_relationships) {
      const t = parseTimestamp(rel.timestamp);
      if (!isNaN(t)) {
        claims.push({
          category: 'spatial',
          claim: rel.relationship || rel.details,
          timestamp: t,
          rawTimestamp: rel.timestamp,
        });
      }
    }
  }

  // Mechanics: contacts
  if (analysis.mechanics?.contacts) {
    for (const contact of analysis.mechanics.contacts) {
      const t = parseTimestamp(contact.timestamp);
      if (!isNaN(t)) {
        claims.push({
          category: 'mechanics',
          claim: `${contact.contact_type} (${contact.force_level}) at ${contact.contact_points}`,
          timestamp: t,
          rawTimestamp: contact.timestamp,
        });
      }
    }
  }

  // Trajectory: motion_segments
  if (analysis.trajectory?.motion_segments) {
    for (const seg of analysis.trajectory.motion_segments) {
      // time_range format: "0:00-0:05" or "0s-5s"
      if (seg.time_range) {
        const parts = seg.time_range.split(/[-–]/);
        if (parts.length === 2) {
          const tStart = parseTimestamp(parts[0]);
          const tEnd = parseTimestamp(parts[1]);
          if (!isNaN(tStart)) {
            claims.push({
              category: 'trajectory',
              claim: `${seg.segment}: ${seg.motion_type} ${seg.velocity}`,
              timestamp: tStart,
              timestampEnd: tEnd,
              rawTimestamp: seg.time_range,
            });
          }
        }
      }
    }
  }

  return claims;
}

/**
 * Detect temporal conflicts between claims across categories.
 */
function detectConflicts(claims) {
  const conflicts = [];
  const OVERLAP_THRESHOLD = 0.5; // seconds — claims within this range are "same time"

  for (let i = 0; i < claims.length; i++) {
    for (let j = i + 1; j < claims.length; j++) {
      const a = claims[i];
      const b = claims[j];

      // Only check cross-category conflicts
      if (a.category === b.category) continue;

      const timeDiff = Math.abs(a.timestamp - b.timestamp);

      // 1. Ordering conflict: temporal says A before B, but mechanics/trajectory timestamps disagree
      if (timeDiff < OVERLAP_THRESHOLD) {
        // Same timepoint — check for contradictory claims
        if (isContradictory(a.claim, b.claim)) {
          conflicts.push({
            category_a: a.category,
            category_b: b.category,
            claim_a: a.claim,
            claim_b: b.claim,
            timestamp_a: a.rawTimestamp,
            timestamp_b: b.rawTimestamp,
            description: `Contradictory claims at same timepoint (~${a.timestamp.toFixed(1)}s): "${a.claim}" vs "${b.claim}"`,
          });
        }
      }

      // 2. Trajectory segment overlap with mechanics contact that's outside the segment range
      if (a.category === 'trajectory' && b.category === 'mechanics' && a.timestampEnd) {
        if (b.timestamp < a.timestamp - OVERLAP_THRESHOLD || b.timestamp > a.timestampEnd + OVERLAP_THRESHOLD) {
          // Mechanics contact outside trajectory segment — potential issue if related
          if (claimsAreRelated(a.claim, b.claim)) {
            conflicts.push({
              category_a: a.category,
              category_b: b.category,
              claim_a: a.claim,
              claim_b: b.claim,
              timestamp_a: a.rawTimestamp,
              timestamp_b: b.rawTimestamp,
              description: `Related mechanics event at ${b.rawTimestamp} falls outside trajectory segment ${a.rawTimestamp}`,
            });
          }
        }
      }

      // 3. Frame range ordering: if temporal says action at frame [10,20] but spatial claims at earlier frame
      if (a.frameRange && b.category === 'spatial') {
        // Check if the spatial claim's timestamp maps to a frame outside the temporal action's range
        // (heuristic: if spatial happens much earlier/later than temporal action)
        if (timeDiff > 5 && claimsAreRelated(a.claim, b.claim)) {
          conflicts.push({
            category_a: a.category,
            category_b: b.category,
            claim_a: a.claim,
            claim_b: b.claim,
            timestamp_a: a.rawTimestamp,
            timestamp_b: b.rawTimestamp,
            description: `Related spatial observation at ${b.rawTimestamp} is ${timeDiff.toFixed(1)}s away from temporal action at ${a.rawTimestamp}`,
          });
        }
      }
    }
  }

  return conflicts;
}

/**
 * Simple heuristic for contradictory claims at the same timestamp.
 */
function isContradictory(claimA, claimB) {
  if (!claimA || !claimB) return false;
  const a = claimA.toLowerCase();
  const b = claimB.toLowerCase();

  // Opposite motion directions
  const directions = [
    ['left', 'right'], ['up', 'down'], ['forward', 'backward'],
    ['open', 'close'], ['grasp', 'release'], ['lift', 'lower'],
    ['approach', 'retreat'], ['push', 'pull'],
  ];

  for (const [d1, d2] of directions) {
    if ((a.includes(d1) && b.includes(d2)) || (a.includes(d2) && b.includes(d1))) {
      return true;
    }
  }

  // Contradictory force levels at same point
  if (a.includes('no contact') && (b.includes('light') || b.includes('medium') || b.includes('strong'))) return true;
  if (b.includes('no contact') && (a.includes('light') || a.includes('medium') || a.includes('strong'))) return true;

  return false;
}

/**
 * Heuristic: are two claims about the same object/action?
 */
function claimsAreRelated(claimA, claimB) {
  if (!claimA || !claimB) return false;
  const a = claimA.toLowerCase().split(/\s+/);
  const b = claimB.toLowerCase().split(/\s+/);
  const stopWords = new Set(['the', 'a', 'an', 'is', 'at', 'in', 'on', 'to', 'of', 'and', 'or', 'with']);
  const wordsA = a.filter(w => w.length > 2 && !stopWords.has(w));
  const wordsB = new Set(b.filter(w => w.length > 2 && !stopWords.has(w)));
  const overlap = wordsA.filter(w => wordsB.has(w));
  return overlap.length >= 2;
}

/**
 * Main entry: validate temporal consistency of a structured VQA analysis.
 * @param {object} analysis — StructuredVQAAnalysis object
 * @returns {{ consistent: boolean, conflicts: Array }}
 */
function validateTemporalConsistency(analysis) {
  if (!analysis) return { consistent: true, conflicts: [] };

  const claims = extractTimestampedClaims(analysis);

  // Also check: action_sequence should be in chronological order
  const temporalActions = claims.filter(c => c.category === 'temporal');
  for (let i = 1; i < temporalActions.length; i++) {
    if (temporalActions[i].timestamp < temporalActions[i - 1].timestamp) {
      const a = temporalActions[i - 1];
      const b = temporalActions[i];
      // This is directly added as a conflict
      claims.push(); // no-op, conflict captured below
      return buildResult(claims, [{
        category_a: 'temporal',
        category_b: 'temporal',
        claim_a: a.claim,
        claim_b: b.claim,
        timestamp_a: a.rawTimestamp,
        timestamp_b: b.rawTimestamp,
        description: `Action sequence out of order: "${a.claim}" at ${a.rawTimestamp} should be before "${b.claim}" at ${b.rawTimestamp}`,
      }, ...detectConflicts(claims)]);
    }
  }

  const crossConflicts = detectConflicts(claims);
  return buildResult(claims, crossConflicts);
}

function buildResult(claims, conflicts) {
  return {
    consistent: conflicts.length === 0,
    conflicts,
    stats: {
      totalClaims: claims.length,
      categoriesChecked: [...new Set(claims.map(c => c.category))],
    },
  };
}

module.exports = { validateTemporalConsistency, parseTimestamp, extractTimestampedClaims };
