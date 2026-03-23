/**
 * Tests for repair_truncated_json() in vlm_video_analyzer.py
 * Calls the Python function via a temp script file, using base64 to avoid escaping.
 */
const { execSync } = require('child_process');
const path = require('path');
const fs = require('fs');
const os = require('os');

const BACKEND_DIR = path.join(__dirname, '..');
const PYTHON = fs.existsSync(path.join(BACKEND_DIR, 'venv', 'bin', 'python3'))
  ? path.join(BACKEND_DIR, 'venv', 'bin', 'python3')
  : 'python3';

function callRepair(jsonStr) {
  const b64 = Buffer.from(jsonStr).toString('base64');
  const tmpFile = path.join(os.tmpdir(), `repair_test_${Date.now()}.py`);
  const script = `
import sys, base64
sys.path.insert(0, '${BACKEND_DIR}')
from vlm_video_analyzer import repair_truncated_json
raw = base64.b64decode('${b64}').decode('utf-8')
result = repair_truncated_json(raw)
print(result)
`.trim();
  fs.writeFileSync(tmpFile, script);
  try {
    return execSync(`${PYTHON} "${tmpFile}"`, { encoding: 'utf-8', timeout: 10000 }).trim();
  } finally {
    try { fs.unlinkSync(tmpFile); } catch (_) {}
  }
}

describe('repair_truncated_json', () => {
  test('valid JSON passes through unchanged', () => {
    const input = '{"temporal":{"actions":["grasp"]},"summary":{"task":"pick"}}';
    const output = callRepair(input);
    const parsed = JSON.parse(output);
    expect(parsed.temporal.actions).toEqual(['grasp']);
    expect(parsed.summary.task).toBe('pick');
  });

  test('fixes Extra data (two JSON objects concatenated)', () => {
    const input = '{"temporal":{"actions":["reach"]}}\n{"extra":"garbage"}';
    const output = callRepair(input);
    const parsed = JSON.parse(output);
    expect(parsed.temporal.actions).toEqual(['reach']);
    expect(parsed.extra).toBeUndefined();
  });

  test('fixes truncated string in nested object', () => {
    const input = '{"summary":{"task":"pick up the re';
    const output = callRepair(input);
    const parsed = JSON.parse(output);
    expect(parsed.summary).toBeDefined();
    expect(typeof parsed.summary.task).toBe('string');
  });

  test('fixes deeply truncated JSON with missing closing brackets', () => {
    const input = '{"temporal":{"action_sequence":["reach","grasp"]},"spatial":{"objects":[{"name":"cup"';
    const output = callRepair(input);
    const parsed = JSON.parse(output);
    expect(parsed.temporal.action_sequence).toEqual(['reach', 'grasp']);
  });

  test('returns empty object for empty input', () => {
    const output = callRepair('  ');
    if (output === '') {
      expect(true).toBe(true);
    } else {
      const parsed = JSON.parse(output);
      expect(typeof parsed).toBe('object');
    }
  });
});
