/**
 * Core API endpoint tests.
 * Tests Python scripts are valid and DB models work.
 */
const { execSync } = require('child_process');
const path = require('path');
const fs = require('fs');

const BACKEND_DIR = path.join(__dirname, '..');
const PYTHON = fs.existsSync(path.join(BACKEND_DIR, 'venv', 'bin', 'python3'))
  ? path.join(BACKEND_DIR, 'venv', 'bin', 'python3')
  : 'python3';

describe('API core', () => {
  test('lerobot_exporter.py exists and is valid Python', () => {
    const exporterPath = path.join(BACKEND_DIR, 'lerobot_exporter.py');
    expect(fs.existsSync(exporterPath)).toBe(true);

    expect(() => {
      execSync(`${PYTHON} -c "import py_compile; py_compile.compile('${exporterPath}', doraise=True)"`, {
        encoding: 'utf-8',
        timeout: 10000,
      });
    }).not.toThrow();
  });

  test('vlm_video_analyzer.py exists and is valid Python', () => {
    const analyzerPath = path.join(BACKEND_DIR, 'vlm_video_analyzer.py');
    expect(fs.existsSync(analyzerPath)).toBe(true);

    expect(() => {
      execSync(`${PYTHON} -c "import py_compile; py_compile.compile('${analyzerPath}', doraise=True)"`, {
        encoding: 'utf-8',
        timeout: 10000,
      });
    }).not.toThrow();
  });

  test('db.js exports required models', () => {
    const db = require(path.join(BACKEND_DIR, 'db.js'));
    expect(db.Annotations).toBeDefined();
    expect(typeof db.Annotations.getVQAAnalyses).toBe('function');
    expect(typeof db.Annotations.getVQAById).toBe('function');
    expect(typeof db.Annotations.insert).toBe('function');
  });

  test('Annotations.getVQAAnalyses returns array', () => {
    const db = require(path.join(BACKEND_DIR, 'db.js'));
    const analyses = db.Annotations.getVQAAnalyses();
    expect(Array.isArray(analyses)).toBe(true);
  });

  test('Annotations.getVQAById returns null for nonexistent id', () => {
    const db = require(path.join(BACKEND_DIR, 'db.js'));
    const result = db.Annotations.getVQAById('nonexistent_id_12345');
    expect(result).toBeFalsy();
  });
});
