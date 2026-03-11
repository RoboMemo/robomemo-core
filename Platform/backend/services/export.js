/**
 * Batch Export Service
 * 
 * Export datasets/annotations in multiple formats:
 *   - CSV (tabular)
 *   - COCO JSON (object detection standard)
 *   - RoboMemo JSON (native structured VQA)
 *   - LeRobot HDF5 metadata JSON
 */

const fs = require('fs');
const path = require('path');

/**
 * Available export formats
 */
const EXPORT_FORMATS = [
  {
    id: 'csv',
    name: 'CSV',
    description: 'Comma-separated values — annotations in tabular format',
    extension: '.csv',
    mimeType: 'text/csv',
  },
  {
    id: 'coco_json',
    name: 'COCO JSON',
    description: 'COCO format — compatible with object detection frameworks',
    extension: '.json',
    mimeType: 'application/json',
  },
  {
    id: 'robomemo_json',
    name: 'RoboMemo JSON',
    description: 'Native structured VQA export with full grounding data',
    extension: '.json',
    mimeType: 'application/json',
  },
  {
    id: 'lerobot_meta',
    name: 'LeRobot Metadata',
    description: 'LeRobot-compatible metadata JSON for HDF5 datasets',
    extension: '.json',
    mimeType: 'application/json',
  },
];

/**
 * Export annotations to CSV format
 */
function exportToCSV(annotations, episodes, datasets) {
  const headers = [
    'annotation_id', 'episode_id', 'dataset_id', 'type', 'label',
    'confidence', 'annotator', 'verified', 'created_at', 'updated_at',
    'category', 'timestamp', 'description', 'grounding_frames', 'grounding_confidence',
  ];

  const rows = [headers.join(',')];

  for (const ann of annotations) {
    // Base annotation fields
    const base = [
      csvEscape(ann.id), csvEscape(ann.episodeId || ann.episode_id),
      csvEscape(ann.datasetId || ann.dataset_id), csvEscape(ann.type),
      csvEscape(ann.label), ann.confidence, csvEscape(ann.annotator),
      ann.verified ? 1 : 0, csvEscape(ann.createdAt || ann.created_at),
      csvEscape(ann.updatedAt || ann.updated_at),
    ];

    // If structured VQA, expand each category
    const analysis = ann.analysis || ann.data?.analysis;
    if (analysis && ann.type === 'structured_vqa') {
      for (const category of ['temporal', 'spatial', 'attribute', 'mechanics', 'reasoning', 'summary', 'trajectory']) {
        const catData = analysis[category];
        if (!catData) continue;

        const items = extractCategoryItems(category, catData);
        for (const item of items) {
          rows.push([
            ...base,
            csvEscape(category),
            csvEscape(item.timestamp || ''),
            csvEscape(item.description || ''),
            csvEscape(item.groundingFrames || ''),
            item.groundingConfidence || '',
          ].join(','));
        }
      }
    } else {
      rows.push([...base, '', '', '', '', ''].join(','));
    }
  }

  return rows.join('\n');
}

/**
 * Export to COCO JSON format
 */
function exportToCOCO(annotations, episodes, datasets) {
  const coco = {
    info: {
      description: 'RoboMemo Robot Manipulation Dataset',
      version: '1.0',
      year: new Date().getFullYear(),
      contributor: 'RoboMemo Platform',
      date_created: new Date().toISOString(),
    },
    licenses: [{ id: 1, name: 'MIT', url: 'https://opensource.org/licenses/MIT' }],
    images: [],
    annotations: [],
    categories: [
      { id: 1, name: 'temporal', supercategory: 'vqa' },
      { id: 2, name: 'spatial', supercategory: 'vqa' },
      { id: 3, name: 'attribute', supercategory: 'vqa' },
      { id: 4, name: 'mechanics', supercategory: 'vqa' },
      { id: 5, name: 'reasoning', supercategory: 'vqa' },
      { id: 6, name: 'summary', supercategory: 'vqa' },
      { id: 7, name: 'trajectory', supercategory: 'vqa' },
    ],
  };

  const categoryMap = {};
  coco.categories.forEach(c => { categoryMap[c.name] = c.id; });

  let imageId = 1;
  let annId = 1;

  for (const ann of annotations) {
    const analysis = ann.analysis || ann.data?.analysis;
    if (!analysis) continue;

    // Create "images" from key frames
    const keyFrames = analysis.visual_evidence?.key_frames || [];
    for (const kf of keyFrames) {
      coco.images.push({
        id: imageId,
        file_name: kf.thumbnail_url || `frame_${kf.frame_idx}.jpg`,
        width: 640,  // placeholder — actual resolution from video
        height: 480,
        frame_idx: kf.frame_idx,
        timestamp: kf.timestamp,
        video_path: analysis.metadata?.video_path,
      });

      // Create annotations for bboxes on this frame
      if (kf.bboxes) {
        for (const bbox of kf.bboxes) {
          coco.annotations.push({
            id: annId++,
            image_id: imageId,
            category_id: 1,
            bbox: [bbox.x * 640, bbox.y * 480, bbox.width * 640, bbox.height * 480],
            area: bbox.width * 640 * bbox.height * 480,
            iscrowd: 0,
            attributes: { label: bbox.label },
          });
        }
      }
      imageId++;
    }

    // Create VQA annotations with grounding bboxes
    for (const category of Object.keys(categoryMap)) {
      const catData = analysis[category];
      if (!catData) continue;

      const items = extractCategoryItems(category, catData);
      for (const item of items) {
        if (item.bboxes && item.bboxes.length > 0) {
          for (const bbox of item.bboxes) {
            coco.annotations.push({
              id: annId++,
              image_id: item.frameIndex || 1,
              category_id: categoryMap[category],
              bbox: [bbox.x * 640, bbox.y * 480, bbox.width * 640, bbox.height * 480],
              area: bbox.width * 640 * bbox.height * 480,
              iscrowd: 0,
              attributes: {
                description: item.description,
                timestamp: item.timestamp,
                label: bbox.label,
              },
            });
          }
        }
      }
    }
  }

  return JSON.stringify(coco, null, 2);
}

/**
 * Export native RoboMemo JSON (full structured VQA)
 */
function exportToRoboMemoJSON(annotations, episodes, datasets) {
  return JSON.stringify({
    version: '1.0',
    exported_at: new Date().toISOString(),
    platform: 'RoboMemo',
    datasets: datasets.map(d => ({
      id: d.id,
      name: d.name,
      description: d.description,
      format: d.format,
      robotType: d.robotType || d.robot_type,
    })),
    episodes: episodes.map(e => ({
      id: e.id,
      datasetId: e.datasetId || e.dataset_id,
      name: e.name,
      skill: e.skill,
      duration: e.duration,
      frameCount: e.frameCount || e.frame_count,
    })),
    annotations: annotations.map(a => ({
      id: a.id,
      type: a.type,
      episodeId: a.episodeId || a.episode_id,
      provider: a.provider,
      model: a.model,
      analysis: a.analysis || a.data?.analysis,
      createdAt: a.createdAt || a.created_at,
    })),
  }, null, 2);
}

/**
 * Export LeRobot-compatible metadata
 */
function exportToLeRobotMeta(annotations, episodes, datasets) {
  return JSON.stringify({
    codebase_version: '1.0',
    robot_type: datasets[0]?.robotType || datasets[0]?.robot_type || 'unknown',
    fps: episodes[0]?.fps || 30,
    total_episodes: episodes.length,
    total_frames: episodes.reduce((sum, e) => sum + (e.frameCount || e.frame_count || 0), 0),
    tasks: [...new Set(episodes.map(e => e.skill).filter(Boolean))],
    episodes: episodes.map(e => ({
      episode_id: e.id,
      dataset_id: e.datasetId || e.dataset_id,
      length: e.frameCount || e.frame_count || 0,
      task: e.skill,
    })),
    vqa_annotations: annotations
      .filter(a => a.type === 'structured_vqa')
      .map(a => ({
        annotation_id: a.id,
        video_path: a.analysis?.metadata?.video_path || a.data?.analysis?.metadata?.video_path,
        categories: Object.keys(a.analysis || a.data?.analysis || {}).filter(k =>
          ['temporal', 'spatial', 'attribute', 'mechanics', 'reasoning', 'summary', 'trajectory'].includes(k)
        ),
      })),
  }, null, 2);
}

// --- Helpers ---

function csvEscape(val) {
  if (val === null || val === undefined) return '';
  const str = String(val);
  if (str.includes(',') || str.includes('"') || str.includes('\n')) {
    return `"${str.replace(/"/g, '""')}"`;
  }
  return str;
}

function extractCategoryItems(category, data) {
  const items = [];

  switch (category) {
    case 'temporal':
      for (const action of (data.action_sequence || [])) {
        items.push({
          timestamp: action.timestamp,
          description: action.description || action.action,
          groundingFrames: action.grounding?.frame_indices?.join(';') || '',
          groundingConfidence: action.grounding?.confidence,
          bboxes: action.grounding?.bboxes,
          frameIndex: action.grounding?.frame_indices?.[0],
        });
      }
      break;

    case 'spatial':
      for (const rel of (data.key_relationships || [])) {
        items.push({
          timestamp: rel.timestamp,
          description: rel.relationship || rel.details,
          groundingFrames: rel.grounding?.frame_indices?.join(';') || '',
          groundingConfidence: rel.grounding?.confidence,
          bboxes: rel.grounding?.bboxes,
          frameIndex: rel.grounding?.frame_indices?.[0],
        });
      }
      break;

    case 'attribute':
      for (const obj of (data.objects || [])) {
        items.push({
          timestamp: '',
          description: `${obj.name}: ${obj.properties?.color} ${obj.properties?.material} ${obj.properties?.shape}`,
          groundingFrames: obj.grounding?.frame_indices?.join(';') || '',
          groundingConfidence: obj.grounding?.confidence,
          bboxes: obj.grounding?.bboxes,
          frameIndex: obj.grounding?.frame_indices?.[0],
        });
      }
      break;

    case 'mechanics':
      for (const contact of (data.contacts || [])) {
        items.push({
          timestamp: contact.timestamp,
          description: `${contact.contact_type} ${contact.force_level} at ${contact.contact_points}`,
          groundingFrames: contact.grounding?.frame_indices?.join(';') || '',
          groundingConfidence: contact.grounding?.confidence,
          bboxes: contact.grounding?.bboxes,
          frameIndex: contact.grounding?.frame_indices?.[0],
        });
      }
      break;

    case 'reasoning':
      for (const just of (data.action_justifications || [])) {
        items.push({
          timestamp: '',
          description: `${just.action}: ${just.reason}`,
          groundingFrames: just.grounding?.frame_indices?.join(';') || '',
          groundingConfidence: just.grounding?.confidence,
          bboxes: just.grounding?.bboxes,
          frameIndex: just.grounding?.frame_indices?.[0],
        });
      }
      break;

    case 'summary':
      items.push({
        timestamp: '',
        description: data.task_description || '',
        groundingFrames: data.grounding_start?.frame_indices?.join(';') || '',
        groundingConfidence: data.grounding_start?.confidence,
      });
      break;

    case 'trajectory':
      for (const seg of (data.motion_segments || [])) {
        items.push({
          timestamp: seg.time_range,
          description: `${seg.segment}: ${seg.motion_type} ${seg.velocity}`,
          groundingFrames: seg.grounding?.frame_indices?.join(';') || '',
          groundingConfidence: seg.grounding?.confidence,
          bboxes: seg.grounding?.bboxes,
          frameIndex: seg.grounding?.frame_indices?.[0],
        });
      }
      break;
  }

  return items;
}

module.exports = {
  EXPORT_FORMATS,
  exportToCSV,
  exportToCOCO,
  exportToRoboMemoJSON,
  exportToLeRobotMeta,
};
