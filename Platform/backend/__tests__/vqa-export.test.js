/**
 * Tests for VQA -> LeRobot export conversion logic.
 */

function convertVQAToEpisode(analysis, annotationId, videoPath, model) {
  const temporal = analysis.temporal || {};
  const spatial = analysis.spatial || {};
  const mechanics = analysis.mechanics || {};
  const summary = analysis.summary || {};
  const trajectory = analysis.trajectory || {};
  const metadata = analysis.metadata || {};

  const actionSequence = temporal.action_sequence || temporal.actions || [];
  const contacts = mechanics.contacts || mechanics.contact_events || [];
  const totalFrames = metadata.total_frames || metadata.frame_count || 0;
  const fps = metadata.fps || 30;
  const duration = metadata.duration || (totalFrames / fps);

  const phases = actionSequence.map((action, idx) => {
    const actionStr = typeof action === 'string' ? action : (action.action || action.description || 'unknown');
    const target = typeof action === 'object' ? (action.target || action.object || '') : '';
    const startFrame = typeof action === 'object' ? (action.start_frame || 0) : 0;
    const endFrame = typeof action === 'object' ? (action.end_frame || 0) : 0;
    const startTime = typeof action === 'object' ? (action.start_time || startFrame / fps) : 0;
    const endTime = typeof action === 'object' ? (action.end_time || endFrame / fps) : 0;

    const contact = contacts[idx] || {};
    return {
      phase_idx: idx,
      start_frame: startFrame,
      end_frame: endFrame,
      start_time: startTime,
      end_time: endTime,
      action_primitive: actionStr,
      target_object: target || (spatial.objects?.[0]?.name || 'unknown'),
      gripper_state: contact.gripper_state || 'open',
      phase_name: `phase_${idx}`,
      confidence: action.confidence || 0.8,
      mechanics: {
        contact_type: contact.contact_type || contact.type || 'none',
        force_level: contact.force_level || contact.force || 'none',
        contact_points: contact.contact_points || '',
        motion_direction: trajectory.motion_type || 'linear',
      },
    };
  });

  if (phases.length === 0) {
    phases.push({
      phase_idx: 0,
      start_frame: 0,
      end_frame: totalFrames,
      start_time: 0,
      end_time: duration,
      action_primitive: summary.task_description || 'manipulation',
      target_object: spatial.objects?.[0]?.name || 'unknown',
      gripper_state: 'open',
      phase_name: 'phase_0',
      confidence: 0.5,
      mechanics: { contact_type: 'none', force_level: 'none', contact_points: '', motion_direction: 'linear' },
    });
  }

  return {
    episode_id: annotationId,
    task_summary: summary.task_description || summary.description || 'Robot manipulation task',
    video_info: { total_frames: totalFrames, fps, duration },
    video_path: videoPath || '',
    phases,
    model: model || 'unknown',
    success: true,
  };
}

describe('VQA -> LeRobot conversion', () => {
  test('converts full 7-category analysis to episode with phases', () => {
    const analysis = {
      temporal: {
        action_sequence: [
          { action: 'reach', target: 'cup', start_frame: 0, end_frame: 30 },
          { action: 'grasp', target: 'cup', start_frame: 30, end_frame: 60 },
        ],
      },
      spatial: { objects: [{ name: 'cup' }] },
      mechanics: {
        contacts: [
          { contact_type: 'none', force_level: 'none' },
          { contact_type: 'wrap', force_level: 'medium', gripper_state: 'closing' },
        ],
      },
      summary: { task_description: 'Pick up the cup' },
      trajectory: { motion_type: 'linear' },
      metadata: { total_frames: 60, fps: 30, duration: 2.0 },
    };

    const episode = convertVQAToEpisode(analysis, 'ann_123', '/video.mp4', 'minicpm-v2.5');
    expect(episode.task_summary).toBe('Pick up the cup');
    expect(episode.phases).toHaveLength(2);
    expect(episode.phases[0].action_primitive).toBe('reach');
    expect(episode.phases[1].mechanics.contact_type).toBe('wrap');
    expect(episode.phases[1].gripper_state).toBe('closing');
    expect(episode.video_info.fps).toBe(30);
  });

  test('creates fallback phase when no actions found', () => {
    const analysis = {
      temporal: {},
      spatial: { objects: [{ name: 'bottle' }] },
      summary: { task_description: 'Open the bottle' },
      metadata: { total_frames: 90, fps: 30 },
    };

    const episode = convertVQAToEpisode(analysis, 'ann_456', '/vid.mp4', 'llama3.2');
    expect(episode.phases).toHaveLength(1);
    expect(episode.phases[0].action_primitive).toBe('Open the bottle');
    expect(episode.phases[0].target_object).toBe('bottle');
    expect(episode.phases[0].end_frame).toBe(90);
  });

  test('handles string-only action_sequence', () => {
    const analysis = {
      temporal: { action_sequence: ['reach', 'grasp', 'lift'] },
      spatial: {},
      summary: {},
      metadata: {},
    };

    const episode = convertVQAToEpisode(analysis, 'ann_789', '', 'test');
    expect(episode.phases).toHaveLength(3);
    expect(episode.phases[0].action_primitive).toBe('reach');
    expect(episode.phases[2].action_primitive).toBe('lift');
  });

  test('handles completely empty analysis', () => {
    const episode = convertVQAToEpisode({}, 'ann_empty', '', 'test');
    expect(episode.phases).toHaveLength(1);
    expect(episode.task_summary).toBe('Robot manipulation task');
    expect(episode.phases[0].action_primitive).toBe('manipulation');
  });
});
