import { useState, useEffect, useRef, useCallback } from 'react';
import {
  Wand2, Play, ChevronDown, ChevronRight, FileJson,
  Download, RefreshCw, AlertCircle, CheckCircle2, Clock,
  Loader2, GripHorizontal
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import api from '@/services/api';

// ── Types ───────────────────────────────────────────────────────────────────

interface OllamaModel {
  name: string;
  size: number;
  modified_at?: string;
}

interface PhaseMechanics {
  contact_type: string;
  force_level: string;
  contact_points: string;
  motion_direction: string;
}

interface Phase {
  phase_idx: number;
  phase_name: string;
  start_time: number;
  end_time: number;
  start_frame: number;
  end_frame: number;
  action_primitive: string;
  target_object: string;
  gripper_state: string;
  mechanics: PhaseMechanics;
  confidence: number;
  description?: string;
}

interface VideoInfo {
  duration: number;
  fps: number;
  resolution: number[];
  total_frames: number;
}

interface LabelResult {
  episode_id: string;
  video_path: string;
  video_info: VideoInfo;
  phases: Phase[];
  task_summary: string;
  success: boolean;
  model: string;
  labeled_at: string;
  error?: string;
}

interface AutoLabelJob {
  id: string;
  status: 'processing' | 'completed' | 'failed';
  videoPath: string;
  model: string;
  createdAt: string;
  completedAt?: string;
  progress: number;
  logs: string[];
  result?: LabelResult;
  error?: string;
}

interface Dataset {
  id: string;
  name: string;
}

interface Episode {
  id: string;
  name: string;
  videoPath?: string;
  h5_path?: string;   // actual video path stored in DB
  datasetId?: string;
}

// ── Helpers ─────────────────────────────────────────────────────────────────

const PRIMITIVE_COLORS: Record<string, string> = {
  approach: 'bg-blue-500',
  align: 'bg-indigo-500',
  grasp: 'bg-purple-500',
  lift: 'bg-violet-500',
  move: 'bg-sky-500',
  rotate_cw: 'bg-cyan-500',
  rotate_ccw: 'bg-teal-500',
  insert: 'bg-emerald-500',
  push: 'bg-green-500',
  pull: 'bg-lime-500',
  place: 'bg-amber-500',
  release: 'bg-orange-500',
  inspect: 'bg-yellow-500',
  wait: 'bg-gray-400',
  retract: 'bg-red-500',
};

function confidenceColor(c: number): string {
  if (c >= 0.8) return 'text-green-600';
  if (c >= 0.5) return 'text-yellow-600';
  return 'text-red-600';
}

function forceBadgeVariant(level: string): 'default' | 'secondary' | 'destructive' | 'outline' {
  if (level === 'strong') return 'destructive';
  if (level === 'medium') return 'default';
  if (level === 'light') return 'secondary';
  return 'outline';
}

// ── Component ───────────────────────────────────────────────────────────────

export default function AutoLabelPipeline() {
  // Models
  const [models, setModels] = useState<OllamaModel[]>([]);
  const [selectedModel, setSelectedModel] = useState('scomper/minicpm-v2.5:latest');
  const [ollamaUrl, setOllamaUrl] = useState('http://localhost:11434');
  const [numFrames, setNumFrames] = useState(16);

  // Video selection
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [selectedDataset, setSelectedDataset] = useState('');
  const [episodes, setEpisodes] = useState<Episode[]>([]);
  const [selectedVideoPath, setSelectedVideoPath] = useState('');
  const [customVideoPath, setCustomVideoPath] = useState('');

  // Job state
  const [currentJob, setCurrentJob] = useState<AutoLabelJob | null>(null);
  const [isRunning, setIsRunning] = useState(false);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Results
  const [results, setResults] = useState<LabelResult[]>([]);
  const [expandedPhases, setExpandedPhases] = useState<Record<string, boolean>>({});
  const [showRawJson, setShowRawJson] = useState<Record<string, boolean>>({});

  // Export
  const [exportDir, setExportDir] = useState('data/lerobot_export');
  const [robotType, setRobotType] = useState('single_arm');
  const [isExporting, setIsExporting] = useState(false);
  const [exportResult, setExportResult] = useState<string>('');

  // ── Load initial data ───────────────────────────────────────────────

  useEffect(() => {
    loadModels();
    loadDatasets();
    loadResults();
  }, []);

  const loadModels = async () => {
    try {
      const data = await api.getLocalVLMModels();
      if (data.available && data.models.length > 0) {
        setModels(data.models);
      }
    } catch {
      // Ollama not running
    }
  };

  const loadDatasets = async () => {
    try {
      const ds = await api.getDatasets();
      setDatasets(ds);
    } catch {
      // ignore
    }
  };

  const loadResults = async () => {
    try {
      const r: LabelResult[] = await api.fetch('/auto-label/results');
      setResults(r);
    } catch {
      // ignore
    }
  };

  const loadEpisodes = useCallback(async (datasetId: string) => {
    if (!datasetId) {
      setEpisodes([]);
      return;
    }
    try {
      const eps = await api.getDatasetEpisodes(datasetId);
      setEpisodes(eps);
    } catch {
      setEpisodes([]);
    }
  }, []);

  useEffect(() => {
    if (selectedDataset) {
      loadEpisodes(selectedDataset);
    }
  }, [selectedDataset, loadEpisodes]);

  // ── Run pipeline ──────────────────────────────────────────────────────

  const runPipeline = async () => {
    const videoPath = selectedVideoPath || customVideoPath;
    if (!videoPath) return;

    setIsRunning(true);
    setCurrentJob(null);

    try {
      const { jobId }: { jobId: string } = await api.fetch('/auto-label/run', {
        method: 'POST',
        body: JSON.stringify({
          videoPath,
          model: selectedModel,
          ollamaUrl,
          numFrames,
        }),
      });

      // Start polling
      pollRef.current = setInterval(async () => {
        try {
          const job: AutoLabelJob = await api.fetch(`/auto-label/jobs/${jobId}`);
          setCurrentJob(job);

          if (job.status === 'completed' || job.status === 'failed') {
            if (pollRef.current) clearInterval(pollRef.current);
            setIsRunning(false);
            if (job.status === 'completed' && job.result) {
              setResults(prev => [...prev, job.result!]);
            }
          }
        } catch {
          // keep polling
        }
      }, 2000);
    } catch (e: unknown) {
      setIsRunning(false);
      const msg = e instanceof Error ? e.message : 'Unknown error';
      setCurrentJob({
        id: 'error',
        status: 'failed',
        videoPath: videoPath,
        model: selectedModel,
        createdAt: new Date().toISOString(),
        progress: 0,
        logs: [],
        error: msg,
      });
    }
  };

  // Cleanup polling on unmount
  useEffect(() => {
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, []);

  // ── Export to LeRobot ─────────────────────────────────────────────────

  const exportToLeRobot = async () => {
    setIsExporting(true);
    setExportResult('');
    try {
      const result = await api.fetch<{ success: boolean; output_dir: string; total_episodes: number }>('/auto-label/export-lerobot', {
        method: 'POST',
        body: JSON.stringify({
          outputDir: exportDir,
          robotType,
        }),
      });
      setExportResult(`Exported ${result.total_episodes} episodes to ${result.output_dir}`);
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : 'Export failed';
      setExportResult(`Error: ${msg}`);
    } finally {
      setIsExporting(false);
    }
  };

  // ── Toggle helpers ────────────────────────────────────────────────────

  const togglePhase = (key: string) => {
    setExpandedPhases(prev => ({ ...prev, [key]: !prev[key] }));
  };

  const toggleRawJson = (key: string) => {
    setShowRawJson(prev => ({ ...prev, [key]: !prev[key] }));
  };

  // ── Render ────────────────────────────────────────────────────────────

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h2 className="text-2xl font-bold flex items-center gap-2">
          <Wand2 className="w-6 h-6" />
          Auto Label Pipeline
        </h2>
        <p className="text-muted-foreground mt-1">
          Turn robot manipulation videos into structured action labels using local Ollama VLM.
          Pipeline: Phase Segmentation → Action Primitives → Mechanics Estimation → JSONL
        </p>
      </div>

      {/* Configuration */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg">Configuration</CardTitle>
          <CardDescription>Select model, video, and pipeline settings</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* Model & Ollama URL */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="space-y-2">
              <Label>VLM Model</Label>
              <Select value={selectedModel} onValueChange={setSelectedModel}>
                <SelectTrigger>
                  <SelectValue placeholder="Select model" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="scomper/minicpm-v2.5:latest">scomper/minicpm-v2.5:latest</SelectItem>
                  {models.map(m => (
                    <SelectItem key={m.name} value={m.name}>{m.name}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-2">
              <Label>Ollama URL</Label>
              <Input
                value={ollamaUrl}
                onChange={e => setOllamaUrl(e.target.value)}
                placeholder="http://localhost:11434"
              />
            </div>
            <div className="space-y-2">
              <Label>Frames for Segmentation</Label>
              <Input
                type="number"
                value={numFrames}
                onChange={e => setNumFrames(parseInt(e.target.value) || 16)}
                min={4}
                max={32}
              />
            </div>
          </div>

          {/* Video Selection */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="space-y-2">
              <Label>Dataset</Label>
              <Select value={selectedDataset} onValueChange={(v) => { setSelectedDataset(v); setSelectedVideoPath(''); }}>
                <SelectTrigger>
                  <SelectValue placeholder="Select dataset" />
                </SelectTrigger>
                <SelectContent>
                  {datasets.map(d => (
                    <SelectItem key={d.id} value={d.id}>{d.name}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-2">
              <Label>Episode Video</Label>
              <Select value={selectedVideoPath} onValueChange={setSelectedVideoPath}>
                <SelectTrigger>
                  <SelectValue placeholder="Select episode" />
                </SelectTrigger>
                <SelectContent>
                  {episodes.filter((e: Episode) => e.videoPath || e.h5_path).map((e: Episode) => {
                    const vpath = e.videoPath || e.h5_path || '';
                    // h5_path is relative like "data/datasets/...", make it absolute
                    const absPath = vpath.startsWith('/') ? vpath
                      : `/Volumes/MOVESPEED/Project/RoboMemo/Platform/backend/${vpath}`;
                    return (
                      <SelectItem key={e.id} value={absPath}>{e.name}</SelectItem>
                    );
                  })}
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-2">
              <Label>Or paste video path</Label>
              <Input
                value={customVideoPath}
                onChange={e => { setCustomVideoPath(e.target.value); setSelectedVideoPath(''); }}
                placeholder="data/datasets/lerobot_aloha_cups/videos/..."
              />
            </div>
          </div>

          {/* Run Button */}
          <div className="flex items-center gap-4">
            <Button
              onClick={runPipeline}
              disabled={isRunning || (!selectedVideoPath && !customVideoPath)}
              className="min-w-[160px]"
            >
              {isRunning ? (
                <><Loader2 className="w-4 h-4 mr-2 animate-spin" /> Running...</>
              ) : (
                <><Play className="w-4 h-4 mr-2" /> Run Pipeline</>
              )}
            </Button>
            <Button variant="outline" onClick={loadModels} size="sm">
              <RefreshCw className="w-4 h-4 mr-1" /> Refresh Models
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Job Progress */}
      {currentJob && (
        <Card>
          <CardHeader>
            <CardTitle className="text-lg flex items-center gap-2">
              {currentJob.status === 'processing' && <Loader2 className="w-5 h-5 animate-spin text-blue-500" />}
              {currentJob.status === 'completed' && <CheckCircle2 className="w-5 h-5 text-green-500" />}
              {currentJob.status === 'failed' && <AlertCircle className="w-5 h-5 text-red-500" />}
              Pipeline Job: {currentJob.id}
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="flex items-center gap-4">
              <Progress value={currentJob.progress} className="flex-1" />
              <span className="text-sm text-muted-foreground w-12 text-right">{currentJob.progress}%</span>
            </div>
            {currentJob.error && (
              <p className="text-sm text-red-600">{currentJob.error}</p>
            )}
            {currentJob.logs.length > 0 && (
              <div className="bg-muted rounded p-3 max-h-40 overflow-y-auto font-mono text-xs space-y-0.5">
                {currentJob.logs.slice(-20).map((line, i) => (
                  <div key={i}>{line}</div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {/* Results */}
      {results.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="text-lg">Labeled Episodes ({results.length})</CardTitle>
            <CardDescription>Click on a result to expand phase details</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {results.map((result, ri) => {
              const rKey = `result_${ri}`;
              return (
                <Card key={rKey} className="border">
                  <CardHeader className="pb-2">
                    <div className="flex items-center justify-between">
                      <div>
                        <CardTitle className="text-base">{result.episode_id}</CardTitle>
                        <CardDescription>{result.task_summary}</CardDescription>
                      </div>
                      <div className="flex items-center gap-2">
                        <Badge variant={result.success ? 'default' : 'destructive'}>
                          {result.success ? 'Success' : 'Failed'}
                        </Badge>
                        <Badge variant="outline">
                          <Clock className="w-3 h-3 mr-1" />
                          {result.video_info.duration}s
                        </Badge>
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => toggleRawJson(rKey)}
                        >
                          <FileJson className="w-4 h-4" />
                        </Button>
                      </div>
                    </div>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    {/* Phase Timeline */}
                    <div className="space-y-1">
                      <Label className="text-xs text-muted-foreground">Phase Timeline</Label>
                      <div className="flex h-8 rounded overflow-hidden border">
                        {result.phases.map((phase, pi) => {
                          const totalDur = result.video_info.duration || 1;
                          const phaseDur = phase.end_time - phase.start_time;
                          const widthPct = Math.max((phaseDur / totalDur) * 100, 2);
                          const color = PRIMITIVE_COLORS[phase.action_primitive] || 'bg-gray-400';
                          return (
                            <div
                              key={pi}
                              className={`${color} flex items-center justify-center text-white text-[10px] font-medium cursor-pointer hover:opacity-80 transition-opacity`}
                              style={{ width: `${widthPct}%` }}
                              title={`${phase.phase_name}: ${phase.action_primitive} (${phase.start_time}s - ${phase.end_time}s)`}
                              onClick={() => togglePhase(`${rKey}_${pi}`)}
                            >
                              {widthPct > 8 ? phase.action_primitive : ''}
                            </div>
                          );
                        })}
                      </div>
                    </div>

                    {/* Phase Details */}
                    {result.phases.map((phase, pi) => {
                      const pKey = `${rKey}_${pi}`;
                      const isExpanded = expandedPhases[pKey];
                      return (
                        <div key={pi} className="border rounded">
                          <button
                            className="w-full flex items-center justify-between px-3 py-2 hover:bg-accent transition-colors text-left"
                            onClick={() => togglePhase(pKey)}
                          >
                            <div className="flex items-center gap-2">
                              {isExpanded ? <ChevronDown className="w-4 h-4" /> : <ChevronRight className="w-4 h-4" />}
                              <GripHorizontal className="w-3 h-3 text-muted-foreground" />
                              <span className="text-sm font-medium">{phase.phase_name}</span>
                              <Badge variant="secondary" className="text-xs">
                                {phase.action_primitive}
                              </Badge>
                              <span className="text-xs text-muted-foreground">
                                {phase.start_time}s - {phase.end_time}s
                              </span>
                            </div>
                            <span className={`text-sm font-mono ${confidenceColor(phase.confidence)}`}>
                              {(phase.confidence * 100).toFixed(0)}%
                            </span>
                          </button>
                          {isExpanded && (
                            <div className="px-3 pb-3 pt-1 space-y-2 border-t">
                              <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">
                                <div>
                                  <span className="text-muted-foreground text-xs">Target Object</span>
                                  <p className="font-medium">{phase.target_object}</p>
                                </div>
                                <div>
                                  <span className="text-muted-foreground text-xs">Gripper State</span>
                                  <p className="font-medium">{phase.gripper_state}</p>
                                </div>
                                <div>
                                  <span className="text-muted-foreground text-xs">Frames</span>
                                  <p className="font-medium">{phase.start_frame} - {phase.end_frame}</p>
                                </div>
                                <div>
                                  <span className="text-muted-foreground text-xs">Confidence</span>
                                  <p className={`font-medium ${confidenceColor(phase.confidence)}`}>
                                    {(phase.confidence * 100).toFixed(1)}%
                                  </p>
                                </div>
                              </div>
                              {phase.mechanics && (
                                <div className="flex flex-wrap gap-2">
                                  <Badge variant={forceBadgeVariant(phase.mechanics.force_level)}>
                                    Force: {phase.mechanics.force_level}
                                  </Badge>
                                  <Badge variant="outline">Contact: {phase.mechanics.contact_type}</Badge>
                                  <Badge variant="outline">Motion: {phase.mechanics.motion_direction}</Badge>
                                  {phase.mechanics.contact_points && (
                                    <span className="text-xs text-muted-foreground">
                                      {phase.mechanics.contact_points}
                                    </span>
                                  )}
                                </div>
                              )}
                              {phase.description && (
                                <p className="text-xs text-muted-foreground italic">{phase.description}</p>
                              )}
                            </div>
                          )}
                        </div>
                      );
                    })}

                    {/* Raw JSON */}
                    {showRawJson[rKey] && (
                      <div className="bg-muted rounded p-3 max-h-60 overflow-auto">
                        <pre className="text-xs font-mono whitespace-pre-wrap">
                          {JSON.stringify(result, null, 2)}
                        </pre>
                      </div>
                    )}
                  </CardContent>
                </Card>
              );
            })}
          </CardContent>
        </Card>
      )}

      {/* Export to LeRobot */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg flex items-center gap-2">
            <Download className="w-5 h-5" />
            Export to LeRobot Format
          </CardTitle>
          <CardDescription>
            Convert labeled episodes to LeRobot-compatible dataset structure
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="space-y-2">
              <Label>Output Directory</Label>
              <Input
                value={exportDir}
                onChange={e => setExportDir(e.target.value)}
                placeholder="data/lerobot_export"
              />
            </div>
            <div className="space-y-2">
              <Label>Robot Type</Label>
              <Select value={robotType} onValueChange={setRobotType}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="single_arm">Single Arm</SelectItem>
                  <SelectItem value="dual_arm">Dual Arm</SelectItem>
                  <SelectItem value="mobile_manipulator">Mobile Manipulator</SelectItem>
                  <SelectItem value="humanoid">Humanoid</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-2">
              <Label>&nbsp;</Label>
              <Button
                onClick={exportToLeRobot}
                disabled={isExporting || results.length === 0}
                className="w-full"
              >
                {isExporting ? (
                  <><Loader2 className="w-4 h-4 mr-2 animate-spin" /> Exporting...</>
                ) : (
                  <><Download className="w-4 h-4 mr-2" /> Export {results.length} Episode(s)</>
                )}
              </Button>
            </div>
          </div>
          {exportResult && (
            <p className={`text-sm ${exportResult.startsWith('Error') ? 'text-red-600' : 'text-green-600'}`}>
              {exportResult}
            </p>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
