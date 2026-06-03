import { useState, useEffect, useRef } from 'react';
import {
  ChevronRight, ChevronLeft, Database, Settings, FileJson,
  Play, Monitor, CheckCircle, StopCircle, RefreshCw
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Switch } from '@/components/ui/switch';
import { Slider } from '@/components/ui/slider';
import { Separator } from '@/components/ui/separator';
import DatasetEpisodePicker from '@/components/DatasetEpisodePicker';
import type { Dataset } from '@/types';

// ── Types ──────────────────────────────────────────────────────────────────────

interface ModelConfig {
  checkpointPath: string;
  loraRank: number;
  loraAlpha: number;
  loraDropout: number;
  learningRate: number;
  batchSize: number;
  maxSteps: number;
  memoryWindowSize: number;
  temporalContextLength: number;
  useEpisodicMemory: boolean;
}

interface ExportConfig {
  memoryContext: boolean;
  temporalFeatures: boolean;
  episodePrevRef: boolean;
}

interface LaunchConfig {
  target: 'local' | 'remote';
  sshHost: string;
  sshUser: string;
  sshKeyPath: string;
  experimentName: string;
}

interface TrainingStatus {
  status: 'idle' | 'running' | 'stopped' | 'completed';
  currentStep: number;
  totalSteps: number;
  loss: number[];
  checkpoints: { step: number; timestamp: string; size: string }[];
}

// ── Step indicator ─────────────────────────────────────────────────────────────

const STEPS = [
  { id: 1, label: '数据集', icon: Database },
  { id: 2, label: '模型配置', icon: Settings },
  { id: 3, label: '导出格式', icon: FileJson },
  { id: 4, label: '启动训练', icon: Play },
  { id: 5, label: '监控', icon: Monitor },
];

function StepIndicator({ current }: { current: number }) {
  return (
    <div className="flex items-center gap-2 mb-8">
      {STEPS.map((step, i) => {
        const Icon = step.icon;
        const done = step.id < current;
        const active = step.id === current;
        return (
          <div key={step.id} className="flex items-center gap-2">
            <div className="flex flex-col items-center gap-1">
              <div className={`w-10 h-10 rounded-full flex items-center justify-center text-sm font-semibold
                ${done ? 'bg-green-500 text-white' : active ? 'bg-primary text-primary-foreground' : 'bg-muted text-muted-foreground'}`}>
                {done ? <CheckCircle className="w-5 h-5" /> : <Icon className="w-5 h-5" />}
              </div>
              <span className={`text-xs ${active ? 'text-primary font-medium' : 'text-muted-foreground'}`}>
                {step.label}
              </span>
            </div>
            {i < STEPS.length - 1 && (
              <div className={`w-8 h-0.5 mb-5 ${done ? 'bg-green-500' : 'bg-muted'}`} />
            )}
          </div>
        );
      })}
    </div>
  );
}

// ── Main component ─────────────────────────────────────────────────────────────

export default function TrainingManager() {
  const [currentStep, setCurrentStep] = useState(1);
  const [selectedDataset, setSelectedDataset] = useState<Dataset | null>(null);

  const [modelConfig, setModelConfig] = useState<ModelConfig>({
    checkpointPath: 's3://openpi-assets/checkpoints/pi0_base',
    loraRank: 32,
    loraAlpha: 64,
    loraDropout: 0.05,
    learningRate: 5e-5,
    batchSize: 16,
    maxSteps: 80000,
    memoryWindowSize: 8,
    temporalContextLength: 4,
    useEpisodicMemory: false,
  });

  const [exportConfig, setExportConfig] = useState<ExportConfig>({
    memoryContext: true,
    temporalFeatures: true,
    episodePrevRef: false,
  });

  const [launchConfig, setLaunchConfig] = useState<LaunchConfig>({
    target: 'local',
    sshHost: '',
    sshUser: '',
    sshKeyPath: '~/.ssh/id_rsa',
    experimentName: `pi06mem_${Date.now()}`,
  });

  const [trainingStatus, setTrainingStatus] = useState<TrainingStatus>({
    status: 'idle',
    currentStep: 0,
    totalSteps: modelConfig.maxSteps,
    loss: [],
    checkpoints: [],
  });

  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const startPolling = () => {
    if (pollRef.current) return;
    pollRef.current = setInterval(async () => {
      try {
        const res = await fetch('/api/training/status');
        if (res.ok) {
          const data = await res.json();
          setTrainingStatus(data);
          if (data.status === 'completed' || data.status === 'stopped') {
            clearInterval(pollRef.current!);
            pollRef.current = null;
          }
        }
      } catch {}
    }, 3000);
  };

  useEffect(() => () => { if (pollRef.current) clearInterval(pollRef.current); }, []);

  // ── Step 1: Dataset ──────────────────────────────────────────────────────────

  const renderStep1 = () => (
    <Card>
      <CardHeader>
        <CardTitle>选择训练数据集</CardTitle>
      </CardHeader>
      <CardContent className="space-y-6">
        <DatasetEpisodePicker
          showEpisode={false}
          onDatasetChange={(_id, ds) => setSelectedDataset(ds)}
        />
        {selectedDataset && (
          <div className="p-4 bg-muted rounded-lg space-y-3">
            <h4 className="font-semibold">{selectedDataset.name}</h4>
            <div className="grid grid-cols-3 gap-4 text-sm">
              <div>
                <p className="text-muted-foreground">Episodes</p>
                <p className="font-bold text-lg">{selectedDataset.episodeCount}</p>
              </div>
              <div>
                <p className="text-muted-foreground">Frames</p>
                <p className="font-bold text-lg">{selectedDataset.frameCount}</p>
              </div>
              <div>
                <p className="text-muted-foreground">格式</p>
                <Badge variant="secondary">{selectedDataset.format}</Badge>
              </div>
            </div>
            {selectedDataset.episodeCount > 0
              ? <Badge className="bg-green-100 text-green-800">✓ 满足训练要求</Badge>
              : <Badge variant="destructive">✗ 数据集为空</Badge>
            }
          </div>
        )}
      </CardContent>
    </Card>
  );

  // ── Step 2: Model Config ─────────────────────────────────────────────────────

  const renderStep2 = () => (
    <div className="space-y-4">
      <Card>
        <CardHeader><CardTitle>基础训练参数</CardTitle></CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <Label>预训练检查点路径</Label>
            <Input
              value={modelConfig.checkpointPath}
              onChange={e => setModelConfig(c => ({ ...c, checkpointPath: e.target.value }))}
            />
          </div>
          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label>LoRA Rank（默认 32）</Label>
              <Input
                type="number"
                value={modelConfig.loraRank}
                onChange={e => setModelConfig(c => ({ ...c, loraRank: Number(e.target.value) }))}
              />
            </div>
            <div className="space-y-2">
              <Label>LoRA Alpha（默认 64）</Label>
              <Input
                type="number"
                value={modelConfig.loraAlpha}
                onChange={e => setModelConfig(c => ({ ...c, loraAlpha: Number(e.target.value) }))}
              />
            </div>
            <div className="space-y-2">
              <Label>LoRA Dropout（默认 0.05）</Label>
              <Input
                type="number"
                step="0.01"
                value={modelConfig.loraDropout}
                onChange={e => setModelConfig(c => ({ ...c, loraDropout: Number(e.target.value) }))}
              />
            </div>
            <div className="space-y-2">
              <Label>Learning Rate（默认 5e-5）</Label>
              <Input
                type="number"
                step="0.00001"
                value={modelConfig.learningRate}
                onChange={e => setModelConfig(c => ({ ...c, learningRate: Number(e.target.value) }))}
              />
            </div>
            <div className="space-y-2">
              <Label>Batch Size（默认 16）</Label>
              <Input
                type="number"
                value={modelConfig.batchSize}
                onChange={e => setModelConfig(c => ({ ...c, batchSize: Number(e.target.value) }))}
              />
            </div>
            <div className="space-y-2">
              <Label>Max Steps（默认 80000）</Label>
              <Input
                type="number"
                value={modelConfig.maxSteps}
                onChange={e => setModelConfig(c => ({ ...c, maxSteps: Number(e.target.value) }))}
              />
            </div>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader><CardTitle>pi-0.6-mem 扩展参数</CardTitle></CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <Label>memory_window_size — 时序记忆窗口（默认 8）</Label>
            <div className="flex items-center gap-4">
              <Slider
                min={1} max={32} step={1}
                value={[modelConfig.memoryWindowSize]}
                onValueChange={([v]) => setModelConfig(c => ({ ...c, memoryWindowSize: v }))}
                className="flex-1"
              />
              <span className="w-8 text-sm font-mono">{modelConfig.memoryWindowSize}</span>
            </div>
          </div>
          <div className="space-y-2">
            <Label>temporal_context_length — 过去帧数（默认 4）</Label>
            <div className="flex items-center gap-4">
              <Slider
                min={1} max={16} step={1}
                value={[modelConfig.temporalContextLength]}
                onValueChange={([v]) => setModelConfig(c => ({ ...c, temporalContextLength: v }))}
                className="flex-1"
              />
              <span className="w-8 text-sm font-mono">{modelConfig.temporalContextLength}</span>
            </div>
          </div>
          <div className="flex items-center justify-between">
            <div>
              <Label>use_episodic_memory</Label>
              <p className="text-xs text-muted-foreground">跨 episode 记忆（实验性）</p>
            </div>
            <Switch
              checked={modelConfig.useEpisodicMemory}
              onCheckedChange={v => setModelConfig(c => ({ ...c, useEpisodicMemory: v }))}
            />
          </div>
        </CardContent>
      </Card>
    </div>
  );

  // ── Step 3: Export Format ────────────────────────────────────────────────────

  const renderStep3 = () => (
    <Card>
      <CardHeader>
        <CardTitle>数据导出格式配置</CardTitle>
        <p className="text-sm text-muted-foreground">基于 LeRobot V2.1，添加 pi-0.6-mem 扩展字段</p>
      </CardHeader>
      <CardContent className="space-y-6">
        <div className="space-y-4">
          <div className="flex items-center justify-between p-3 border rounded-lg">
            <div>
              <p className="font-medium">memory_context</p>
              <p className="text-xs text-muted-foreground">前 N 步的动作/观测 embedding</p>
            </div>
            <Switch
              checked={exportConfig.memoryContext}
              onCheckedChange={v => setExportConfig(c => ({ ...c, memoryContext: v }))}
            />
          </div>
          <div className="flex items-center justify-between p-3 border rounded-lg">
            <div>
              <p className="font-medium">temporal_features</p>
              <p className="text-xs text-muted-foreground">光流 / 帧差分特征</p>
            </div>
            <Switch
              checked={exportConfig.temporalFeatures}
              onCheckedChange={v => setExportConfig(c => ({ ...c, temporalFeatures: v }))}
            />
          </div>
          <div className="flex items-center justify-between p-3 border rounded-lg">
            <div>
              <p className="font-medium">episode_id_prev</p>
              <p className="text-xs text-muted-foreground">前一个 episode 的引用（episodic memory）</p>
            </div>
            <Switch
              checked={exportConfig.episodePrevRef}
              onCheckedChange={v => setExportConfig(c => ({ ...c, episodePrevRef: v }))}
            />
          </div>
        </div>

        <Separator />

        <Button
          className="w-full"
          onClick={async () => {
            try {
              const res = await fetch('/api/training/export-config', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ datasetId: selectedDataset?.id, modelConfig, exportConfig }),
              });
              const data = await res.json();
              alert('导出配置已生成: ' + JSON.stringify(data, null, 2));
            } catch (e) {
              alert('生成失败: ' + String(e));
            }
          }}
        >
          <FileJson className="w-4 h-4 mr-2" />
          生成导出配置
        </Button>
      </CardContent>
    </Card>
  );

  // ── Step 4: Launch ───────────────────────────────────────────────────────────

  const renderStep4 = () => (
    <Card>
      <CardHeader><CardTitle>启动训练</CardTitle></CardHeader>
      <CardContent className="space-y-6">
        <div className="space-y-2">
          <Label>训练目标机器</Label>
          <div className="flex gap-3">
            {(['local', 'remote'] as const).map(t => (
              <button
                key={t}
                onClick={() => setLaunchConfig(c => ({ ...c, target: t }))}
                className={`flex-1 py-3 rounded-lg border text-sm font-medium transition-colors
                  ${launchConfig.target === t ? 'bg-primary text-primary-foreground border-primary' : 'hover:bg-accent'}`}
              >
                {t === 'local' ? '本地 Mac Mini' : '远程 SSH'}
              </button>
            ))}
          </div>
        </div>

        {launchConfig.target === 'remote' && (
          <div className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label>SSH Host</Label>
                <Input
                  placeholder="192.168.1.100"
                  value={launchConfig.sshHost}
                  onChange={e => setLaunchConfig(c => ({ ...c, sshHost: e.target.value }))}
                />
              </div>
              <div className="space-y-2">
                <Label>SSH User</Label>
                <Input
                  placeholder="ubuntu"
                  value={launchConfig.sshUser}
                  onChange={e => setLaunchConfig(c => ({ ...c, sshUser: e.target.value }))}
                />
              </div>
            </div>
            <div className="space-y-2">
              <Label>SSH Key 路径</Label>
              <Input
                placeholder="~/.ssh/id_rsa"
                value={launchConfig.sshKeyPath}
                onChange={e => setLaunchConfig(c => ({ ...c, sshKeyPath: e.target.value }))}
              />
            </div>
          </div>
        )}

        <div className="space-y-2">
          <Label>实验名称</Label>
          <Input
            value={launchConfig.experimentName}
            onChange={e => setLaunchConfig(c => ({ ...c, experimentName: e.target.value }))}
          />
        </div>

        <Button
          className="w-full"
          disabled={!selectedDataset}
          onClick={async () => {
            try {
              const res = await fetch('/api/training/launch', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ datasetId: selectedDataset?.id, modelConfig, exportConfig, launchConfig }),
              });
              if (res.ok) {
                setCurrentStep(5);
                startPolling();
              } else {
                const err = await res.json();
                alert('启动失败: ' + err.error);
              }
            } catch (e) {
              alert('启动失败: ' + String(e));
            }
          }}
        >
          <Play className="w-4 h-4 mr-2" />
          启动训练
        </Button>
      </CardContent>
    </Card>
  );

  // ── Step 5: Monitor ──────────────────────────────────────────────────────────

  const renderStep5 = () => (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle>训练状态</CardTitle>
            <Badge variant={
              trainingStatus.status === 'running' ? 'default' :
              trainingStatus.status === 'completed' ? 'secondary' : 'outline'
            }>
              {trainingStatus.status === 'running' ? '运行中' :
               trainingStatus.status === 'completed' ? '已完成' :
               trainingStatus.status === 'stopped' ? '已停止' : '空闲'}
            </Badge>
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div>
              <p className="text-muted-foreground">当前 Step</p>
              <p className="text-2xl font-bold">{trainingStatus.currentStep.toLocaleString()}</p>
            </div>
            <div>
              <p className="text-muted-foreground">总 Steps</p>
              <p className="text-2xl font-bold">{trainingStatus.totalSteps.toLocaleString()}</p>
            </div>
          </div>
          {trainingStatus.totalSteps > 0 && (
            <div className="space-y-1">
              <div className="flex justify-between text-xs text-muted-foreground">
                <span>进度</span>
                <span>{((trainingStatus.currentStep / trainingStatus.totalSteps) * 100).toFixed(1)}%</span>
              </div>
              <div className="w-full bg-muted rounded-full h-2">
                <div
                  className="bg-primary h-2 rounded-full transition-all"
                  style={{ width: `${(trainingStatus.currentStep / trainingStatus.totalSteps) * 100}%` }}
                />
              </div>
            </div>
          )}
          {trainingStatus.loss.length > 0 && (
            <div>
              <p className="text-sm text-muted-foreground mb-2">
                最近 Loss: {trainingStatus.loss[trainingStatus.loss.length - 1]?.toFixed(4)}
              </p>
              <div className="flex items-end gap-0.5 h-12">
                {trainingStatus.loss.slice(-30).map((l, i) => (
                  <div
                    key={i}
                    className="flex-1 bg-primary/60 rounded-sm min-h-[2px]"
                    style={{ height: `${Math.min(100, (l / Math.max(...trainingStatus.loss)) * 100)}%` }}
                  />
                ))}
              </div>
            </div>
          )}
          <div className="flex gap-2">
            <Button
              variant="outline"
              className="flex-1"
              onClick={async () => {
                try {
                  const res = await fetch('/api/training/status');
                  if (res.ok) setTrainingStatus(await res.json());
                } catch {}
              }}
            >
              <RefreshCw className="w-4 h-4 mr-2" />
              刷新
            </Button>
            <Button
              variant="destructive"
              className="flex-1"
              disabled={trainingStatus.status !== 'running'}
              onClick={async () => {
                try {
                  await fetch('/api/training/cancel', { method: 'POST' });
                } catch {}
              }}
            >
              <StopCircle className="w-4 h-4 mr-2" />
              停止训练
            </Button>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader><CardTitle>Checkpoints</CardTitle></CardHeader>
        <CardContent>
          {trainingStatus.checkpoints.length === 0 ? (
            <p className="text-sm text-muted-foreground text-center py-4">暂无 checkpoint</p>
          ) : (
            <div className="space-y-2">
              {trainingStatus.checkpoints.map((ck, i) => (
                <div key={i} className="flex items-center justify-between p-3 bg-muted rounded-lg text-sm">
                  <div>
                    <span className="font-medium">Step {ck.step.toLocaleString()}</span>
                    <span className="text-muted-foreground ml-3">{ck.timestamp}</span>
                  </div>
                  <Badge variant="outline">{ck.size}</Badge>
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );

  // ── Navigation ───────────────────────────────────────────────────────────────

  const renderCurrentStep = () => {
    switch (currentStep) {
      case 1: return renderStep1();
      case 2: return renderStep2();
      case 3: return renderStep3();
      case 4: return renderStep4();
      case 5: return renderStep5();
      default: return null;
    }
  };

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold">训练管理</h2>
        <p className="text-muted-foreground">pi-0.6-mem 完整训练流程</p>
      </div>

      <StepIndicator current={currentStep} />

      {renderCurrentStep()}

      <div className="flex justify-between pt-4">
        <Button
          variant="outline"
          onClick={() => setCurrentStep(s => Math.max(1, s - 1))}
          disabled={currentStep === 1}
        >
          <ChevronLeft className="w-4 h-4 mr-2" />
          上一步
        </Button>
        {currentStep < 5 && (
          <Button
            onClick={() => setCurrentStep(s => Math.min(5, s + 1))}
            disabled={currentStep === 1 && !selectedDataset}
          >
            下一步
            <ChevronRight className="w-4 h-4 ml-2" />
          </Button>
        )}
      </div>
    </div>
  );
}
