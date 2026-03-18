import { useState, useEffect, useRef } from 'react';
import {
  Clock,
  MapPin,
  Tag,
  Zap,
  Brain,
  FileText,
  Navigation,
  Upload,
  Loader2,
  CheckCircle2,
  XCircle,
  ChevronRight,
  Star,
  BarChart2,
  Download,
  Eye,
  RefreshCw,
  Settings2,
  AlertTriangle,
  Info,
  Layers,
  Cpu,
  Sparkles,
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Progress } from '@/components/ui/progress';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Separator } from '@/components/ui/separator';
import { api } from '@/services/api';
import type {
  VLMProvider,
  StructuredVQAAnalysis,
  VQAAnnotationRecord,
  VQAConfidenceScores,
  VisualGrounding,
  Dataset,
  Episode,
} from '@/types';
import { GroundingBadge, GroundingDetail, TemporalConsistencyAlert } from '@/components/GroundingOverlay';

// ─── VQA Category Configs ────────────────────────────────────────────────────
const VQA_CATEGORIES = [
  {
    id: 'temporal',
    label: 'Temporal',
    labelZh: '时间关系',
    icon: Clock,
    color: 'text-blue-500',
    bg: 'bg-blue-50 dark:bg-blue-950',
    border: 'border-blue-200 dark:border-blue-800',
    badge: 'bg-blue-100 text-blue-700 dark:bg-blue-900 dark:text-blue-300',
    description: '动作时序与前后关系',
  },
  {
    id: 'spatial',
    label: 'Spatial',
    labelZh: '空间关系',
    icon: MapPin,
    color: 'text-green-500',
    bg: 'bg-green-50 dark:bg-green-950',
    border: 'border-green-200 dark:border-green-800',
    badge: 'bg-green-100 text-green-700 dark:bg-green-900 dark:text-green-300',
    description: '手/抓手与物体的空间位置',
  },
  {
    id: 'attribute',
    label: 'Attribute',
    labelZh: '物体属性',
    icon: Tag,
    color: 'text-purple-500',
    bg: 'bg-purple-50 dark:bg-purple-950',
    border: 'border-purple-200 dark:border-purple-800',
    badge: 'bg-purple-100 text-purple-700 dark:bg-purple-900 dark:text-purple-300',
    description: '颜色、材质、形状等物体属性',
  },
  {
    id: 'mechanics',
    label: 'Mechanics',
    labelZh: '力学信息',
    icon: Zap,
    color: 'text-orange-500',
    bg: 'bg-orange-50 dark:bg-orange-950',
    border: 'border-orange-200 dark:border-orange-800',
    badge: 'bg-orange-100 text-orange-700 dark:bg-orange-900 dark:text-orange-300',
    description: '接触点、施力大小与方式',
  },
  {
    id: 'reasoning',
    label: 'Reasoning',
    labelZh: '推理过程',
    icon: Brain,
    color: 'text-pink-500',
    bg: 'bg-pink-50 dark:bg-pink-950',
    border: 'border-pink-200 dark:border-pink-800',
    badge: 'bg-pink-100 text-pink-700 dark:bg-pink-900 dark:text-pink-300',
    description: '为何这样操作的策略推理',
  },
  {
    id: 'summary',
    label: 'Summary',
    labelZh: '场景总结',
    icon: FileText,
    color: 'text-teal-500',
    bg: 'bg-teal-50 dark:bg-teal-950',
    border: 'border-teal-200 dark:border-teal-800',
    badge: 'bg-teal-100 text-teal-700 dark:bg-teal-900 dark:text-teal-300',
    description: '整体任务描述与完成状态',
  },
  {
    id: 'trajectory',
    label: 'Trajectory',
    labelZh: '轨迹描述',
    icon: Navigation,
    color: 'text-indigo-500',
    bg: 'bg-indigo-50 dark:bg-indigo-950',
    border: 'border-indigo-200 dark:border-indigo-800',
    badge: 'bg-indigo-100 text-indigo-700 dark:bg-indigo-900 dark:text-indigo-300',
    description: '末端执行器运动路径与速度',
  },
];

// ─── Confidence Gauge ────────────────────────────────────────────────────────
function ConfidenceGauge({ value, size = 'md' }: { value: number; size?: 'sm' | 'md' }) {
  const pct = Math.round(value * 100);
  const color =
    pct >= 80 ? 'text-green-500' : pct >= 60 ? 'text-yellow-500' : 'text-red-400';
  const barColor =
    pct >= 80 ? 'bg-green-500' : pct >= 60 ? 'bg-yellow-500' : 'bg-red-400';

  return (
    <div className={`flex items-center gap-2 ${size === 'sm' ? 'text-xs' : 'text-sm'}`}>
      <div className="flex-1 bg-muted rounded-full h-1.5 min-w-[60px]">
        <div className={`h-1.5 rounded-full transition-all ${barColor}`} style={{ width: `${pct}%` }} />
      </div>
      <span className={`font-mono font-semibold ${color} ${size === 'sm' ? 'text-xs' : ''}`}>
        {pct}%
      </span>
    </div>
  );
}

// ─── Force Level Badge ────────────────────────────────────────────────────────
function ForceBadge({ level }: { level: string }) {
  const map: Record<string, string> = {
    light: 'bg-green-100 text-green-700',
    medium: 'bg-yellow-100 text-yellow-700',
    strong: 'bg-red-100 text-red-700',
  };
  return (
    <span className={`px-2 py-0.5 rounded-full text-xs font-medium ${map[level] ?? 'bg-muted text-muted-foreground'}`}>
      {level}
    </span>
  );
}

// ─── Motion Type Badge ────────────────────────────────────────────────────────
function MotionBadge({ type }: { type: string }) {
  const map: Record<string, string> = {
    linear: 'bg-blue-100 text-blue-700',
    curved: 'bg-purple-100 text-purple-700',
    rotational: 'bg-orange-100 text-orange-700',
  };
  return (
    <span className={`px-2 py-0.5 rounded-full text-xs font-medium ${map[type] ?? 'bg-muted text-muted-foreground'}`}>
      {type}
    </span>
  );
}

// ─── Category Section Components ─────────────────────────────────────────────

function TemporalView({ data, frameImageUrls }: { data: StructuredVQAAnalysis['temporal']; frameImageUrls?: string[] }) {
  const [expandedGrounding, setExpandedGrounding] = useState<VisualGrounding | null>(null);
  return (
    <div className="space-y-4">
      <div>
        <h4 className="text-sm font-semibold mb-2 flex items-center gap-1.5">
          <Clock className="w-4 h-4 text-blue-500" />动作序列
        </h4>
        <div className="relative pl-4">
          <div className="absolute left-[7px] top-2 bottom-2 w-0.5 bg-blue-200 dark:bg-blue-800" />
          {data?.action_sequence?.map((item, i) => (
            <div key={i} className="relative mb-4 last:mb-0">
              <div className="absolute -left-4 top-1.5 w-3 h-3 rounded-full bg-blue-500 border-2 border-background shadow-sm" />
              <div className="ml-2 p-2.5 rounded-lg bg-blue-50 dark:bg-blue-950 border border-blue-100 dark:border-blue-900">
                <div className="flex items-center justify-between mb-1">
                  <span className="font-medium text-sm">{item.action}</span>
                  <div className="flex items-center gap-1.5">
                    {item.grounding && (
                      <GroundingBadge
                        grounding={item.grounding}
                        frameImageUrls={frameImageUrls}
                        colorClass="text-blue-500"
                        onClickView={setExpandedGrounding}
                      />
                    )}
                    <Badge variant="outline" className="text-xs font-mono">{item.timestamp}</Badge>
                  </div>
                </div>
                <p className="text-xs text-muted-foreground">{item.description}</p>
                {item.frame_range && (
                  <span className="text-xs text-blue-500 mt-1 block">
                    Frames {item.frame_range[0]}–{item.frame_range[1]}
                  </span>
                )}
                {expandedGrounding === item.grounding && item.grounding && (
                  <div className="mt-2">
                    <GroundingDetail grounding={item.grounding} frameImageUrls={frameImageUrls} />
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>
      </div>
      {data?.relationships?.length > 0 && (
        <div>
          <h4 className="text-sm font-semibold mb-2">时序关系</h4>
          <div className="space-y-1.5">
            {data.relationships.map((rel, i) => (
              <div key={i} className="flex items-center gap-2 text-sm p-2 rounded bg-muted/50">
                <ChevronRight className="w-3 h-3 text-blue-500 shrink-0" />
                <span>{rel}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function SpatialView({ data, frameImageUrls }: { data: StructuredVQAAnalysis['spatial']; frameImageUrls?: string[] }) {
  const [expandedGrounding, setExpandedGrounding] = useState<VisualGrounding | null>(null);
  return (
    <div className="space-y-4">
      <div>
        <h4 className="text-sm font-semibold mb-2 flex items-center gap-1.5">
          <MapPin className="w-4 h-4 text-green-500" />空间关系
        </h4>
        <div className="space-y-2">
          {data?.key_relationships?.map((rel, i) => (
            <div key={i} className="p-2.5 rounded-lg bg-green-50 dark:bg-green-950 border border-green-100 dark:border-green-900">
              <div className="flex items-center justify-between mb-1">
                <span className="font-medium text-sm">{rel.relationship}</span>
                <div className="flex items-center gap-1.5">
                  {rel.grounding && (
                    <GroundingBadge
                      grounding={rel.grounding}
                      frameImageUrls={frameImageUrls}
                      colorClass="text-green-500"
                      onClickView={setExpandedGrounding}
                    />
                  )}
                  <Badge variant="outline" className="text-xs font-mono">{rel.timestamp}</Badge>
                </div>
              </div>
              <p className="text-xs text-muted-foreground">{rel.details}</p>
              {expandedGrounding === rel.grounding && rel.grounding && (
                <div className="mt-2">
                  <GroundingDetail grounding={rel.grounding} frameImageUrls={frameImageUrls} />
                </div>
              )}
            </div>
          ))}
        </div>
      </div>
      {data?.trajectory_spatial && (
        <div>
          <h4 className="text-sm font-semibold mb-1.5">整体空间策略</h4>
          <p className="text-sm text-muted-foreground leading-relaxed p-2.5 rounded bg-muted/50">
            {data.trajectory_spatial}
          </p>
        </div>
      )}
    </div>
  );
}

function AttributeView({ data, frameImageUrls }: { data: StructuredVQAAnalysis['attribute']; frameImageUrls?: string[] }) {
  const [expandedGrounding, setExpandedGrounding] = useState<VisualGrounding | null>(null);
  return (
    <div className="space-y-3">
      {data?.objects?.map((obj, i) => (
        <div key={i} className="p-3 rounded-lg border border-purple-100 dark:border-purple-900 bg-purple-50 dark:bg-purple-950">
          <div className="flex items-center gap-2 mb-2">
            <Tag className="w-4 h-4 text-purple-500" />
            <span className="font-semibold text-sm">{obj.name}</span>
            {obj.grounding && (
              <GroundingBadge
                grounding={obj.grounding}
                frameImageUrls={frameImageUrls}
                colorClass="text-purple-500"
                onClickView={setExpandedGrounding}
              />
            )}
          </div>
          <div className="grid grid-cols-2 gap-1.5 text-xs mb-2">
            {Object.entries(obj.properties || {}).map(([key, val]) => (
              val && (
                <div key={key} className="flex gap-1">
                  <span className="text-muted-foreground capitalize">{key}:</span>
                  <span className="font-medium">{val as string}</span>
                </div>
              )
            ))}
          </div>
          {obj.state_changes?.length > 0 && (
            <div>
              <p className="text-xs text-muted-foreground font-medium mb-1">状态变化:</p>
              <div className="flex flex-wrap gap-1">
                {obj.state_changes.map((sc, j) => (
                  <Badge key={j} variant="secondary" className="text-xs">{sc}</Badge>
                ))}
              </div>
            </div>
          )}
          {expandedGrounding === obj.grounding && obj.grounding && (
            <div className="mt-2">
              <GroundingDetail grounding={obj.grounding} frameImageUrls={frameImageUrls} />
            </div>
          )}
        </div>
      ))}
    </div>
  );
}

function MechanicsView({ data, frameImageUrls }: { data: StructuredVQAAnalysis['mechanics']; frameImageUrls?: string[] }) {
  const [expandedGrounding, setExpandedGrounding] = useState<VisualGrounding | null>(null);
  return (
    <div className="space-y-4">
      <div>
        <h4 className="text-sm font-semibold mb-2 flex items-center gap-1.5">
          <Zap className="w-4 h-4 text-orange-500" />接触与力
        </h4>
        <div className="space-y-2">
          {data?.contacts?.map((c, i) => (
            <div key={i} className="p-2.5 rounded-lg bg-orange-50 dark:bg-orange-950 border border-orange-100 dark:border-orange-900">
              <div className="flex items-center justify-between mb-1.5">
                <div className="flex items-center gap-2">
                  <span className="font-medium text-sm">{c.contact_type}</span>
                  <ForceBadge level={c.force_level} />
                  {c.grounding && (
                    <GroundingBadge
                      grounding={c.grounding}
                      frameImageUrls={frameImageUrls}
                      colorClass="text-orange-500"
                      onClickView={setExpandedGrounding}
                    />
                  )}
                </div>
                <Badge variant="outline" className="text-xs font-mono">{c.timestamp}</Badge>
              </div>
              <div className="space-y-0.5 text-xs text-muted-foreground">
                <p>接触点: {c.contact_points}</p>
                <p>接触面积: {c.area}</p>
              </div>
              {expandedGrounding === c.grounding && c.grounding && (
                <div className="mt-2">
                  <GroundingDetail grounding={c.grounding} frameImageUrls={frameImageUrls} />
                </div>
              )}
            </div>
          ))}
        </div>
      </div>
      {data?.force_profile && (
        <div>
          <h4 className="text-sm font-semibold mb-1.5">力变化曲线</h4>
          <p className="text-sm text-muted-foreground leading-relaxed p-2.5 rounded bg-muted/50">
            {data.force_profile}
          </p>
        </div>
      )}
    </div>
  );
}

function ReasoningView({ data, frameImageUrls }: { data: StructuredVQAAnalysis['reasoning']; frameImageUrls?: string[] }) {
  const [expandedGrounding, setExpandedGrounding] = useState<VisualGrounding | null>(null);
  return (
    <div className="space-y-4">
      <div>
        <h4 className="text-sm font-semibold mb-2 flex items-center gap-1.5">
          <Brain className="w-4 h-4 text-pink-500" />动作推理
        </h4>
        <div className="space-y-2">
          {data?.action_justifications?.map((j, i) => (
            <div key={i} className="p-2.5 rounded-lg bg-pink-50 dark:bg-pink-950 border border-pink-100 dark:border-pink-900">
              <div className="flex items-center justify-between mb-1">
                <p className="font-medium text-sm">{j.action}</p>
                {j.grounding && (
                  <GroundingBadge
                    grounding={j.grounding}
                    frameImageUrls={frameImageUrls}
                    colorClass="text-pink-500"
                    onClickView={setExpandedGrounding}
                  />
                )}
              </div>
              <p className="text-xs text-muted-foreground mb-1.5">{j.reason}</p>
              {j.constraints?.length > 0 && (
                <div className="flex flex-wrap gap-1">
                  {j.constraints.map((c, k) => (
                    <Badge key={k} variant="outline" className="text-xs">{c}</Badge>
                  ))}
                </div>
              )}
              {expandedGrounding === j.grounding && j.grounding && (
                <div className="mt-2">
                  <GroundingDetail grounding={j.grounding} frameImageUrls={frameImageUrls} />
                </div>
              )}
            </div>
          ))}
        </div>
      </div>
      {data?.overall_strategy && (
        <div>
          <h4 className="text-sm font-semibold mb-1.5">整体策略</h4>
          <p className="text-sm text-muted-foreground leading-relaxed p-2.5 rounded bg-pink-50 dark:bg-pink-950 border border-pink-100 dark:border-pink-900">
            {data.overall_strategy}
          </p>
        </div>
      )}
    </div>
  );
}

function SummaryView({ data, frameImageUrls }: { data: StructuredVQAAnalysis['summary']; frameImageUrls?: string[] }) {
  return (
    <div className="space-y-4">
      <div className="p-3 rounded-lg bg-teal-50 dark:bg-teal-950 border border-teal-100 dark:border-teal-900">
        <div className="flex items-start justify-between gap-2 mb-2">
          <p className="font-semibold text-sm leading-snug">{data?.task_description}</p>
          {data?.success !== undefined && (
            data.success
              ? <CheckCircle2 className="w-5 h-5 text-green-500 shrink-0" />
              : <XCircle className="w-5 h-5 text-red-500 shrink-0" />
          )}
        </div>
        {data?.duration && (
          <p className="text-xs text-muted-foreground">时长: {data.duration}</p>
        )}
      </div>
      <div className="grid grid-cols-2 gap-2">
        <div className="p-2.5 rounded border bg-muted/30">
          <p className="text-xs font-medium text-muted-foreground mb-1">初始状态</p>
          <p className="text-sm">{data?.start_state}</p>
          {data?.grounding_start && (
            <div className="mt-1.5">
              <GroundingDetail grounding={data.grounding_start} frameImageUrls={frameImageUrls} />
            </div>
          )}
        </div>
        <div className="p-2.5 rounded border bg-muted/30">
          <p className="text-xs font-medium text-muted-foreground mb-1">结束状态</p>
          <p className="text-sm">{data?.end_state}</p>
          {data?.grounding_end && (
            <div className="mt-1.5">
              <GroundingDetail grounding={data.grounding_end} frameImageUrls={frameImageUrls} />
            </div>
          )}
        </div>
      </div>
      {data?.key_milestones?.length > 0 && (
        <div>
          <h4 className="text-sm font-semibold mb-2">关键里程碑</h4>
          <ol className="space-y-1.5">
            {data.key_milestones.map((m, i) => (
              <li key={i} className="flex items-start gap-2 text-sm">
                <span className="text-teal-500 font-bold shrink-0">{i + 1}.</span>
                <span>{m}</span>
              </li>
            ))}
          </ol>
        </div>
      )}
    </div>
  );
}

function TrajectoryView({ data, frameImageUrls }: { data: StructuredVQAAnalysis['trajectory']; frameImageUrls?: string[] }) {
  const [expandedGrounding, setExpandedGrounding] = useState<VisualGrounding | null>(null);
  return (
    <div className="space-y-4">
      <div>
        <h4 className="text-sm font-semibold mb-2 flex items-center gap-1.5">
          <Navigation className="w-4 h-4 text-indigo-500" />运动分段
        </h4>
        <div className="space-y-2">
          {data?.motion_segments?.map((seg, i) => (
            <div key={i} className="p-2.5 rounded-lg bg-indigo-50 dark:bg-indigo-950 border border-indigo-100 dark:border-indigo-900">
              <div className="flex items-center gap-2 mb-1.5 flex-wrap">
                <span className="font-medium text-sm">{seg.segment}</span>
                <MotionBadge type={seg.motion_type} />
                <Badge variant="secondary" className="text-xs">{seg.velocity}</Badge>
                {seg.grounding && (
                  <GroundingBadge
                    grounding={seg.grounding}
                    frameImageUrls={frameImageUrls}
                    colorClass="text-indigo-500"
                    onClickView={setExpandedGrounding}
                  />
                )}
                <Badge variant="outline" className="text-xs font-mono ml-auto">{seg.time_range}</Badge>
              </div>
              {seg.waypoints?.length > 0 && (
                <div className="flex flex-wrap gap-1 mt-1">
                  {seg.waypoints.map((wp, j) => (
                    <span key={j} className="text-xs px-1.5 py-0.5 rounded bg-indigo-100 dark:bg-indigo-900 text-indigo-700 dark:text-indigo-300">
                      {wp}
                    </span>
                  ))}
                </div>
              )}
              {expandedGrounding === seg.grounding && seg.grounding && (
                <div className="mt-2">
                  <GroundingDetail grounding={seg.grounding} frameImageUrls={frameImageUrls} />
                </div>
              )}
            </div>
          ))}
        </div>
      </div>
      {data?.overall_path && (
        <div>
          <h4 className="text-sm font-semibold mb-1.5">整体路径描述</h4>
          <p className="text-sm text-muted-foreground leading-relaxed p-2.5 rounded bg-muted/50">
            {data.overall_path}
          </p>
        </div>
      )}
    </div>
  );
}

// ─── Confidence Dashboard ─────────────────────────────────────────────────────
function ConfidenceDashboard({ scores }: { scores: VQAConfidenceScores }) {
  const categories = VQA_CATEGORIES;
  const avg = Object.values(scores).reduce((a, b) => a + b, 0) / Object.values(scores).length;

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-sm flex items-center gap-2">
          <BarChart2 className="w-4 h-4" />
          Confidence Scores
          <Badge variant="secondary" className="ml-auto font-mono">avg {Math.round(avg * 100)}%</Badge>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-2.5">
          {categories.map((cat) => (
            <div key={cat.id} className="flex items-center gap-2">
              <cat.icon className={`w-3.5 h-3.5 shrink-0 ${cat.color}`} />
              <span className="text-xs w-20 shrink-0">{cat.labelZh}</span>
              <ConfidenceGauge value={scores[cat.id as keyof VQAConfidenceScores] ?? 0} size="sm" />
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}

// ─── History Record Card ──────────────────────────────────────────────────────
function HistoryCard({
  record,
  onLoad,
}: {
  record: VQAAnnotationRecord;
  onLoad: (record: VQAAnnotationRecord) => void;
}) {
  const scores = record.analysis?.confidence_scores;
  const avg = scores
    ? Math.round(
        (Object.values(scores).reduce((a: number, b: unknown) => a + (b as number), 0) /
          Object.values(scores).length) *
          100
      )
    : 0;

  return (
    <div className="p-3 rounded-lg border hover:bg-accent/50 transition-colors cursor-pointer group"
      onClick={() => onLoad(record)}>
      <div className="flex items-start justify-between gap-2">
        <div className="min-w-0 flex-1">
          <p className="text-xs font-medium truncate">{record.videoPath?.split('/').pop()}</p>
          <p className="text-xs text-muted-foreground mt-0.5">
            {new Date(record.createdAt).toLocaleString('zh-CN')}
          </p>
        </div>
        <div className="flex items-center gap-1.5 shrink-0">
          <Badge variant="outline" className="text-xs">{record.model}</Badge>
          <Badge
            className={`text-xs ${avg >= 80 ? 'bg-green-100 text-green-700' : avg >= 60 ? 'bg-yellow-100 text-yellow-700' : 'bg-red-100 text-red-700'}`}>
            {avg}%
          </Badge>
        </div>
      </div>
      <Button size="sm" variant="ghost" className="mt-1 h-6 text-xs opacity-0 group-hover:opacity-100 transition-opacity">
        <Eye className="w-3 h-3 mr-1" />载入
      </Button>
    </div>
  );
}

// ─── Main Component ───────────────────────────────────────────────────────────
export default function StructuredVQA() {
  // Config state
  const [providers, setProviders] = useState<VLMProvider[]>([]);
  const [selectedProvider, setSelectedProvider] = useState<string>('gemini');
  const [apiKey, setApiKey] = useState('');
  const [selectedModel, setSelectedModel] = useState<string>('');
  const [numFrames, setNumFrames] = useState(32);

  // Local Ollama models
  const [localModels, setLocalModels] = useState<{ name: string; size: number }[]>([]);
  const [_localAvailable, setLocalAvailable] = useState(false);
  const [loadingLocal, setLoadingLocal] = useState(false);
  const [localError, setLocalError] = useState<string | null>(null);

  const isLocal = selectedProvider === 'local';

  // Dataset/Episode selection (like Auto-Annotation)
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [episodes, setEpisodes] = useState<Episode[]>([]);
  const [selectedDataset, setSelectedDataset] = useState<string>('');
  const [selectedEpisode, setSelectedEpisode] = useState<string>('');
  const [inputMode, setInputMode] = useState<'dataset' | 'upload'>('dataset');

  // Video state
  const [videoPath, setVideoPath] = useState('');
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [videoUrl, setVideoUrl] = useState<string>('');
  const [uploadedServerPath, setUploadedServerPath] = useState<string>('');
  const [uploadProgress, setUploadProgress] = useState(0);
  const [isUploading, setIsUploading] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Analysis state
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [progressMsg, setProgressMsg] = useState('');
  const [analysis, setAnalysis] = useState<StructuredVQAAnalysis | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [activeCategory, setActiveCategory] = useState('temporal');

  // History state
  const [history, setHistory] = useState<VQAAnnotationRecord[]>([]);
  const [showHistory, setShowHistory] = useState(false);

  // Fetch providers & history & datasets on mount
  useEffect(() => {
    api.getVLMProviders().then(setProviders).catch(console.error);
    api.getStructuredAnalyses().then(setHistory).catch(console.error);
    api.getDatasets().then(setDatasets).catch(console.error);

    // Restore API key from localStorage
    const saved = localStorage.getItem('vlm_api_key');
    if (saved) setApiKey(saved);
  }, []);

  // Load episodes when dataset changes
  useEffect(() => {
    if (selectedDataset) {
      api.getDatasetEpisodes(selectedDataset).then(setEpisodes).catch(console.error);
    } else {
      setEpisodes([]);
    }
  }, [selectedDataset]);

  // 切换到本地模式时读取 Ollama 模型列表
  useEffect(() => {
    if (!isLocal) return;
    setLoadingLocal(true);
    setLocalError(null);
    api.getLocalVLMModels()
      .then((res) => {
        setLocalAvailable(res.available);
        setLocalModels(res.models || []);
        if (res.error) setLocalError(res.error);
        // 默认选择第一个山川视觉模型
        if (res.models?.length > 0 && !selectedModel) {
          setSelectedModel(res.models[0].name);
        }
      })
      .catch(() => {
        setLocalAvailable(false);
        setLocalError('Ollama 未运行，请先执行: ollama serve');
      })
      .finally(() => setLoadingLocal(false));
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isLocal]);

  const currentProvider = providers.find((p) => p.id === selectedProvider);

  const handleFileSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setVideoFile(file);
    setVideoPath(file.name);
    const url = URL.createObjectURL(file);
    setVideoUrl(url);
    setAnalysis(null);
    setError(null);
    setUploadedServerPath('');
    setUploadProgress(0);
    setIsUploading(true);
    try {
      const result = await api.uploadVideo(file, setUploadProgress);
      setUploadedServerPath(result.serverPath);
    } catch (err: any) {
      setError(`Upload failed: ${err.message}. Using local path fallback.`);
    } finally {
      setIsUploading(false);
      setUploadProgress(0);
    }
  };

  const saveApiKey = () => {
    localStorage.setItem('vlm_api_key', apiKey);
  };

  const runAnalysis = async () => {
    if (!videoPath) return;
    if (!isLocal && !apiKey) {
      setError('请输入 API Key');
      return;
    }
    if (isLocal && !selectedModel) {
      setError('请选择本地 Ollama 模型');
      return;
    }

    setIsAnalyzing(true);
    setError(null);
    setAnalysis(null);

    // Simulate progress stages
    const stages = [
      { pct: 10, msg: '提取视频帧...' },
      { pct: 30, msg: '上传帧到 VLM...' },
      { pct: 60, msg: 'VLM 分析中 (7 类 VQA)...' },
      { pct: 85, msg: '解析结构化输出...' },
      { pct: 95, msg: '保存标注记录...' },
    ];

    let stageIdx = 0;
    const progressTimer = setInterval(() => {
      if (stageIdx < stages.length) {
        setProgress(stages[stageIdx].pct);
        setProgressMsg(stages[stageIdx].msg);
        stageIdx++;
      }
    }, 800);

    try {
      const result = await api.runStructuredVQAAnalysis({
        videoPath: uploadedServerPath || (videoFile ? `/tmp/${videoFile.name}` : videoPath),
        provider: selectedProvider,
        apiKey: isLocal ? 'local' : apiKey,
        numFrames: isLocal ? Math.min(numFrames, 16) : numFrames,
        model: isLocal ? selectedModel : (selectedModel || undefined),
      });

      clearInterval(progressTimer);
      setProgress(100);
      setProgressMsg('分析完成！');

      if (result.success && result.analysis) {
        setAnalysis(result.analysis);
        // Refresh history
        api.getStructuredAnalyses().then(setHistory).catch(console.error);
      } else {
        setError('分析返回空结果，请检查视频路径和 API Key');
      }
    } catch (err: any) {
      clearInterval(progressTimer);
      setError(err.message || '分析失败');
    } finally {
      setIsAnalyzing(false);
      setTimeout(() => { setProgress(0); setProgressMsg(''); }, 1500);
    }
  };

  const exportAnalysis = () => {
    if (!analysis) return;
    const blob = new Blob([JSON.stringify(analysis, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `vqa_analysis_${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const loadFromHistory = (record: VQAAnnotationRecord) => {
    setAnalysis(record.analysis);
    setVideoPath(record.videoPath || '');
    setShowHistory(false);
    setError(null);
  };

  const activeCat = VQA_CATEGORIES.find((c) => c.id === activeCategory)!;

  return (
    <div className="flex flex-col h-full gap-4 p-4">

      {/* ── Header ── */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-bold flex items-center gap-2">
            <Layers className="w-5 h-5 text-primary" />
            Video Analysis
          </h1>
          <p className="text-sm text-muted-foreground mt-0.5">
            使用 VLM 将机器人视频分解为 7 类结构化问答标注
          </p>
        </div>
        <div className="flex items-center gap-2">
          {history.length > 0 && (
            <Button variant="outline" size="sm" onClick={() => setShowHistory(!showHistory)}>
              <RefreshCw className="w-3.5 h-3.5 mr-1.5" />
              历史 ({history.length})
            </Button>
          )}
          {analysis && (
            <Button variant="outline" size="sm" onClick={exportAnalysis}>
              <Download className="w-3.5 h-3.5 mr-1.5" />
              导出 JSON
            </Button>
          )}
        </div>
      </div>

      <div className="flex flex-1 gap-4 min-h-0">

        {/* ── Left Panel: Config + History ── */}
        <div className="w-72 shrink-0 flex flex-col gap-3">

          {/* Provider Config */}
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm flex items-center gap-1.5">
                <Settings2 className="w-4 h-4" />模型配置
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              {/* Provider */}
              <div className="space-y-1.5">
                <Label className="text-xs">VLM Provider</Label>
                <Select value={selectedProvider} onValueChange={(v) => { setSelectedProvider(v); setSelectedModel(''); }}>
                  <SelectTrigger className="h-8 text-sm">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {providers.map((p) => (
                      <SelectItem key={p.id} value={p.id}>
                        <div className="flex items-center gap-1.5">
                          {p.recommended && <Star className="w-3 h-3 text-yellow-500" />}
                          {p.name}
                        </div>
                      </SelectItem>
                    ))}
                    {providers.length === 0 && (
                      <>
                        <SelectItem value="gemini">⭐ Google Gemini 1.5 Pro</SelectItem>
                        <SelectItem value="claude">Anthropic Claude 3.5</SelectItem>
                        <SelectItem value="openai">OpenAI GPT-4o</SelectItem>
                      </>
                    )}
                  </SelectContent>
                </Select>
                {currentProvider && (
                  <p className="text-xs text-muted-foreground">{currentProvider.description}</p>
                )}
              </div>

              {/* Model override */}
              {currentProvider?.models && currentProvider.models.length > 0 && (
                <div className="space-y-1.5">
                  <Label className="text-xs">具体模型（可选）</Label>
                  <Select value={selectedModel || '__default__'} onValueChange={(v) => setSelectedModel(v === '__default__' ? '' : v)}>
                    <SelectTrigger className="h-8 text-sm">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="__default__">默认</SelectItem>
                      {currentProvider.models.map((m) => (
                        <SelectItem key={m} value={m}>{m}</SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
              )}

              {/* API Key 或本地模型 */}
              {isLocal ? (
                <div className="space-y-1.5">
                  <Label className="text-xs flex items-center justify-between">
                    <span>Ollama 模型</span>
                    {loadingLocal && <span className="text-muted-foreground">Loading...</span>}
                  </Label>
                  {localError ? (
                    <div className="text-xs text-red-500 p-2 rounded bg-red-50 border border-red-200">
                      {localError}
                    </div>
                  ) : localModels.length > 0 ? (
                    <Select value={selectedModel} onValueChange={setSelectedModel}>
                      <SelectTrigger className="h-8 text-sm">
                        <SelectValue placeholder="选择模型..." />
                      </SelectTrigger>
                      <SelectContent>
                        {localModels.map((m) => (
                          <SelectItem key={m.name} value={m.name}>
                            <span className="flex items-center gap-1.5">
                              <Cpu className="w-3 h-3 text-green-500" />
                              <span className="font-mono text-xs">{m.name}</span>
                              <span className="text-muted-foreground text-xs">
                                {m.size ? `${(m.size / 1e9).toFixed(1)}G` : ''}
                              </span>
                            </span>
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  ) : (
                    <div className="text-xs text-muted-foreground p-2 rounded bg-muted">
                      {loadingLocal ? '检测中...' : '未检测到视觉模型'}
                    </div>
                  )}
                  <p className="text-xs text-green-600 font-medium">✓ 完全离线，无需 API Key</p>
                </div>
              ) : (
                <div className="space-y-1.5">
                  <Label className="text-xs">API Key</Label>
                  <div className="flex gap-1.5">
                    <Input
                      type="password"
                      placeholder="sk-..."
                      value={apiKey}
                      onChange={(e) => setApiKey(e.target.value)}
                      className="h-8 text-sm flex-1"
                    />
                    <Button size="sm" variant="outline" className="h-8 px-2" onClick={saveApiKey}>
                      保存
                    </Button>
                  </div>
                </div>
              )}

              {/* Num Frames */}
              <div className="space-y-1.5">
                <Label className="text-xs flex justify-between">
                  <span>分析帧数{isLocal ? <span className="text-orange-500 ml-1">(本地最多 16)</span> : null}</span>
                  <span className="font-mono">{isLocal ? Math.min(numFrames, 16) : numFrames}</span>
                </Label>
                <input
                  type="range" min={8} max={isLocal ? 16 : 64} step={isLocal ? 4 : 8}
                  value={isLocal ? Math.min(numFrames, 16) : numFrames}
                  onChange={(e) => setNumFrames(Number(e.target.value))}
                  className="w-full h-1.5 cursor-pointer accent-primary"
                />
                <div className="flex justify-between text-xs text-muted-foreground">
                  <span>8 (快速)</span><span>{isLocal ? '16 (本地最大)' : '64 (精准)'}</span>
                </div>
              </div>

              {/* Capability tags */}
              <div className="flex flex-wrap gap-1">
                {VQA_CATEGORIES.map((cat) => (
                  <span key={cat.id} className={`text-xs px-1.5 py-0.5 rounded ${cat.badge}`}>
                    {cat.label}
                  </span>
                ))}
              </div>
            </CardContent>
          </Card>

          {/* Video Input */}
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm flex items-center gap-1.5">
                <Upload className="w-4 h-4" />视频输入
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-2.5">
              {/* Input mode toggle */}
              <div className="grid grid-cols-2 gap-1 p-0.5 rounded-md bg-muted">
                <button
                  className={`text-xs py-1 px-2 rounded transition-colors ${inputMode === 'dataset' ? 'bg-background shadow font-medium' : 'text-muted-foreground hover:text-foreground'}`}
                  onClick={() => setInputMode('dataset')}
                >
                  📊 From Dataset
                </button>
                <button
                  className={`text-xs py-1 px-2 rounded transition-colors ${inputMode === 'upload' ? 'bg-background shadow font-medium' : 'text-muted-foreground hover:text-foreground'}`}
                  onClick={() => setInputMode('upload')}
                >
                  📁 Upload / Path
                </button>
              </div>

              {inputMode === 'dataset' ? (
                <div className="space-y-2">
                  <div className="space-y-1">
                    <Label className="text-xs">Dataset</Label>
                    <Select value={selectedDataset} onValueChange={(v) => { setSelectedDataset(v); setSelectedEpisode(''); }}>
                      <SelectTrigger className="h-8 text-sm">
                        <SelectValue placeholder="Select dataset..." />
                      </SelectTrigger>
                      <SelectContent>
                        {datasets.map(ds => (
                          <SelectItem key={ds.id} value={ds.id}>
                            <div className="flex items-center gap-1.5">
                              <span>{ds.name}</span>
                              <span className="text-muted-foreground text-xs">({ds.episodeCount} ep)</span>
                            </div>
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                  {selectedDataset && (
                    <div className="space-y-1">
                      <Label className="text-xs">Episode</Label>
                      <Select value={selectedEpisode} onValueChange={(v) => {
                        setSelectedEpisode(v);
                        const ep = episodes.find(e => e.id === v);
                        if (ep) {
                          setVideoPath(`episode_${v}`);
                          setAnalysis(null);
                          setError(null);
                        }
                      }}>
                        <SelectTrigger className="h-8 text-sm">
                          <SelectValue placeholder="Select episode..." />
                        </SelectTrigger>
                        <SelectContent>
                          {episodes.map(ep => (
                            <SelectItem key={ep.id} value={ep.id}>
                              <div className="flex items-center gap-1.5">
                                <span className="truncate">{ep.name || ep.id}</span>
                                <span className="text-muted-foreground text-xs shrink-0">{ep.frameCount}f · {ep.duration?.toFixed(1)}s</span>
                              </div>
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>
                  )}
                  {selectedEpisode && (() => {
                    const ep = episodes.find(e => e.id === selectedEpisode);
                    const ds = datasets.find(d => d.id === selectedDataset);
                    if (!ep) return null;
                    const sensors = ds?.sensorConfig?.sensors || [];
                    return (
                      <div className="p-2 rounded border bg-muted/30 space-y-1.5">
                        <div className="flex items-center justify-between text-xs">
                          <span className="font-medium">{ep.name || ep.id}</span>
                          <Badge variant="outline" className="text-[10px]">{(ep as any).skill || ds?.robotType}</Badge>
                        </div>
                        <div className="flex gap-2 text-[10px] text-muted-foreground">
                          <span>{ep.frameCount} frames</span>
                          <span>{ep.duration?.toFixed(1)}s</span>
                          <span>{(ep as any).fps || 30} fps</span>
                        </div>
                        {sensors.length > 0 && (
                          <div className="flex flex-wrap gap-1">
                            {sensors.map((s: any, i: number) => (
                              <span key={i} className="text-[9px] px-1.5 py-0.5 rounded bg-primary/10 text-primary">
                                {s.name || s.type}
                              </span>
                            ))}
                          </div>
                        )}
                      </div>
                    );
                  })()}
                </div>
              ) : (
                <>
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept="video/*"
                    className="hidden"
                    onChange={handleFileSelect}
                  />
                  <Button
                    variant="outline"
                    size="sm"
                    className="w-full"
                    onClick={() => fileInputRef.current?.click()}
                  >
                    <Upload className="w-3.5 h-3.5 mr-1.5" />
                    选择视频文件
                  </Button>
                  <div className="space-y-1">
                    <Label className="text-xs">或输入服务器路径</Label>
                    <Input
                      placeholder="/path/to/video.mp4"
                      value={videoPath}
                      onChange={(e) => setVideoPath(e.target.value)}
                      className="h-8 text-xs font-mono"
                    />
                  </div>
                  {isUploading && (
                    <div className="space-y-1">
                      <div className="flex justify-between text-xs text-muted-foreground">
                        <span>Uploading to server…</span>
                        <span>{uploadProgress}%</span>
                      </div>
                      <Progress value={uploadProgress} className="h-1.5" />
                    </div>
                  )}
                  {uploadedServerPath && !isUploading && (
                    <div className="flex items-center gap-1.5 text-xs text-green-600 dark:text-green-400">
                      <CheckCircle2 className="w-3.5 h-3.5" />
                      <span className="font-mono truncate">Ready: {uploadedServerPath.split('/').pop()}</span>
                    </div>
                  )}
                  {videoUrl && (
                    <video
                      src={videoUrl}
                      controls
                      className="w-full rounded border aspect-video object-cover bg-black"
                    />
                  )}
                </>
              )}
            </CardContent>
          </Card>

          {/* Run Button */}
          <div className="grid grid-cols-2 gap-2">
            <Button
              className="col-span-2"
              size="lg"
              disabled={isAnalyzing || isUploading || (!videoPath && !videoFile) || (!isLocal && !apiKey) || (isLocal && !selectedModel)}
              onClick={runAnalysis}
            >
              {isAnalyzing ? (
                <><Loader2 className="w-4 h-4 mr-2 animate-spin" />分析中...</>
              ) : (
                <><Cpu className="w-4 h-4 mr-2" />运行 VQA 分析</>
              )}
            </Button>
            <Button
              variant="outline"
              size="sm"
              className="col-span-2"
              disabled={isAnalyzing}
              onClick={async () => {
                setIsAnalyzing(true);
                setError(null);
                setProgress(20);
                setProgressMsg('Loading demo pick-and-place analysis…');
                try {
                  const result = await api.runDemoAnalysis('Pick and Place Red Cup');
                  setProgress(100);
                  setProgressMsg('Demo loaded!');
                  if (result.analysis) {
                    setAnalysis(result.analysis);
                  }
                } catch (err: any) {
                  setError(`Demo failed: ${err.message}`);
                } finally {
                  setIsAnalyzing(false);
                  setTimeout(() => { setProgress(0); setProgressMsg(''); }, 1500);
                }
              }}
            >
              <Sparkles className="w-3.5 h-3.5 mr-1.5" />
              Demo Mode (no video/API key needed)
            </Button>
          </div>

          {/* Progress */}
          {isAnalyzing && (
            <div className="space-y-1.5">
              <Progress value={progress} className="h-2" />
              <p className="text-xs text-center text-muted-foreground">{progressMsg}</p>
            </div>
          )}

          {/* Error */}
          {error && (
            <div className="flex items-start gap-2 p-2.5 rounded-lg bg-red-50 dark:bg-red-950 border border-red-200 dark:border-red-800 text-red-700 dark:text-red-400 text-xs">
              <AlertTriangle className="w-3.5 h-3.5 shrink-0 mt-0.5" />
              <span>{error}</span>
            </div>
          )}

          {/* History */}
          {showHistory && history.length > 0 && (
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-xs">历史分析</CardTitle>
              </CardHeader>
              <CardContent className="p-2">
                <ScrollArea className="h-52">
                  <div className="space-y-1.5 pr-2">
                    {history.map((rec) => (
                      <HistoryCard key={rec.id} record={rec} onLoad={loadFromHistory} />
                    ))}
                  </div>
                </ScrollArea>
              </CardContent>
            </Card>
          )}
        </div>

        {/* ── Right Panel: Analysis Results ── */}
        <div className="flex-1 min-w-0 flex flex-col gap-3">
          {!analysis && !isAnalyzing && (
            <div className="flex-1 flex flex-col items-center justify-center text-center text-muted-foreground gap-4">
              <div className="w-16 h-16 rounded-full bg-muted flex items-center justify-center">
                <Layers className="w-8 h-8" />
              </div>
              <div>
                <p className="font-medium">选择视频并运行分析</p>
                <p className="text-sm mt-1">VLM 将视频分解为 7 类结构化 VQA 标注</p>
              </div>
              <div className="grid grid-cols-3 gap-2 max-w-md text-left">
                {VQA_CATEGORIES.map((cat) => (
                  <div key={cat.id} className={`p-2 rounded-lg border text-xs ${cat.bg} ${cat.border}`}>
                    <div className={`flex items-center gap-1 mb-0.5 font-medium ${cat.color}`}>
                      <cat.icon className="w-3 h-3" />
                      {cat.label}
                    </div>
                    <p className="text-muted-foreground leading-tight">{cat.description}</p>
                  </div>
                ))}
              </div>
            </div>
          )}

          {analysis && (
            <>
              {/* Metadata bar */}
              <div className="flex items-center gap-3 text-xs text-muted-foreground p-2.5 rounded-lg border bg-card flex-wrap">
                <span className="flex items-center gap-1">
                  <Cpu className="w-3 h-3" />
                  {analysis.metadata?.model}
                </span>
                <Separator orientation="vertical" className="h-3" />
                <span>{analysis.metadata?.num_frames_analyzed} frames</span>
                <Separator orientation="vertical" className="h-3" />
                <span>{analysis.metadata?.video_info?.duration?.toFixed(1)}s</span>
                <Separator orientation="vertical" className="h-3" />
                <span>{analysis.metadata?.video_info?.fps?.toFixed(1)} fps</span>
                {analysis.summary?.success !== undefined && (
                  <>
                    <Separator orientation="vertical" className="h-3" />
                    {analysis.summary.success
                      ? <span className="flex items-center gap-1 text-green-600"><CheckCircle2 className="w-3 h-3" />Task Success</span>
                      : <span className="flex items-center gap-1 text-red-500"><XCircle className="w-3 h-3" />Task Failed</span>
                    }
                  </>
                )}
              </div>

              <div className="flex gap-3 flex-1 min-h-0">
                {/* Category Tabs (vertical) */}
                <div className="flex flex-col gap-1 shrink-0 w-36">
                  {VQA_CATEGORIES.map((cat) => {
                    const score = analysis.confidence_scores?.[cat.id as keyof VQAConfidenceScores];
                    const isActive = activeCategory === cat.id;
                    return (
                      <button
                        key={cat.id}
                        onClick={() => setActiveCategory(cat.id)}
                        className={`flex items-center gap-2 px-2.5 py-2 rounded-lg text-left transition-all text-xs border
                          ${isActive
                            ? `${cat.bg} ${cat.border} font-semibold`
                            : 'hover:bg-accent border-transparent'
                          }`}
                      >
                        <cat.icon className={`w-3.5 h-3.5 shrink-0 ${cat.color}`} />
                        <div className="min-w-0 flex-1">
                          <div>{cat.label}</div>
                          <div className="text-muted-foreground text-[10px]">{cat.labelZh}</div>
                          {score !== undefined && (
                            <div className="mt-0.5 h-1 rounded-full bg-muted overflow-hidden">
                              <div
                                className={`h-full rounded-full ${score >= 0.8 ? 'bg-green-500' : score >= 0.6 ? 'bg-yellow-500' : 'bg-red-400'}`}
                                style={{ width: `${score * 100}%` }}
                              />
                            </div>
                          )}
                        </div>
                      </button>
                    );
                  })}
                </div>

                {/* Content Panel */}
                <div className="flex-1 min-w-0 flex flex-col gap-3">
                  <Card className={`flex-1 border ${activeCat.border}`}>
                    <CardHeader className={`pb-2 ${activeCat.bg} rounded-t-lg`}>
                      <CardTitle className="text-sm flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <activeCat.icon className={`w-4 h-4 ${activeCat.color}`} />
                          <span>{activeCat.label}</span>
                          <span className="text-muted-foreground font-normal">— {activeCat.labelZh}</span>
                        </div>
                        {analysis.confidence_scores?.[activeCategory as keyof VQAConfidenceScores] !== undefined && (
                          <div className="flex items-center gap-2">
                            <ConfidenceGauge
                              value={analysis.confidence_scores[activeCategory as keyof VQAConfidenceScores]}
                              size="sm"
                            />
                          </div>
                        )}
                      </CardTitle>
                    </CardHeader>
                    <CardContent className="pt-3">
                      <ScrollArea className="h-[calc(100vh-380px)]">
                        <div className="pr-3">
                          {activeCategory === 'temporal' && <TemporalView data={analysis.temporal} frameImageUrls={analysis.metadata?.frame_image_urls} />}
                          {activeCategory === 'spatial' && <SpatialView data={analysis.spatial} frameImageUrls={analysis.metadata?.frame_image_urls} />}
                          {activeCategory === 'attribute' && <AttributeView data={analysis.attribute} frameImageUrls={analysis.metadata?.frame_image_urls} />}
                          {activeCategory === 'mechanics' && <MechanicsView data={analysis.mechanics} frameImageUrls={analysis.metadata?.frame_image_urls} />}
                          {activeCategory === 'reasoning' && <ReasoningView data={analysis.reasoning} frameImageUrls={analysis.metadata?.frame_image_urls} />}
                          {activeCategory === 'summary' && <SummaryView data={analysis.summary} frameImageUrls={analysis.metadata?.frame_image_urls} />}
                          {activeCategory === 'trajectory' && <TrajectoryView data={analysis.trajectory} frameImageUrls={analysis.metadata?.frame_image_urls} />}
                        </div>
                      </ScrollArea>
                    </CardContent>
                  </Card>
                </div>

                {/* Right sidebar: Confidence + Key frames */}
                <div className="w-52 shrink-0 flex flex-col gap-3">
                  {analysis.confidence_scores && (
                    <ConfidenceDashboard scores={analysis.confidence_scores} />
                  )}

                  {/* Temporal Consistency Check */}
                  {analysis.temporal_consistency && !analysis.temporal_consistency.consistent && (
                    <TemporalConsistencyAlert conflicts={analysis.temporal_consistency.conflicts} />
                  )}
                  {analysis.temporal_consistency?.consistent && (
                    <div className="flex items-center gap-1.5 text-xs text-green-600 p-2 rounded border border-green-200 bg-green-50 dark:bg-green-950 dark:border-green-800">
                      <CheckCircle2 className="w-3.5 h-3.5" />
                      <span>时间一致性校验通过</span>
                    </div>
                  )}

                  {analysis.visual_evidence?.key_frames?.length > 0 && (
                    <Card>
                      <CardHeader className="pb-2">
                        <CardTitle className="text-xs flex items-center gap-1.5">
                          <Eye className="w-3.5 h-3.5" />关键帧
                        </CardTitle>
                      </CardHeader>
                      <CardContent className="p-2">
                        <ScrollArea className="h-52">
                          <div className="space-y-1.5 pr-1">
                            {analysis.visual_evidence.key_frames.map((kf, i) => (
                              <div key={i} className="p-2 rounded border text-xs bg-muted/30">
                                <div className="flex justify-between items-center mb-0.5">
                                  <span className="font-mono text-primary">#{kf.frame_idx}</span>
                                  <span className="text-muted-foreground">{kf.timestamp}</span>
                                </div>
                                <p className="text-muted-foreground leading-tight">{kf.significance}</p>
                              </div>
                            ))}
                          </div>
                        </ScrollArea>
                      </CardContent>
                    </Card>
                  )}

                  {/* Info tip */}
                  <div className="flex items-start gap-1.5 text-xs text-muted-foreground p-2 rounded border">
                    <Info className="w-3 h-3 shrink-0 mt-0.5" />
                    <span>所有标注均基于视觉证据，确保时间一致性</span>
                  </div>
                </div>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
