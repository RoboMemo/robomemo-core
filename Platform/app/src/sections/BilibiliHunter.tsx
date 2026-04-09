/**
 * BilibiliHunter.tsx
 * ==================
 * B站视频搜索、预筛选、下载与SFT流水线界面
 * 整合 OpenClaw 的自动搜索能力
 */

import { useState, useCallback } from 'react';
import {
  Search,
  Download,
  Play,
  RefreshCw,
  CheckCircle,
  AlertTriangle,
  XCircle,
  Info,
  BarChart3,
  FlaskConical,
  Bot,
  Sparkles,
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '@/components/ui/dialog';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Slider } from '@/components/ui/slider';
import { Label } from '@/components/ui/label';
import { Progress } from '@/components/ui/progress';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Checkbox } from '@/components/ui/checkbox';
import { Separator } from '@/components/ui/separator';
import { ScrollArea } from '@/components/ui/scroll-area';

interface BilibiliVideo {
  bvid: string;
  title: string;
  url: string;
  snippet: string;
  duration: string;
  views: number;
  author: string;
  cover: string;
  pubdate: number;
  score?: number;
}

interface PrescreenResult {
  bvid: string;
  title: string;
  score: number;
  verdict: string;
  reasons: string[];
  details: {
    author: string;
    duration: number;
    views: number;
    resolution: string;
    tags: string[];
    cover: string;
  };
}

interface PipelineJob {
  id: string;
  status: 'processing' | 'completed' | 'failed';
  stage: string;
  progress: number;
  results: Record<string, unknown>;
  error?: string;
  outputDir?: string;
  completedAt?: string;
}

export default function BilibiliHunter() {
  // 搜索状态
  const [keyword, setKeyword] = useState('拧螺丝');
  const [searching, setSearching] = useState(false);
  const [searchResults, setSearchResults] = useState<BilibiliVideo[]>([]);
  const [totalResults, setTotalResults] = useState(0);

  // Agent 自动搜索状态
  const [intent, setIntent] = useState('寻找拧螺丝的第一人称视角视频');
  const [hunting, setHunting] = useState(false);

  // 预筛选状态
  const [selectedBvids, setSelectedBvids] = useState<string[]>([]);
  const [prescreening, setPrescreening] = useState(false);
  const [prescreenResults, setPrescreenResults] = useState<PrescreenResult[]>([]);
  const [minScore, setMinScore] = useState(50);

  // 下载状态
  const [downloading, setDownloading] = useState(false);
  const [downloadProgress, setDownloadProgress] = useState(0);

  // SFT流水线状态
  const [pipelineDialogOpen, setPipelineDialogOpen] = useState(false);
  const [pipelineConfig, setPipelineConfig] = useState({
    maxVideos: 10,
    vlmBackend: 'mock',
    dryRun: true,
  });
  const [pipelineJob, setPipelineJob] = useState<PipelineJob | null>(null);
  const [pipelinePolling, setPipelinePolling] = useState(false);

  // 普通搜索
  const handleSearch = useCallback(async () => {
    if (!keyword.trim()) return;

    setSearching(true);
    try {
      const response = await fetch('/api/bilibili/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          keyword: keyword.trim(),
          pageSize: 20,
          page: 1,
          order: 'totalrank',
        }),
      });

      const data = await response.json();
      if (data.success) {
        setSearchResults(data.videos);
        setTotalResults(data.total);
      } else {
        console.error('Search failed:', data.error);
      }
    } catch (error) {
      console.error('Search error:', error);
    } finally {
      setSearching(false);
    }
  }, [keyword]);

  // Agent 自动搜索（OpenClaw 风格）
  const handleHunt = useCallback(async () => {
    if (!intent.trim()) return;

    setHunting(true);
    try {
      const response = await fetch('/api/bilibili/hunt', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          intent: intent.trim(),
          minScore: minScore,
          maxResults: 20,
        }),
      });

      const data = await response.json();
      if (data.bvids && data.bvids.length > 0) {
        // 将 Agent 结果转换为搜索结果格式
        setSearchResults(data.results.map((r: BilibiliVideo & { score: number }) => ({
          ...r,
          views: r.views || 0,
          duration: r.duration || '0:00',
          author: r.author || '未知',
          snippet: r.snippet || '',
          cover: '',
          pubdate: 0,
        })));
        setTotalResults(data.total_found || data.results.length);
        setSelectedBvids(data.bvids);
      } else {
        console.log('No videos found by Agent');
      }
    } catch (error) {
      console.error('Hunt error:', error);
    } finally {
      setHunting(false);
    }
  }, [intent, minScore]);

  // 预筛选视频
  const handlePrescreen = useCallback(async () => {
    if (selectedBvids.length === 0) return;

    setPrescreening(true);
    try {
      const response = await fetch('/api/bilibili/prescreen', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          bvids: selectedBvids,
          minScore,
        }),
      });

      const data = await response.json();
      setPrescreenResults(data.results || []);
    } catch (error) {
      console.error('Prescreen error:', error);
    } finally {
      setPrescreening(false);
    }
  }, [selectedBvids, minScore]);

  // 下载视频
  const handleDownload = useCallback(async (bvids: string[]) => {
    if (bvids.length === 0) return;

    setDownloading(true);
    setDownloadProgress(0);

    try {
      const response = await fetch('/api/bilibili/download', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ bvids }),
      });

      const data = await response.json();
      
      const pollInterval = setInterval(async () => {
        try {
          const statusRes = await fetch(`/api/bilibili/jobs/${data.jobId}`);
          const statusData = await statusRes.json();
          setDownloadProgress(statusData.progress || 0);
          if (statusData.status === 'completed' || statusData.status === 'failed') {
            clearInterval(pollInterval);
            setDownloading(false);
          }
        } catch {
          clearInterval(pollInterval);
          setDownloading(false);
        }
      }, 2000);
    } catch (error) {
      console.error('Download error:', error);
      setDownloading(false);
    }
  }, []);

  // 运行SFT流水线
  const handleRunPipeline = useCallback(async () => {
    setPipelinePolling(true);
    setPipelineJob(null);

    try {
      const response = await fetch('/api/bilibili/pipeline', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          keyword: keyword.trim(),
          bvids: selectedBvids.length > 0 ? selectedBvids : undefined,
          minScore,
          ...pipelineConfig,
        }),
      });

      const data = await response.json();
      setPipelineJob({
        id: data.jobId,
        status: 'processing',
        stage: 'initializing',
        progress: 0,
        results: {},
      });

      const pollInterval = setInterval(async () => {
        try {
          const statusRes = await fetch(`/api/bilibili/jobs/${data.jobId}`);
          const statusData = await statusRes.json();
          setPipelineJob(statusData);
          if (statusData.status === 'completed' || statusData.status === 'failed') {
            clearInterval(pollInterval);
            setPipelinePolling(false);
          }
        } catch {
          clearInterval(pollInterval);
          setPipelinePolling(false);
        }
      }, 3000);

      setPipelineDialogOpen(false);
    } catch (error) {
      console.error('Pipeline error:', error);
      setPipelinePolling(false);
    }
  }, [keyword, selectedBvids, minScore, pipelineConfig]);

  const toggleBvid = (bvid: string) => {
    setSelectedBvids((prev) =>
      prev.includes(bvid) ? prev.filter((id) => id !== bvid) : [...prev, bvid]
    );
  };

  const toggleAll = () => {
    if (selectedBvids.length === searchResults.length) {
      setSelectedBvids([]);
    } else {
      setSelectedBvids(searchResults.map((v) => v.bvid));
    }
  };

  const getScoreVariant = (score: number): "default" | "secondary" | "destructive" | "outline" => {
    if (score >= 70) return 'default';
    if (score >= 50) return 'secondary';
    return 'destructive';
  };

  const getVerdictIcon = (verdict: string) => {
    if (verdict.startsWith('✅')) return <CheckCircle className="h-4 w-4 text-green-500" />;
    if (verdict.startsWith('⚠️')) return <AlertTriangle className="h-4 w-4 text-yellow-500" />;
    if (verdict.startsWith('❓')) return <Info className="h-4 w-4 text-blue-500" />;
    return <XCircle className="h-4 w-4 text-red-500" />;
  };

  return (
    <div className="p-6 space-y-6">
      {/* 标题 */}
      <div className="flex items-center gap-2">
        <FlaskConical className="h-6 w-6" />
        <h1 className="text-2xl font-bold">Bilibili 视频采集</h1>
        <Badge variant="outline">OpenClaw × RoboMemo</Badge>
      </div>

      <p className="text-sm text-muted-foreground">
        搜索B站机器人操作视频 → 智能筛选 → 下载 → VLM自动标注 → LeRobot格式导出 → π₀.5 SFT训练
      </p>

      {/* Agent 自动搜索区域 */}
      <Card className="border-primary/50 bg-primary/5">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Bot className="h-5 w-5" />
            Agent 自动搜索（OpenClaw 风格）
          </CardTitle>
          <CardDescription>
            输入你的意图，Agent 会自动搜索、过滤并返回高质量的 BV 号
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex gap-2">
            <Input
              placeholder="例如：寻找拧螺丝的第一人称视角视频"
              value={intent}
              onChange={(e) => setIntent(e.target.value)}
              className="flex-1"
            />
            <Button onClick={handleHunt} disabled={hunting}>
              {hunting ? (
                <RefreshCw className="h-4 w-4 animate-spin mr-2" />
              ) : (
                <Sparkles className="h-4 w-4 mr-2" />
              )}
              自动搜索
            </Button>
          </div>
          <p className="text-xs text-muted-foreground mt-2">
            Agent 会自动：提取关键词 → 多轮搜索 → 过滤不相关内容（游戏/动画/广告）→ 返回干净 BV 号列表
          </p>
        </CardContent>
      </Card>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* 左侧：搜索与结果 */}
        <div className="lg:col-span-2 space-y-4">
          {/* 普通搜索卡片 */}
          <Card>
            <CardHeader>
              <CardTitle>手动搜索</CardTitle>
              <CardDescription>手动输入关键词搜索B站视频</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="flex gap-2">
                <Input
                  placeholder="搜索关键词，如：拧螺丝、机器人抓取、DIY..."
                  value={keyword}
                  onChange={(e) => setKeyword(e.target.value)}
                  onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
                  className="flex-1"
                />
                <Button onClick={handleSearch} disabled={searching}>
                  {searching ? (
                    <RefreshCw className="h-4 w-4 animate-spin mr-2" />
                  ) : (
                    <Search className="h-4 w-4 mr-2" />
                  )}
                  搜索
                </Button>
              </div>

              {totalResults > 0 && (
                <p className="text-sm text-muted-foreground mt-2">
                  找到 {totalResults} 个视频，已选择 {selectedBvids.length} 个
                </p>
              )}
            </CardContent>
          </Card>

          {/* 搜索结果表格 */}
          {searchResults.length > 0 && (
            <Card>
              <CardContent className="p-0">
                <ScrollArea className="h-[400px]">
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead className="w-12">
                          <Checkbox checked={selectedBvids.length === searchResults.length} onCheckedChange={toggleAll} />
                        </TableHead>
                        <TableHead>BV号</TableHead>
                        <TableHead>标题</TableHead>
                        <TableHead>时长</TableHead>
                        <TableHead>播放量</TableHead>
                        <TableHead>UP主</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {searchResults.map((video) => (
                        <TableRow key={video.bvid} className="cursor-pointer" onClick={() => toggleBvid(video.bvid)}>
                          <TableCell>
                            <Checkbox checked={selectedBvids.includes(video.bvid)} onCheckedChange={() => toggleBvid(video.bvid)} />
                          </TableCell>
                          <TableCell className="font-mono text-xs">{video.bvid}</TableCell>
                          <TableCell className="max-w-[300px] truncate">{video.title}</TableCell>
                          <TableCell>{video.duration}</TableCell>
                          <TableCell>{(video.views / 1000).toFixed(1)}K</TableCell>
                          <TableCell>{video.author}</TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </ScrollArea>
              </CardContent>
            </Card>
          )}
        </div>

        {/* 右侧：操作面板 */}
        <div className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>操作面板</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <Label>最低分数阈值: {minScore}</Label>
                <Slider value={[minScore]} onValueChange={(v) => setMinScore(v[0])} min={0} max={100} step={5} />
              </div>

              <Separator />

              <Button variant="outline" className="w-full" onClick={handlePrescreen} disabled={prescreening || selectedBvids.length === 0}>
                <BarChart3 className="h-4 w-4 mr-2" />
                预筛选 ({selectedBvids.length} 个视频)
              </Button>

              <Button variant="outline" className="w-full" onClick={() => handleDownload(selectedBvids)} disabled={downloading || selectedBvids.length === 0}>
                <Download className="h-4 w-4 mr-2" />
                {downloading ? `下载中 ${downloadProgress}%` : '下载选中视频'}
              </Button>

              {downloading && <Progress value={downloadProgress} className="h-2" />}

              <Separator />

              <Dialog open={pipelineDialogOpen} onOpenChange={setPipelineDialogOpen}>
                <DialogTrigger asChild>
                  <Button className="w-full" disabled={pipelinePolling}>
                    <Play className="h-4 w-4 mr-2" />
                    运行 SFT 流水线
                  </Button>
                </DialogTrigger>
                <DialogContent>
                  <DialogHeader>
                    <DialogTitle>SFT 流水线配置</DialogTitle>
                    <DialogDescription>配置并运行完整的SFT流水线</DialogDescription>
                  </DialogHeader>
                  <div className="space-y-4 py-4">
                    <div className="space-y-2">
                      <Label>最大视频数量</Label>
                      <Input type="number" value={pipelineConfig.maxVideos} onChange={(e) => setPipelineConfig({ ...pipelineConfig, maxVideos: parseInt(e.target.value) || 10 })} min={1} max={50} />
                    </div>
                    <div className="space-y-2">
                      <Label>VLM 后端</Label>
                      <Select value={pipelineConfig.vlmBackend} onValueChange={(v) => setPipelineConfig({ ...pipelineConfig, vlmBackend: v })}>
                        <SelectTrigger><SelectValue /></SelectTrigger>
                        <SelectContent>
                          <SelectItem value="mock">Mock (测试用)</SelectItem>
                          <SelectItem value="gemini">Google Gemini</SelectItem>
                          <SelectItem value="ollama">Ollama (本地)</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                    <Alert>
                      <Info className="h-4 w-4" />
                      <AlertDescription>
                        当前选择: {selectedBvids.length > 0 ? `${selectedBvids.length} 个已选视频` : `关键词"${keyword}"搜索结果`}
                      </AlertDescription>
                    </Alert>
                  </div>
                  <DialogFooter>
                    <Button variant="outline" onClick={() => setPipelineDialogOpen(false)}>取消</Button>
                    <Button onClick={handleRunPipeline}><Play className="h-4 w-4 mr-2" />开始运行</Button>
                  </DialogFooter>
                </DialogContent>
              </Dialog>
            </CardContent>
          </Card>

          {/* 预筛选结果 */}
          {prescreenResults.length > 0 && (
            <Card>
              <CardHeader><CardTitle>预筛选结果</CardTitle></CardHeader>
              <CardContent>
                <ScrollArea className="h-[300px] space-y-2">
                  {prescreenResults.slice(0, 10).map((result) => (
                    <div key={result.bvid} className="flex items-start gap-2 p-2 rounded-lg border">
                      {getVerdictIcon(result.verdict)}
                      <div className="flex-1 min-w-0">
                        <p className="text-xs font-mono">{result.bvid}</p>
                        <p className="text-sm truncate">{result.title}</p>
                        <div className="flex gap-1 mt-1">
                          <Badge variant={getScoreVariant(result.score)}>分数: {result.score}</Badge>
                          <Badge variant="outline">{result.details.resolution}</Badge>
                        </div>
                      </div>
                    </div>
                  ))}
                </ScrollArea>
              </CardContent>
            </Card>
          )}
        </div>
      </div>

      {/* 流水线任务状态 */}
      {pipelineJob && (
        <Card>
          <CardHeader><CardTitle>SFT 流水线状态</CardTitle></CardHeader>
          <CardContent className="space-y-4">
            <div className="flex items-center gap-2">
              <Badge variant={pipelineJob.status === 'completed' ? 'default' : pipelineJob.status === 'failed' ? 'destructive' : 'secondary'}>
                {pipelineJob.status}
              </Badge>
              <span className="text-sm text-muted-foreground">{pipelineJob.stage}</span>
            </div>
            <Progress value={pipelineJob.progress} className="h-2" />
            
            {/* 详细统计 */}
            {pipelineJob.results && (
              <div className="grid grid-cols-4 gap-4 text-center">
                {pipelineJob.results.search && (
                  <div className="p-2 bg-muted rounded">
                    <div className="text-2xl font-bold">{(pipelineJob.results.search as {found: number}).found}</div>
                    <div className="text-xs text-muted-foreground">搜索结果</div>
                  </div>
                )}
                {pipelineJob.results.prescreen && (
                  <div className="p-2 bg-muted rounded">
                    <div className="text-2xl font-bold">{(pipelineJob.results.prescreen as {passed: number}).passed}</div>
                    <div className="text-xs text-muted-foreground">通过筛选</div>
                  </div>
                )}
                {pipelineJob.results.download && (
                  <div className="p-2 bg-muted rounded">
                    <div className="text-2xl font-bold">{(pipelineJob.results.download as {downloaded: number}).downloaded}</div>
                    <div className="text-xs text-muted-foreground">下载完成</div>
                  </div>
                )}
                {pipelineJob.results.sft && (
                  <div className="p-2 bg-muted rounded">
                    <div className="text-2xl font-bold">{(pipelineJob.results.sft as {episodes_count?: number}).episodes_count || 1}</div>
                    <div className="text-xs text-muted-foreground">Episodes</div>
                  </div>
                )}
              </div>
            )}
            
            {pipelineJob.results?.sft && (
              <Alert>
                <CheckCircle className="h-4 w-4" />
                <AlertTitle>✅ SFT 流水线完成！</AlertTitle>
                <AlertDescription className="space-y-2">
                  <div><strong>输出目录:</strong> <code className="text-xs bg-muted px-1 rounded">{pipelineJob.outputDir || (pipelineJob.results.sft as {output_dir?: string}).output_dir}</code></div>
                  <div><strong>LeRobot 数据:</strong> <code className="text-xs bg-muted px-1 rounded">{(pipelineJob.results.sft as {lerobot_dir?: string}).lerobot_dir}</code></div>
                  <div><strong>训练配置:</strong> <code className="text-xs bg-muted px-1 rounded">{(pipelineJob.results.sft as {config_path?: string}).config_path}</code></div>
                  <div className="mt-2 pt-2 border-t">
                    <span className="text-muted-foreground">可在此目录查看：</span>
                    <ul className="text-xs mt-1 space-y-1">
                      <li>• <code>lerobot/meta/info.json</code> - 数据集元信息</li>
                      <li>• <code>lerobot/data/</code> - 帧数据与动作标注</li>
                      <li>• <code>configs/openpi_finetune.json</code> - π₀.5训练配置</li>
                    </ul>
                  </div>
                </AlertDescription>
              </Alert>
            )}
            {pipelineJob.error && (
              <Alert variant="destructive"><AlertDescription>{pipelineJob.error}</AlertDescription></Alert>
            )}
          </CardContent>
        </Card>
      )}
    </div>
  );
}
