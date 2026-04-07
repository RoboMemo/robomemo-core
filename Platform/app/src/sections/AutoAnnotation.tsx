import { useState, useEffect, useRef } from 'react';
import { 
  Sparkles, 
  MessageSquare, 
  Scissors, 
  FileText, 
  Search,
  Play,
  Pause,
  SkipBack,
  Wand2,
  Layers,
  BrainCircuit,
  Send,
  Loader2,
  Video,
  AlignLeft,
  ListTree,
  Zap
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Label } from '@/components/ui/label';
import { Input } from '@/components/ui/input';
import { Slider } from '@/components/ui/slider';
import { Progress } from '@/components/ui/progress';
import { ScrollArea } from '@/components/ui/scroll-area';
import { api } from '@/services/api';
import type { Dataset, Episode } from '@/types';

interface VideoSegment {
  start: number;
  end: number;
  caption: string;
  confidence: number;
  keyframes?: number[];
}

interface QueryResult {
  answer: string;
  relevantFrames: number[];
  confidence: number;
  visualEvidence: string;
}

interface AutoAnnotationModel {
  id: string;
  name: string;
  provider: string;
  description: string;
  capabilities: string[];
  maxVideoLength: number;
  languageSupport: string[];
}

export default function AutoAnnotation() {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [episodes, setEpisodes] = useState<Episode[]>([]);
  const [selectedDataset, setSelectedDataset] = useState<string>('');
  const [selectedEpisode, setSelectedEpisode] = useState<string>('');
  const [models, setModels] = useState<AutoAnnotationModel[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>('');
  const [activeTab, setActiveTab] = useState('segment');
  
  // Video player state
  const videoRef = useRef<HTMLVideoElement>(null);
  const [currentFrame, setCurrentFrame] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const playbackInterval = useRef<ReturnType<typeof setInterval> | null>(null);
  
  // Segmentation results
  const [segments, setSegments] = useState<VideoSegment[]>([]);
  const [isSegmenting, setIsSegmenting] = useState(false);
  const [selectedSegment, setSelectedSegment] = useState<VideoSegment | null>(null);
  
  // Query state
  const [query, setQuery] = useState('');
  const [queryResult, setQueryResult] = useState<QueryResult | null>(null);
  const [isQuerying, setIsQuerying] = useState(false);
  const [queryHistory, setQueryHistory] = useState<{query: string; result: QueryResult}[]>([]);
  
  // Summary state
  const [summaryType, setSummaryType] = useState('detailed');
  const [summary, setSummary] = useState<any>(null);
  const [isSummarizing, setIsSummarizing] = useState(false);


  
  // Search state
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState<any[]>([]);
  const [isSearching, setIsSearching] = useState(false);
  
  // Batch state
  const [batchProgress, setBatchProgress] = useState(0);
  const [isBatchProcessing, setIsBatchProcessing] = useState(false);

  useEffect(() => {
    loadData();
    return () => {
      if (playbackInterval.current) {
        clearInterval(playbackInterval.current);
      }
    };
  }, []);

  useEffect(() => {
    if (selectedDataset) {
      loadEpisodes(selectedDataset);
    }
  }, [selectedDataset]);

  const loadData = async () => {
    try {
      const [datasetsData, modelsData] = await Promise.all([
        api.getDatasets(),
        api.getAutoAnnotationModels()
      ]);
      setDatasets(datasetsData);
      setModels(modelsData);
      if (modelsData.length > 0) {
        setSelectedModel(modelsData[0].id);
      }
    } catch (error) {
      console.error('Failed to load data:', error);
    }
  };

  const loadEpisodes = async (datasetId: string) => {
    try {
      const data = await api.getDatasetEpisodes(datasetId);
      setEpisodes(data);
    } catch (error) {
      console.error('Failed to load episodes:', error);
    }
  };

  const togglePlayback = () => {
    if (isPlaying) {
      if (playbackInterval.current) {
        clearInterval(playbackInterval.current);
      }
      setIsPlaying(false);
    } else {
      setIsPlaying(true);
      playbackInterval.current = setInterval(() => {
        setCurrentFrame(prev => (prev + 1) % 100);
      }, 100);
    }
  };

  const handleSegment = async () => {
    if (!selectedEpisode || !selectedModel) return;

    setIsSegmenting(true);
    try {
      let result;
      if (selectedModel === 'local-smolvlm') {
        result = await api.segmentVideoLocal(`episode_${selectedEpisode}.mp4`);
      } else {
        result = await api.segmentVideo(
          `episode_${selectedEpisode}.mp4`,
          selectedModel,
          { numSegments: 5 }
        );
      }
      setSegments(result);
    } catch (error) {
      console.error('Failed to segment video:', error);
    } finally {
      setIsSegmenting(false);
    }
  };

  const handleQuery = async () => {
    if (!query.trim() || !selectedEpisode || !selectedModel) return;

    setIsQuerying(true);
    try {
      let result;
      if (selectedModel === 'local-smolvlm') {
        result = await api.queryVideoLocal(`episode_${selectedEpisode}.mp4`, query);
      } else {
        result = await api.queryVideo(
          `episode_${selectedEpisode}.mp4`,
          query,
          selectedModel
        );
      }
      setQueryResult(result);
      setQueryHistory(prev => [...prev, { query, result }]);
      setQuery('');
    } catch (error) {
      console.error('Failed to query video:', error);
    } finally {
      setIsQuerying(false);
    }
  };

  const handleSummarize = async () => {
    if (!selectedEpisode || !selectedModel) return;
    
    setIsSummarizing(true);
    try {
      const result = await api.summarizeVideo(
        `episode_${selectedEpisode}.mp4`,
        selectedModel,
        summaryType
      );
      setSummary(result);
    } catch (error) {
      console.error('Failed to summarize video:', error);
    } finally {
      setIsSummarizing(false);
    }
  };

  const handleSearch = async () => {
    if (!searchQuery.trim() || !selectedDataset || !selectedModel) return;
    
    setIsSearching(true);
    try {
      const result = await api.searchVideos(
        selectedDataset,
        searchQuery,
        selectedModel,
        5
      );
      setSearchResults(result.results);
    } catch (error) {
      console.error('Failed to search videos:', error);
    } finally {
      setIsSearching(false);
    }
  };

  const handleBatchAnnotate = async () => {
    if (!selectedDataset || !selectedModel) return;
    
    setIsBatchProcessing(true);
    setBatchProgress(0);
    
    try {
      const episodeIds = episodes.map(e => e.id);
      await api.batchAutoAnnotate(episodeIds, selectedModel, 'temporal_segmentation');
      
      // Simulate progress
      const interval = setInterval(() => {
        setBatchProgress(prev => {
          if (prev >= 100) {
            clearInterval(interval);
            setIsBatchProcessing(false);
            return 100;
          }
          return prev + 10;
        });
      }, 500);
    } catch (error) {
      console.error('Failed to batch annotate:', error);
      setIsBatchProcessing(false);
    }
  };

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  const selectedModelData = models.find(m => m.id === selectedModel);

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold flex items-center gap-2">
          <Sparkles className="w-6 h-6 text-primary" />
          Auto-Annotation with VLM
        </h2>
        <p className="text-muted-foreground">
          Use Vision-Language Models to automatically segment, caption, and query your videos
        </p>
      </div>

      {/* Configuration */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg flex items-center gap-2">
            <BrainCircuit className="w-5 h-5" />
            VLM Configuration
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="space-y-2">
              <Label>VLM Model</Label>
              <Select value={selectedModel} onValueChange={setSelectedModel}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {models.map(model => (
                    <SelectItem key={model.id} value={model.id}>
                      {model.name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-2">
              <Label>Dataset</Label>
              <Select value={selectedDataset} onValueChange={setSelectedDataset}>
                <SelectTrigger>
                  <SelectValue placeholder="Select dataset" />
                </SelectTrigger>
                <SelectContent>
                  {datasets.map(ds => (
                    <SelectItem key={ds.id} value={ds.id}>{ds.name}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-2">
              <Label>Episode</Label>
              <Select value={selectedEpisode} onValueChange={setSelectedEpisode}>
                <SelectTrigger>
                  <SelectValue placeholder="Select episode" />
                </SelectTrigger>
                <SelectContent>
                  {episodes.map(ep => (
                    <SelectItem key={ep.id} value={ep.id}>{ep.name || ep.id}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </div>

          {selectedModelData && (
            <div className="p-4 bg-muted rounded-lg">
              <div className="flex items-start gap-4">
                <div className="w-12 h-12 rounded-lg bg-primary/10 flex items-center justify-center">
                  <BrainCircuit className="w-6 h-6 text-primary" />
                </div>
                <div className="flex-1">
                  <h4 className="font-medium">{selectedModelData.name}</h4>
                  <p className="text-sm text-muted-foreground">{selectedModelData.description}</p>
                  <div className="flex flex-wrap gap-2 mt-2">
                    {selectedModelData.capabilities.map(cap => (
                      <Badge key={cap} variant="secondary" className="text-xs">
                        {cap.replace(/_/g, ' ')}
                      </Badge>
                    ))}
                  </div>
                  <p className="text-xs text-muted-foreground mt-2">
                    Max video length: {selectedModelData.maxVideoLength}s | 
                    Languages: {selectedModelData.languageSupport.join(', ')}
                  </p>
                </div>
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Main Content */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Episode Data Preview */}
        <Card className="w-full">
          <CardHeader>
            <CardTitle className="flex items-center"><Video className="mr-2" /> Episode Data Preview</CardTitle>
          </CardHeader>
          <CardContent>
            {!selectedEpisode ? (
              <div className="aspect-video bg-muted rounded-lg flex items-center justify-center text-muted-foreground">
                <div className="text-center">
                  <Video className="w-12 h-12 mx-auto mb-2 opacity-50" />
                  <p className="text-sm">Select a dataset and episode to preview</p>
                </div>
              </div>
            ) : (
              <div className="space-y-4">
                {/* Episode Info */}
                {(() => {
                  const ep = episodes.find(e => e.id === selectedEpisode);
                  const ds = datasets.find(d => d.id === selectedDataset);
                  if (!ep) return null;
                  
                  const sensorConfig = ds?.sensorConfig;
                  const sensors = sensorConfig?.sensors || [];
                  const rgbdSensors = sensors.filter((s: any) => s.type === 'rgbd' || s.type === 'camera' || s.type === 'rgb');
                  const ftSensors = sensors.filter((s: any) => s.type === 'force_torque');
                  const otherSensors = sensors.filter((s: any) => !['rgbd', 'camera', 'rgb', 'force_torque'].includes(s.type));
                  
                  return (
                    <>
                      {/* Episode header */}
                      <div className="p-3 rounded-lg bg-muted/50 border">
                        <div className="flex items-center justify-between mb-2">
                          <span className="font-semibold text-sm">{ep.name || ep.id}</span>
                          <Badge variant="secondary" className="text-xs">{(ep as any).skill || 'general'}</Badge>
                        </div>
                        {(ep as any).description && (
                          <p className="text-xs text-muted-foreground mb-2">{(ep as any).description}</p>
                        )}
                        <div className="grid grid-cols-3 gap-2 text-xs">
                          <div className="text-center p-1.5 rounded bg-background border">
                            <div className="font-mono font-bold text-primary">{ep.frameCount}</div>
                            <div className="text-muted-foreground">Frames</div>
                          </div>
                          <div className="text-center p-1.5 rounded bg-background border">
                            <div className="font-mono font-bold text-primary">{ep.duration?.toFixed(1)}s</div>
                            <div className="text-muted-foreground">Duration</div>
                          </div>
                          <div className="text-center p-1.5 rounded bg-background border">
                            <div className="font-mono font-bold text-primary">{(ep as any).fps || 30} fps</div>
                            <div className="text-muted-foreground">Frame Rate</div>
                          </div>
                        </div>
                      </div>

                      {/* RGBD Camera Views */}
                      {rgbdSensors.length > 0 && (
                        <div>
                          <h4 className="text-sm font-semibold mb-2 flex items-center gap-1.5">
                            📷 Camera Views ({rgbdSensors.length}x RGBD)
                          </h4>
                          <div className="grid gap-2" style={{ gridTemplateColumns: `repeat(${Math.min(rgbdSensors.length, 3)}, 1fr)` }}>
                            {rgbdSensors.map((sensor: any, i: number) => (
                              <div key={i} className="rounded-lg border overflow-hidden">
                                <div className="aspect-video bg-gradient-to-br from-slate-800 to-slate-900 flex items-center justify-center relative">
                                  {/* Simulated RGBD view */}
                                  <div className="text-center">
                                    <div className="text-2xl mb-1">
                                      {sensor.location === 'chest' || sensor.mount === 'chest' ? '🎥' : 
                                       sensor.location?.includes('left') || sensor.mount?.includes('left') ? '🤛' : '🤜'}
                                    </div>
                                    <div className="text-xs text-slate-400">{sensor.name}</div>
                                    <div className="text-[10px] text-slate-500 mt-0.5">640×480 RGB-D</div>
                                  </div>
                                  {/* RGB/Depth toggle indicators */}
                                  <div className="absolute top-1 right-1 flex gap-0.5">
                                    <span className="text-[9px] px-1 py-0.5 rounded bg-green-500/20 text-green-400 border border-green-500/30">RGB</span>
                                    <span className="text-[9px] px-1 py-0.5 rounded bg-blue-500/20 text-blue-400 border border-blue-500/30">Depth</span>
                                  </div>
                                </div>
                                <div className="px-2 py-1.5 bg-muted/50 text-xs">
                                  <span className="font-medium">{sensor.name}</span>
                                  <span className="text-muted-foreground ml-1">@ {sensor.location || sensor.mount}</span>
                                </div>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}

                      {/* Force/Torque Sensors */}
                      {ftSensors.length > 0 && (
                        <div>
                          <h4 className="text-sm font-semibold mb-2 flex items-center gap-1.5">
                            ⚡ Force/Torque Sensors ({ftSensors.length}x 6-axis)
                          </h4>
                          <div className="grid grid-cols-2 gap-2">
                            {ftSensors.map((sensor: any, i: number) => (
                              <div key={i} className="p-2.5 rounded-lg border bg-orange-50 dark:bg-orange-950 border-orange-200 dark:border-orange-800">
                                <div className="flex items-center gap-2 mb-1.5">
                                  <Zap className="w-3.5 h-3.5 text-orange-500" />
                                  <span className="font-medium text-xs">{sensor.name}</span>
                                </div>
                                <div className="grid grid-cols-2 gap-1 text-[10px]">
                                  <div className="text-center p-1 rounded bg-background/60">
                                    <div className="font-mono text-orange-600">Fx Fy Fz</div>
                                    <div className="text-muted-foreground">Force (N)</div>
                                  </div>
                                  <div className="text-center p-1 rounded bg-background/60">
                                    <div className="font-mono text-orange-600">Tx Ty Tz</div>
                                    <div className="text-muted-foreground">Torque (Nm)</div>
                                  </div>
                                </div>
                                <div className="text-[10px] text-muted-foreground mt-1">
                                  @ {sensor.location || sensor.mount} · 1000 Hz
                                </div>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}

                      {/* Other Sensors */}
                      {otherSensors.length > 0 && (
                        <div>
                          <h4 className="text-sm font-semibold mb-2">Other Sensors</h4>
                          <div className="flex flex-wrap gap-2">
                            {otherSensors.map((sensor: any, i: number) => (
                              <Badge key={i} variant="outline" className="text-xs">
                                {sensor.name} ({sensor.type}) @ {sensor.location || sensor.mount}
                              </Badge>
                            ))}
                          </div>
                        </div>
                      )}

                      {/* Frame Scrubber */}
                      <div className="space-y-2">
                        <div className="flex items-center justify-between text-xs text-muted-foreground">
                          <span>Frame {currentFrame} / {ep.frameCount}</span>
                          <span>{(currentFrame / ((ep as any).fps || 30)).toFixed(2)}s</span>
                        </div>
                        <div className="flex items-center gap-2">
                          <Button size="icon" variant="outline" className="h-7 w-7" onClick={() => setCurrentFrame(0)}>
                            <SkipBack className="h-3 w-3" />
                          </Button>
                          <Button size="icon" variant="outline" className="h-7 w-7" onClick={togglePlayback}>
                            {isPlaying ? <Pause className="h-3 w-3" /> : <Play className="h-3 w-3" />}
                          </Button>
                          <div className="flex-1">
                            <Slider
                              value={[currentFrame]}
                              max={ep.frameCount}
                              onValueChange={(value) => setCurrentFrame(value[0])}
                            />
                          </div>
                        </div>
                      </div>

                      {/* Sensor Config Name */}
                      {sensorConfig?.name && (
                        <div className="text-xs text-muted-foreground flex items-center gap-1.5 p-2 rounded bg-muted/30 border">
                          <Layers className="w-3 h-3" />
                          <span>Sensor Config: <strong>{sensorConfig.name}</strong> ({sensors.length} sensors)</span>
                        </div>
                      )}
                    </>
                  );
                })()}
              </div>
            )}
          </CardContent>
        </Card>

        {/* Function Tabs */}
        <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-4">
          <TabsList className="grid grid-cols-4">
            <TabsTrigger value="segment">
              <Scissors className="w-4 h-4 mr-2" />
              Segment
            </TabsTrigger>
            <TabsTrigger value="query">
              <MessageSquare className="w-4 h-4 mr-2" />
              Query
            </TabsTrigger>
            <TabsTrigger value="summary">
              <FileText className="w-4 h-4 mr-2" />
              Summary
            </TabsTrigger>
            <TabsTrigger value="search">
              <Search className="w-4 h-4 mr-2" />
              Search
            </TabsTrigger>
          </TabsList>

          {/* Segment Tab */}
          <TabsContent value="segment" className="space-y-4">
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm">Temporal Segmentation</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-muted-foreground mb-4">
                  Automatically segment video into meaningful actions with captions
                </p>
                <Button 
                  onClick={handleSegment} 
                  disabled={!selectedEpisode || isSegmenting}
                  className="w-full"
                >
                  {isSegmenting ? (
                    <><Loader2 className="w-4 h-4 mr-2 animate-spin" /> Segmenting...</>
                  ) : (
                    <><Scissors className="w-4 h-4 mr-2" /> Auto Segment</>
                  )}
                </Button>
              </CardContent>
            </Card>

            {segments.length > 0 && (
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm flex items-center gap-2">
                    <ListTree className="w-4 h-4" />
                    Segments ({segments.length})
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <ScrollArea className="h-64">
                    <div className="space-y-2">
                      {segments.map((segment, i) => (
                        <div 
                          key={i}
                          onClick={() => {
                            setSelectedSegment(segment);
                            if (videoRef.current) {
                              videoRef.current.currentTime = segment.start;
                              setCurrentFrame(Math.floor(segment.start * 30));
                            }
                          }}
                          className={`p-3 rounded-lg border cursor-pointer transition-colors ${
                            selectedSegment === segment 
                              ? 'border-primary bg-primary/5' 
                              : 'hover:bg-accent'
                          }`}
                        >
                          <div className="flex items-center justify-between">
                            <span className="font-medium text-sm">{formatTime(segment.start)} - {formatTime(segment.end)}</span>
                            <Badge variant="secondary" className="text-xs">
                              {(segment.confidence * 100).toFixed(0)}%
                            </Badge>
                          </div>
                          <p className="text-sm text-muted-foreground mt-1">{segment.caption}</p>
                        </div>
                      ))}
                    </div>
                  </ScrollArea>
                </CardContent>
              </Card>
            )}
          </TabsContent>

          {/* Query Tab */}
          <TabsContent value="query" className="space-y-4">
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm">Natural Language Query</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-muted-foreground mb-4">
                  Ask questions about the video in natural language
                </p>
                <div className="flex gap-2">
                  <Input 
                    placeholder="e.g., When does the robot grasp the object?"
                    value={query}
                    onChange={e => setQuery(e.target.value)}
                    onKeyDown={e => e.key === 'Enter' && handleQuery()}
                  />
                  <Button onClick={handleQuery} disabled={isQuerying || !query.trim()}>
                    {isQuerying ? <Loader2 className="w-4 h-4 animate-spin" /> : <Send className="w-4 h-4" />}
                  </Button>
                </div>
                <div className="flex flex-wrap gap-2 mt-2">
                  {['When does grasping happen?', 'What objects are visible?', 'Describe the trajectory'].map(q => (
                    <button
                      key={q}
                      onClick={() => setQuery(q)}
                      className="text-xs px-2 py-1 bg-muted rounded-full hover:bg-accent transition-colors"
                    >
                      {q}
                    </button>
                  ))}
                </div>
              </CardContent>
            </Card>

            {queryResult && (
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm">Answer</CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-sm">{queryResult.answer}</p>
                  <div className="flex items-center gap-4 mt-3 text-xs text-muted-foreground">
                    <span>Confidence: {(queryResult.confidence * 100).toFixed(0)}%</span>
                    <span>Frames: {queryResult.relevantFrames.map(f => Math.floor(f / 30)).join(', ')}s</span>
                  </div>
                  <p className="text-xs text-muted-foreground mt-1">
                    Evidence: {queryResult.visualEvidence}
                  </p>
                </CardContent>
              </Card>
            )}

            {queryHistory.length > 0 && (
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm">Query History</CardTitle>
                </CardHeader>
                <CardContent>
                  <ScrollArea className="h-40">
                    <div className="space-y-2">
                      {queryHistory.map((item, i) => (
                        <div key={i} className="p-2 bg-muted rounded text-sm">
                          <p className="font-medium">Q: {item.query}</p>
                          <p className="text-muted-foreground">A: {item.result.answer}</p>
                        </div>
                      ))}
                    </div>
                  </ScrollArea>
                </CardContent>
              </Card>
            )}
          </TabsContent>

          {/* Summary Tab */}
          <TabsContent value="summary" className="space-y-4">
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm">Video Summary</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <Select value={summaryType} onValueChange={setSummaryType}>
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="brief">Brief (one sentence)</SelectItem>
                      <SelectItem value="detailed">Detailed (paragraph)</SelectItem>
                      <SelectItem value="structured">Structured (JSON)</SelectItem>
                      <SelectItem value="instructional">Instructional (steps)</SelectItem>
                    </SelectContent>
                  </Select>
                  <Button 
                    onClick={handleSummarize} 
                    disabled={!selectedEpisode || isSummarizing}
                    className="w-full"
                  >
                    {isSummarizing ? (
                      <><Loader2 className="w-4 h-4 mr-2 animate-spin" /> Generating...</>
                    ) : (
                      <><FileText className="w-4 h-4 mr-2" /> Generate Summary</>
                    )}
                  </Button>
                </div>
              </CardContent>
            </Card>

            {summary && (
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm flex items-center gap-2">
                    <AlignLeft className="w-4 h-4" />
                    Result
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  {typeof summary.summary === 'string' ? (
                    <p className="text-sm">{summary.summary}</p>
                  ) : (
                    <pre className="text-xs bg-muted p-3 rounded overflow-auto">
                      {JSON.stringify(summary.summary, null, 2)}
                    </pre>
                  )}
                  <p className="text-xs text-muted-foreground mt-2">
                    Processing time: {summary.processingTime}s
                  </p>
                </CardContent>
              </Card>
            )}
          </TabsContent>

          {/* Search Tab */}
          <TabsContent value="search" className="space-y-4">
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm">Semantic Video Search</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-muted-foreground mb-4">
                  Search across all episodes using natural language
                </p>
                <div className="flex gap-2">
                  <Input 
                    placeholder="e.g., episodes where robot picks up red objects"
                    value={searchQuery}
                    onChange={e => setSearchQuery(e.target.value)}
                    onKeyDown={e => e.key === 'Enter' && handleSearch()}
                  />
                  <Button onClick={handleSearch} disabled={isSearching || !searchQuery.trim()}>
                    {isSearching ? <Loader2 className="w-4 h-4 animate-spin" /> : <Search className="w-4 h-4" />}
                  </Button>
                </div>
              </CardContent>
            </Card>

            {searchResults.length > 0 && (
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm">Results ({searchResults.length})</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-2">
                    {searchResults.map((result, i) => (
                      <div key={i} className="p-3 border rounded-lg">
                        <div className="flex items-center justify-between">
                          <span className="font-medium text-sm">{result.episodeId}</span>
                          <Badge variant="secondary">{(result.score * 100).toFixed(0)}% match</Badge>
                        </div>
                        <p className="text-sm text-muted-foreground mt-1">{result.caption}</p>
                        <div className="flex gap-2 mt-2">
                          {result.matchedSegments.map((seg: any, j: number) => (
                            <Badge key={j} variant="outline" className="text-xs">
                              {formatTime(seg.start)}-{formatTime(seg.end)}
                            </Badge>
                          ))}
                        </div>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            )}
          </TabsContent>
        </Tabs>
      </div>

      {/* Batch Processing */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg flex items-center gap-2">
            <Layers className="w-5 h-5" />
            Batch Auto-Annotation
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-muted-foreground">
                Automatically annotate all episodes in the selected dataset
              </p>
              <p className="text-sm">
                {selectedDataset ? `${episodes.length} episodes ready` : 'Select a dataset first'}
              </p>
            </div>
            <Button 
              onClick={handleBatchAnnotate}
              disabled={!selectedDataset || isBatchProcessing}
            >
              {isBatchProcessing ? (
                <><Loader2 className="w-4 h-4 mr-2 animate-spin" /> Processing...</>
              ) : (
                <><Wand2 className="w-4 h-4 mr-2" /> Batch Annotate</>
              )}
            </Button>
          </div>
          {isBatchProcessing && (
            <div className="mt-4">
              <Progress value={batchProgress} className="h-2" />
              <p className="text-sm text-muted-foreground mt-2 text-center">
                {batchProgress}% complete
              </p>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
