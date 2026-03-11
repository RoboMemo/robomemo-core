import { useState, useEffect, useRef } from 'react';
import { 
  MousePointer2, 
  Square, 
  Circle, 
  Pencil,
  ChevronLeft,
  ChevronRight,
  Play,
  Pause,
  SkipBack,
  Check,
  X,
  Download,
  FolderOpen,
  Edit
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog';
import { Label } from '@/components/ui/label';
import { api } from '@/services/api';
import type { Dataset, Episode, Frame, Annotation } from '@/types';

type AnnotationTool = 'select' | 'bbox' | 'polygon' | 'keypoint';
type AnnotationLabel = 'object' | 'gripper' | 'target' | 'obstacle' | 'custom';

interface DrawingState {
  isDrawing: boolean;
  startX: number;
  startY: number;
  currentX: number;
  currentY: number;
}

export default function DataAnnotation() {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [selectedDataset, setSelectedDataset] = useState<string>('');
  const [episodes, setEpisodes] = useState<Episode[]>([]);
  const [selectedEpisode, setSelectedEpisode] = useState<string>('');
  const [frames, setFrames] = useState<Frame[]>([]);
  const [currentFrameIndex, setCurrentFrameIndex] = useState(0);
  const [annotations, setAnnotations] = useState<Annotation[]>([]);
  const [activeTool, setActiveTool] = useState<AnnotationTool>('select');
  const [activeLabel, setActiveLabel] = useState<AnnotationLabel>('object');
  const [isPlaying, setIsPlaying] = useState(false);
  const [drawing, setDrawing] = useState<DrawingState | null>(null);
  const [isEditingFrame, setIsEditingFrame] = useState(false);
  const [frameComment, setFrameComment] = useState<string>('');
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const playbackInterval = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    loadDatasets();
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

  useEffect(() => {
    if (selectedEpisode) {
      loadFrames(selectedEpisode);
    }
  }, [selectedEpisode]);

  useEffect(() => {
    if (frames[currentFrameIndex]) {
      loadAnnotations(frames[currentFrameIndex].id);
    }
  }, [frames, currentFrameIndex]);

  const loadDatasets = async () => {
    try {
      const data = await api.getDatasets();
      setDatasets(data);
    } catch (error) {
      console.error('Failed to load datasets:', error);
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

  const loadFrames = async (episodeId: string) => {
    const mockFrames: Frame[] = Array.from({ length: 100 }, (_, i) => ({
      id: `frame_${episodeId}_${i}`,
      episodeId,
      datasetId: selectedDataset,
      frameIndex: i,
      timestamp: new Date().toISOString(),
      observations: {}
    }));
    setFrames(mockFrames);
    setCurrentFrameIndex(0);
  };

  const loadAnnotations = async (frameId: string) => {
    try {
      const data = await api.getAnnotationsByFrame(frameId);
      setAnnotations(data);
    } catch (error) {
      console.error('Failed to load annotations:', error);
    }
  };

  const handlePrevFrame = () => {
    setCurrentFrameIndex(prev => Math.max(0, prev - 1));
  };

  const handleNextFrame = () => {
    setCurrentFrameIndex(prev => Math.min(frames.length - 1, prev + 1));
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
        setCurrentFrameIndex(prev => {
          if (prev >= frames.length - 1) {
            if (playbackInterval.current) {
              clearInterval(playbackInterval.current);
            }
            setIsPlaying(false);
            return prev;
          }
          return prev + 1;
        });
      }, 100);
    }
  };

  const handleCanvasMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (activeTool === 'select') return;
    
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    setDrawing({
      isDrawing: true,
      startX: x,
      startY: y,
      currentX: x,
      currentY: y
    });
  };

  const handleCanvasMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!drawing?.isDrawing) return;
    
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    setDrawing(prev => prev ? { ...prev, currentX: x, currentY: y } : null);
    
    drawTempAnnotation();
  };

  const handleCanvasMouseUp = async () => {
    if (!drawing?.isDrawing) return;
    
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const newAnnotation: Partial<Annotation> = {
      frameId: frames[currentFrameIndex]?.id,
      episodeId: selectedEpisode,
      datasetId: selectedDataset,
      type: activeTool === 'bbox' ? 'bbox' : 'mask',
      label: activeLabel,
      bbox: activeTool === 'bbox' ? {
        x: Math.min(drawing.startX, drawing.currentX),
        y: Math.min(drawing.startY, drawing.currentY),
        width: Math.abs(drawing.currentX - drawing.startX),
        height: Math.abs(drawing.currentY - drawing.startY)
      } : undefined,
      confidence: 1.0,
      annotator: 'manual'
    };
    
    try {
      const saved = await api.createAnnotation(newAnnotation);
      setAnnotations(prev => [...prev, saved]);
    } catch (error) {
      console.error('Failed to save annotation:', error);
    }
    
    setDrawing(null);
    drawAnnotations();
  };

  const drawTempAnnotation = () => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext('2d');
    if (!canvas || !ctx || !drawing) return;
    
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    drawAnnotations();
    
    ctx.strokeStyle = '#3b82f6';
    ctx.lineWidth = 2;
    ctx.setLineDash([5, 5]);
    
    if (activeTool === 'bbox') {
      const x = Math.min(drawing.startX, drawing.currentX);
      const y = Math.min(drawing.startY, drawing.currentY);
      const width = Math.abs(drawing.currentX - drawing.startX);
      const height = Math.abs(drawing.currentY - drawing.startY);
      ctx.strokeRect(x, y, width, height);
    }
    
    ctx.setLineDash([]);
  };

  const drawAnnotations = () => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext('2d');
    if (!canvas || !ctx) return;
    
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    annotations.forEach(ann => {
      if (ann.type === 'bbox' && ann.bbox) {
        ctx.strokeStyle = ann.verified ? '#22c55e' : '#ef4444';
        ctx.lineWidth = 2;
        ctx.strokeRect(ann.bbox.x, ann.bbox.y, ann.bbox.width, ann.bbox.height);
        
        ctx.fillStyle = ann.verified ? '#22c55e' : '#ef4444';
        ctx.font = '12px sans-serif';
        ctx.fillText(ann.label, ann.bbox.x, ann.bbox.y - 5);
      }
    });
  };

  const deleteAnnotation = async (id: string) => {
    try {
      await api.deleteAnnotation(id);
      setAnnotations(prev => prev.filter(a => a.id !== id));
    } catch (error) {
      console.error('Failed to delete annotation:', error);
    }
  };

  const verifyAnnotation = async (id: string) => {
    try {
      const updated = await api.updateAnnotation(id, { verified: true, verifier: 'user' });
      setAnnotations(prev => prev.map(a => a.id === id ? updated : a));
    } catch (error) {
      console.error('Failed to verify annotation:', error);
    }
  };

  const handleOpenFrame = () => {
    console.log('Opening frame:', currentFrameIndex);
    alert(`Frame ${currentFrameIndex + 1} opened for detailed view`);
  };

  const handleSaveFrameComment = async () => {
    try {
      const currentFrame = frames[currentFrameIndex];
      if (currentFrame) {
        // Frame comment saved locally (no updateFrame endpoint)
        console.log('Frame comment saved for', currentFrame.id, frameComment);
        setIsEditingFrame(false);
        console.log('Frame comment saved');
      }
    } catch (error) {
      console.error('Failed to save frame comment:', error);
    }
  };

  const handleExportAnnotations = async () => {
    try {
      const exportData = {
        frameIndex: currentFrameIndex,
        episodeId: selectedEpisode,
        datasetId: selectedDataset,
        annotations: annotations,
        exportedAt: new Date().toISOString()
      };
      
      const dataStr = JSON.stringify(exportData, null, 2);
      const dataBlob = new Blob([dataStr], { type: 'application/json' });
      const url = URL.createObjectURL(dataBlob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `annotations-frame-${currentFrameIndex}.json`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
    } catch (error) {
      console.error('Failed to export annotations:', error);
    }
  };

  useEffect(() => {
    drawAnnotations();
  }, [annotations]);

  return (
    <div className="space-y-4">
      <div>
        <h2 className="text-2xl font-bold">Data Annotation</h2>
        <p className="text-muted-foreground">Annotate your embodied intelligence data</p>
      </div>

      {/* Controls */}
      <div className="flex flex-wrap items-center gap-4">
        <Select value={selectedDataset} onValueChange={setSelectedDataset}>
          <SelectTrigger className="w-48">
            <SelectValue placeholder="Select dataset" />
          </SelectTrigger>
          <SelectContent>
            {datasets.map(ds => (
              <SelectItem key={ds.id} value={ds.id}>{ds.name}</SelectItem>
            ))}
          </SelectContent>
        </Select>

        <Select value={selectedEpisode} onValueChange={setSelectedEpisode}>
          <SelectTrigger className="w-48">
            <SelectValue placeholder="Select episode" />
          </SelectTrigger>
          <SelectContent>
            {episodes.map(ep => (
              <SelectItem key={ep.id} value={ep.id}>{ep.name || ep.id}</SelectItem>
            ))}
          </SelectContent>
        </Select>

        <div className="flex items-center gap-2 ml-auto">
          <Button variant="outline" size="icon" onClick={handleOpenFrame} title="Open frame">
            <FolderOpen className="w-4 h-4" />
          </Button>
          <Dialog open={isEditingFrame} onOpenChange={setIsEditingFrame}>
            <DialogTrigger asChild>
              <Button variant="outline" size="icon" title="Edit frame">
                <Edit className="w-4 h-4" />
              </Button>
            </DialogTrigger>
            <DialogContent className="max-w-md">
              <DialogHeader>
                <DialogTitle>Edit Frame {currentFrameIndex + 1}</DialogTitle>
              </DialogHeader>
              <div className="space-y-4 pt-4">
                <div className="space-y-2">
                  <Label>Frame Comment</Label>
                  <textarea 
                    placeholder="Add notes about this frame..."
                    value={frameComment}
                    onChange={(e) => setFrameComment(e.target.value)}
                    className="w-full px-3 py-2 border border-input rounded-md text-sm h-24"
                  />
                </div>
                <div className="flex gap-2">
                  <Button onClick={handleSaveFrameComment} className="flex-1">
                    Save Comment
                  </Button>
                  <Button onClick={() => setIsEditingFrame(false)} variant="outline" className="flex-1">
                    Cancel
                  </Button>
                </div>
              </div>
            </DialogContent>
          </Dialog>
          <Button variant="outline" size="icon" onClick={handleExportAnnotations} title="Export annotations">
            <Download className="w-4 h-4" />
          </Button>
          <Button variant="outline" size="icon" onClick={() => setCurrentFrameIndex(0)}>
            <SkipBack className="w-4 h-4" />
          </Button>
          <Button variant="outline" size="icon" onClick={handlePrevFrame}>
            <ChevronLeft className="w-4 h-4" />
          </Button>
          <Button variant="outline" size="icon" onClick={togglePlayback}>
            {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
          </Button>
          <Button variant="outline" size="icon" onClick={handleNextFrame}>
            <ChevronRight className="w-4 h-4" />
          </Button>
        </div>

        <div className="text-sm text-muted-foreground">
          Frame {currentFrameIndex + 1} / {frames.length}
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-4">
        {/* Toolbar */}
        <Card className="lg:col-span-1">
          <CardContent className="p-4 space-y-4">
            <div>
              <h4 className="text-sm font-medium mb-2">Tools</h4>
              <div className="grid grid-cols-2 gap-2">
                {[
                  { id: 'select', icon: MousePointer2, label: 'Select' },
                  { id: 'bbox', icon: Square, label: 'BBox' },
                  { id: 'polygon', icon: Pencil, label: 'Polygon' },
                  { id: 'keypoint', icon: Circle, label: 'Keypoint' },
                ].map(tool => {
                  const Icon = tool.icon;
                  return (
                    <button
                      key={tool.id}
                      onClick={() => setActiveTool(tool.id as AnnotationTool)}
                      className={`flex items-center gap-2 px-3 py-2 rounded-lg text-sm transition-colors ${
                        activeTool === tool.id
                          ? 'bg-primary text-primary-foreground'
                          : 'hover:bg-accent'
                      }`}
                    >
                      <Icon className="w-4 h-4" />
                      {tool.label}
                    </button>
                  );
                })}
              </div>
            </div>

            <div>
              <h4 className="text-sm font-medium mb-2">Labels</h4>
              <Select value={activeLabel} onValueChange={v => setActiveLabel(v as AnnotationLabel)}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="object">Object</SelectItem>
                  <SelectItem value="gripper">Gripper</SelectItem>
                  <SelectItem value="target">Target</SelectItem>
                  <SelectItem value="obstacle">Obstacle</SelectItem>
                  <SelectItem value="custom">Custom</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div>
              <h4 className="text-sm font-medium mb-2">Annotations</h4>
              <div className="space-y-2 max-h-64 overflow-auto">
                {annotations.map(ann => (
                  <div 
                    key={ann.id} 
                    className="flex items-center justify-between p-2 rounded-lg border text-sm"
                  >
                    <div className="flex items-center gap-2">
                      <Badge variant={ann.verified ? 'default' : 'secondary'} className="text-xs">
                        {ann.type}
                      </Badge>
                      <span>{ann.label}</span>
                    </div>
                    <div className="flex items-center gap-1">
                      {!ann.verified && (
                        <Button 
                          variant="ghost" 
                          size="icon" 
                          className="w-6 h-6"
                          onClick={() => verifyAnnotation(ann.id)}
                        >
                          <Check className="w-3 h-3" />
                        </Button>
                      )}
                      <Button 
                        variant="ghost" 
                        size="icon" 
                        className="w-6 h-6 text-destructive"
                        onClick={() => deleteAnnotation(ann.id)}
                      >
                        <X className="w-3 h-3" />
                      </Button>
                    </div>
                  </div>
                ))}
                {annotations.length === 0 && (
                  <p className="text-sm text-muted-foreground text-center py-4">
                    No annotations yet
                  </p>
                )}
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Canvas */}
        <Card className="lg:col-span-3">
          <CardContent className="p-4">
            <div className="relative aspect-video bg-muted rounded-lg overflow-hidden">
              <div className="absolute inset-0 flex items-center justify-center">
                <img 
                  src={`https://picsum.photos/seed/frame${currentFrameIndex}/640/480`}
                  alt="Frame"
                  className="w-full h-full object-contain"
                />
              </div>
              
              <canvas
                ref={canvasRef}
                width={640}
                height={480}
                className="absolute inset-0 w-full h-full cursor-crosshair"
                onMouseDown={handleCanvasMouseDown}
                onMouseMove={handleCanvasMouseMove}
                onMouseUp={handleCanvasMouseUp}
                onMouseLeave={handleCanvasMouseUp}
              />
              
              <div className="absolute top-4 left-4 bg-black/50 text-white px-3 py-1 rounded text-sm">
                Frame {currentFrameIndex + 1}
              </div>
            </div>

            <div className="mt-4">
              <input
                type="range"
                min={0}
                max={frames.length - 1}
                value={currentFrameIndex}
                onChange={e => setCurrentFrameIndex(parseInt(e.target.value))}
                className="w-full"
              />
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
