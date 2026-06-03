import { useState, useEffect } from 'react';
import { 
  Plus, 
  Search, 
  MoreVertical, 
  FolderOpen, 
  Trash2, 
  Edit, 
  Download,
  FileJson,
  Database,
  Layers,
  RefreshCw,
  Eye,
  BarChart3,
  Zap
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog';
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from '@/components/ui/dropdown-menu';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Textarea } from '@/components/ui/textarea';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { api } from '@/services/api';
import type { Dataset } from '@/types';

// ── GenRobot interfaces ──────────────────────────────────────────────────────

interface Sample {
  id: string;
  episode_id: string;
  frames: number;
  task: string;
  robot_type: string;
  actions_dim: number;
  observations: Record<string, unknown>;
}

interface GenRobotMetadata {
  name: string;
  source: string;
  description: string;
  samples: Sample[];
}

// ── Constants ────────────────────────────────────────────────────────────────

const formatLabels: Record<string, string> = {
  lerobot: 'LeRobot',
  rtx: 'RT-X',
  rlds: 'RLDS',
  openx: 'Open X-Embodiment'
};

const robotTypeLabels: Record<string, string> = {
  single_arm: 'Single Arm',
  dual_arm: 'Dual Arm',
  bimanual: 'Bimanual',
  humanoid: 'Humanoid',
  mobile_manipulator: 'Mobile Manipulator',
  quadruped: 'Quadruped'
};

// ── Main component ───────────────────────────────────────────────────────────

export default function DatasetManager() {
  // ── My Datasets state ──────────────────────────────────────────────────────
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [isCreateDialogOpen, setIsCreateDialogOpen] = useState(false);
  const [isEditDialogOpen, setIsEditDialogOpen] = useState(false);
  const [selectedDatasetForEdit, setSelectedDatasetForEdit] = useState<Dataset | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [newDataset, setNewDataset] = useState<Partial<Dataset>>({
    format: 'lerobot',
    robotType: 'single_arm'
  });

  // ── GenRobot state ─────────────────────────────────────────────────────────
  const [grMetadata, setGrMetadata] = useState<GenRobotMetadata | null>(null);
  const [grSelectedSample, setGrSelectedSample] = useState<Sample | null>(null);
  const [grIsLoading, setGrIsLoading] = useState(true);
  const [grError, setGrError] = useState<string | null>(null);

  // ── My Datasets effects & handlers ────────────────────────────────────────

  useEffect(() => {
    loadDatasets();
  }, []);

  const loadDatasets = async () => {
    try {
      const data = await api.getDatasets();
      setDatasets(data);
    } catch (error) {
      console.error('Failed to load datasets:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleCreateDataset = async () => {
    try {
      await api.createDataset(newDataset);
      setIsCreateDialogOpen(false);
      setNewDataset({ format: 'lerobot', robotType: 'single_arm' });
      loadDatasets();
    } catch (error) {
      console.error('Failed to create dataset:', error);
    }
  };

  const handleUpdateDataset = async () => {
    if (!selectedDatasetForEdit) return;
    try {
      await api.updateDataset(selectedDatasetForEdit.id, selectedDatasetForEdit);
      setIsEditDialogOpen(false);
      setSelectedDatasetForEdit(null);
      loadDatasets();
    } catch (error) {
      console.error('Failed to update dataset:', error);
    }
  };

  const handleDeleteDataset = async (id: string) => {
    if (!confirm('Are you sure you want to delete this dataset?')) return;
    try {
      await api.deleteDataset(id);
      loadDatasets();
    } catch (error) {
      console.error('Failed to delete dataset:', error);
    }
  };

  const handleExportDataset = async (dataset: Dataset) => {
    try {
      const dataStr = JSON.stringify(dataset, null, 2);
      const dataBlob = new Blob([dataStr], { type: 'application/json' });
      const url = URL.createObjectURL(dataBlob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `${dataset.name}-metadata.json`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
    } catch (error) {
      console.error('Failed to export dataset:', error);
    }
  };

  const handleOpenDataset = (dataset: Dataset) => {
    console.log('Opening dataset:', dataset);
    alert(`Dataset "${dataset.name}" opened successfully!`);
  };

  const filteredDatasets = datasets.filter(d => 
    d.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
    d.description?.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  // ── GenRobot effects & handlers ────────────────────────────────────────────

  useEffect(() => {
    loadGenRobotDataset();
  }, []);

  const loadGenRobotDataset = async () => {
    setGrIsLoading(true);
    setGrError(null);
    try {
      const response = await fetch('/api/datasets/genrobot');
      if (!response.ok) {
        throw new Error('Failed to load GenRobot dataset');
      }
      const data = await response.json();
      setGrMetadata(data);
      if (data.samples.length > 0) {
        setGrSelectedSample(data.samples[0]);
      }
    } catch (err) {
      setGrError(err instanceof Error ? err.message : 'Unknown error');
      console.error('Failed to load GenRobot dataset:', err);
    } finally {
      setGrIsLoading(false);
    }
  };

  // ── Render ─────────────────────────────────────────────────────────────────

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold">数据集</h2>
        <p className="text-muted-foreground">管理数据集与开源数据</p>
      </div>

      <Tabs defaultValue="my">
        <TabsList>
          <TabsTrigger value="my">我的数据集</TabsTrigger>
          <TabsTrigger value="opensource">开源数据集</TabsTrigger>
        </TabsList>

        {/* ── Tab: 我的数据集 ── */}
        <TabsContent value="my" className="space-y-6 mt-4">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-lg font-semibold">我的数据集</h3>
              <p className="text-sm text-muted-foreground">Manage your embodied intelligence datasets</p>
            </div>
            <Dialog open={isCreateDialogOpen} onOpenChange={setIsCreateDialogOpen}>
              <DialogTrigger asChild>
                <Button>
                  <Plus className="w-4 h-4 mr-2" />
                  New Dataset
                </Button>
              </DialogTrigger>
              <DialogContent className="max-w-lg">
                <DialogHeader>
                  <DialogTitle>Create New Dataset</DialogTitle>
                </DialogHeader>
                <div className="space-y-4 pt-4">
                  <div className="space-y-2">
                    <Label>Name</Label>
                    <Input 
                      placeholder="Dataset name"
                      value={newDataset.name || ''}
                      onChange={e => setNewDataset({ ...newDataset, name: e.target.value })}
                    />
                  </div>
                  <div className="space-y-2">
                    <Label>Description</Label>
                    <Textarea 
                      placeholder="Dataset description"
                      value={newDataset.description || ''}
                      onChange={e => setNewDataset({ ...newDataset, description: e.target.value })}
                    />
                  </div>
                  <div className="grid grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <Label>Format</Label>
                      <Select 
                        value={newDataset.format} 
                        onValueChange={v => setNewDataset({ ...newDataset, format: v as Dataset['format'] })}
                      >
                        <SelectTrigger>
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="lerobot">LeRobot</SelectItem>
                          <SelectItem value="rtx">RT-X</SelectItem>
                          <SelectItem value="rlds">RLDS</SelectItem>
                          <SelectItem value="openx">Open X-Embodiment</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                    <div className="space-y-2">
                      <Label>Robot Type</Label>
                      <Select 
                        value={newDataset.robotType} 
                        onValueChange={v => setNewDataset({ ...newDataset, robotType: v })}
                      >
                        <SelectTrigger>
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="single_arm">Single Arm</SelectItem>
                          <SelectItem value="dual_arm">Dual Arm</SelectItem>
                          <SelectItem value="bimanual">Bimanual</SelectItem>
                          <SelectItem value="humanoid">Humanoid</SelectItem>
                          <SelectItem value="mobile_manipulator">Mobile Manipulator</SelectItem>
                          <SelectItem value="quadruped">Quadruped</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                  </div>
                  <div className="space-y-2">
                    <Label>Task Description</Label>
                    <Textarea 
                      placeholder="Describe the task/instruction for this dataset"
                      value={newDataset.taskDescription || ''}
                      onChange={e => setNewDataset({ ...newDataset, taskDescription: e.target.value })}
                    />
                  </div>
                  <Button onClick={handleCreateDataset} className="w-full">
                    Create Dataset
                  </Button>
                </div>
              </DialogContent>
            </Dialog>
          </div>

          {/* Edit Dataset Dialog */}
          <Dialog open={isEditDialogOpen} onOpenChange={setIsEditDialogOpen}>
            <DialogContent className="max-w-lg">
              <DialogHeader>
                <DialogTitle>Edit Dataset</DialogTitle>
              </DialogHeader>
              {selectedDatasetForEdit && (
                <div className="space-y-4 pt-4">
                  <div className="space-y-2">
                    <Label>Name</Label>
                    <Input 
                      placeholder="Dataset name"
                      value={selectedDatasetForEdit.name || ''}
                      onChange={e => setSelectedDatasetForEdit({ ...selectedDatasetForEdit, name: e.target.value })}
                    />
                  </div>
                  <div className="space-y-2">
                    <Label>Description</Label>
                    <Textarea 
                      placeholder="Dataset description"
                      value={selectedDatasetForEdit.description || ''}
                      onChange={e => setSelectedDatasetForEdit({ ...selectedDatasetForEdit, description: e.target.value })}
                    />
                  </div>
                  <div className="grid grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <Label>Format</Label>
                      <Select 
                        value={selectedDatasetForEdit.format} 
                        onValueChange={v => setSelectedDatasetForEdit({ ...selectedDatasetForEdit, format: v as Dataset['format'] })}
                      >
                        <SelectTrigger>
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="lerobot">LeRobot</SelectItem>
                          <SelectItem value="rtx">RT-X</SelectItem>
                          <SelectItem value="rlds">RLDS</SelectItem>
                          <SelectItem value="openx">Open X-Embodiment</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                    <div className="space-y-2">
                      <Label>Robot Type</Label>
                      <Select 
                        value={selectedDatasetForEdit.robotType} 
                        onValueChange={v => setSelectedDatasetForEdit({ ...selectedDatasetForEdit, robotType: v })}
                      >
                        <SelectTrigger>
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="single_arm">Single Arm</SelectItem>
                          <SelectItem value="dual_arm">Dual Arm</SelectItem>
                          <SelectItem value="bimanual">Bimanual</SelectItem>
                          <SelectItem value="humanoid">Humanoid</SelectItem>
                          <SelectItem value="mobile_manipulator">Mobile Manipulator</SelectItem>
                          <SelectItem value="quadruped">Quadruped</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                  </div>
                  <div className="space-y-2">
                    <Label>Task Description</Label>
                    <Textarea 
                      placeholder="Describe the task/instruction for this dataset"
                      value={selectedDatasetForEdit.taskDescription || ''}
                      onChange={e => setSelectedDatasetForEdit({ ...selectedDatasetForEdit, taskDescription: e.target.value })}
                    />
                  </div>
                  <div className="flex gap-2">
                    <Button onClick={handleUpdateDataset} className="flex-1">
                      Save Changes
                    </Button>
                    <Button 
                      onClick={() => setIsEditDialogOpen(false)} 
                      variant="outline"
                      className="flex-1"
                    >
                      Cancel
                    </Button>
                  </div>
                </div>
              )}
            </DialogContent>
          </Dialog>

          <div className="flex items-center gap-4">
            <div className="relative flex-1 max-w-md">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
              <Input 
                placeholder="Search datasets..."
                className="pl-10"
                value={searchQuery}
                onChange={e => setSearchQuery(e.target.value)}
              />
            </div>
          </div>

          {isLoading ? (
            <div className="text-center py-12">
              <div className="animate-spin w-8 h-8 border-2 border-primary border-t-transparent rounded-full mx-auto mb-4" />
              <p className="text-muted-foreground">Loading datasets...</p>
            </div>
          ) : filteredDatasets.length === 0 ? (
            <div className="text-center py-12">
              <Database className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
              <p className="text-muted-foreground">No datasets found</p>
              <Button variant="outline" className="mt-4" onClick={() => setIsCreateDialogOpen(true)}>
                Create your first dataset
              </Button>
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
              {filteredDatasets.map(dataset => (
                <Card key={dataset.id} className="hover:shadow-lg transition-shadow">
                  <CardHeader className="pb-3">
                    <div className="flex items-start justify-between">
                      <div className="flex items-center gap-3">
                        <div className="w-10 h-10 rounded-lg bg-primary/10 flex items-center justify-center">
                          <FileJson className="w-5 h-5 text-primary" />
                        </div>
                        <div>
                          <CardTitle className="text-lg">{dataset.name}</CardTitle>
                          <p className="text-xs text-muted-foreground">
                            {new Date(dataset.createdAt).toLocaleDateString()}
                          </p>
                        </div>
                      </div>
                      <DropdownMenu>
                        <DropdownMenuTrigger asChild>
                          <Button variant="ghost" size="icon">
                            <MoreVertical className="w-4 h-4" />
                          </Button>
                        </DropdownMenuTrigger>
                        <DropdownMenuContent align="end">
                          <DropdownMenuItem onClick={() => handleOpenDataset(dataset)}>
                            <FolderOpen className="w-4 h-4 mr-2" />
                            Open
                          </DropdownMenuItem>
                          <DropdownMenuItem 
                            onClick={() => {
                              setSelectedDatasetForEdit(dataset);
                              setIsEditDialogOpen(true);
                            }}
                          >
                            <Edit className="w-4 h-4 mr-2" />
                            Edit
                          </DropdownMenuItem>
                          <DropdownMenuItem onClick={() => handleExportDataset(dataset)}>
                            <Download className="w-4 h-4 mr-2" />
                            Export
                          </DropdownMenuItem>
                          <DropdownMenuItem 
                            className="text-destructive"
                            onClick={() => handleDeleteDataset(dataset.id)}
                          >
                            <Trash2 className="w-4 h-4 mr-2" />
                            Delete
                          </DropdownMenuItem>
                        </DropdownMenuContent>
                      </DropdownMenu>
                    </div>
                  </CardHeader>
                  <CardContent>
                    <p className="text-sm text-muted-foreground line-clamp-2 mb-4">
                      {dataset.description || 'No description'}
                    </p>
                    <div className="flex flex-wrap gap-2 mb-4">
                      <Badge variant="secondary">{formatLabels[dataset.format] || dataset.format}</Badge>
                      <Badge variant="outline">{robotTypeLabels[dataset.robotType] || dataset.robotType}</Badge>
                    </div>
                    <div className="grid grid-cols-3 gap-4 pt-4 border-t">
                      <div className="text-center">
                        <Layers className="w-4 h-4 mx-auto mb-1 text-muted-foreground" />
                        <p className="text-lg font-semibold">{dataset.episodeCount}</p>
                        <p className="text-xs text-muted-foreground">Episodes</p>
                      </div>
                      <div className="text-center">
                        <Database className="w-4 h-4 mx-auto mb-1 text-muted-foreground" />
                        <p className="text-lg font-semibold">{dataset.frameCount}</p>
                        <p className="text-xs text-muted-foreground">Frames</p>
                      </div>
                      <div className="text-center">
                        <FileJson className="w-4 h-4 mx-auto mb-1 text-muted-foreground" />
                        <p className="text-lg font-semibold">{formatFileSize(dataset.size)}</p>
                        <p className="text-xs text-muted-foreground">Size</p>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          )}
        </TabsContent>

        {/* ── Tab: 开源数据集 ── */}
        <TabsContent value="opensource" className="space-y-6 mt-4">
          {grIsLoading ? (
            <div className="flex items-center justify-center py-12">
              <div className="animate-spin w-8 h-8 border-2 border-primary border-t-transparent rounded-full mr-4" />
              <p className="text-muted-foreground">Loading GenRobot dataset...</p>
            </div>
          ) : grError ? (
            <div className="space-y-4">
              <Card className="border-destructive">
                <CardContent className="pt-6">
                  <p className="text-destructive font-medium">Error: {grError}</p>
                  <p className="text-sm text-muted-foreground mt-2">
                    Please run `python download_data.py` in the backend directory first.
                  </p>
                  <Button onClick={loadGenRobotDataset} className="mt-4">
                    <RefreshCw className="w-4 h-4 mr-2" />
                    Retry
                  </Button>
                </CardContent>
              </Card>
            </div>
          ) : !grMetadata ? (
            <div className="text-center py-12 text-muted-foreground">No data available</div>
          ) : (
            <div className="space-y-6">
              <div>
                <h3 className="text-lg font-semibold">GenRobot Open Dataset</h3>
                <p className="text-muted-foreground">Advanced robot learning dataset visualization</p>
              </div>

              {/* Overview */}
              <Card>
                <CardHeader>
                  <CardTitle className="text-lg flex items-center gap-2">
                    <Database className="w-5 h-5" />
                    Dataset Overview
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    <div className="space-y-2">
                      <p className="text-sm text-muted-foreground">Total Samples</p>
                      <p className="text-2xl font-bold">{grMetadata.samples.length}</p>
                    </div>
                    <div className="space-y-2">
                      <p className="text-sm text-muted-foreground">Total Frames</p>
                      <p className="text-2xl font-bold">
                        {grMetadata.samples.reduce((sum, s) => sum + s.frames, 0)}
                      </p>
                    </div>
                    <div className="space-y-2">
                      <p className="text-sm text-muted-foreground">Data Source</p>
                      <p className="text-sm font-medium truncate">GenRobot AI</p>
                    </div>
                    <div className="space-y-2">
                      <p className="text-sm text-muted-foreground">Status</p>
                      <Badge className="w-fit">Ready</Badge>
                    </div>
                  </div>
                  <p className="text-sm text-muted-foreground mt-4">
                    {grMetadata.description}
                  </p>
                </CardContent>
              </Card>

              <Tabs defaultValue="samples" className="w-full">
                <TabsList className="grid w-full max-w-md grid-cols-3">
                  <TabsTrigger value="samples">
                    <Layers className="w-4 h-4 mr-2" />
                    Samples
                  </TabsTrigger>
                  <TabsTrigger value="details">
                    <Eye className="w-4 h-4 mr-2" />
                    Details
                  </TabsTrigger>
                  <TabsTrigger value="stats">
                    <BarChart3 className="w-4 h-4 mr-2" />
                    Stats
                  </TabsTrigger>
                </TabsList>

                {/* Samples Tab */}
                <TabsContent value="samples" className="space-y-4">
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                    {grMetadata.samples.map(sample => (
                      <Card
                        key={sample.id}
                        className={`cursor-pointer transition-all hover:shadow-lg ${
                          grSelectedSample?.id === sample.id ? 'ring-2 ring-primary' : ''
                        }`}
                        onClick={() => setGrSelectedSample(sample)}
                      >
                        <CardContent className="pt-6">
                          <div className="space-y-3">
                            <div>
                              <p className="text-sm text-muted-foreground">Episode ID</p>
                              <p className="font-medium">{sample.episode_id}</p>
                            </div>
                            <div>
                              <p className="text-sm text-muted-foreground">Task</p>
                              <p className="font-medium">{sample.task}</p>
                            </div>
                            <div className="grid grid-cols-2 gap-4 text-sm">
                              <div>
                                <p className="text-muted-foreground">Frames</p>
                                <p className="font-semibold text-lg">{sample.frames}</p>
                              </div>
                              <div>
                                <p className="text-muted-foreground">Action Dims</p>
                                <p className="font-semibold text-lg">{sample.actions_dim}</p>
                              </div>
                            </div>
                            <Badge variant="secondary">{sample.robot_type}</Badge>
                          </div>
                        </CardContent>
                      </Card>
                    ))}
                  </div>
                </TabsContent>

                {/* Details Tab */}
                <TabsContent value="details" className="space-y-4">
                  {grSelectedSample ? (
                    <Card>
                      <CardHeader>
                        <CardTitle className="text-lg">Sample: {grSelectedSample.id}</CardTitle>
                      </CardHeader>
                      <CardContent className="space-y-6">
                        <div className="grid grid-cols-2 gap-4">
                          <div>
                            <p className="text-sm text-muted-foreground">Episode ID</p>
                            <p className="font-medium">{grSelectedSample.episode_id}</p>
                          </div>
                          <div>
                            <p className="text-sm text-muted-foreground">Task</p>
                            <p className="font-medium">{grSelectedSample.task}</p>
                          </div>
                          <div>
                            <p className="text-sm text-muted-foreground">Total Frames</p>
                            <p className="font-medium text-xl">{grSelectedSample.frames}</p>
                          </div>
                          <div>
                            <p className="text-sm text-muted-foreground">Robot Type</p>
                            <p className="font-medium">{grSelectedSample.robot_type}</p>
                          </div>
                        </div>

                        <div>
                          <h4 className="font-semibold mb-3 flex items-center gap-2">
                            <Eye className="w-4 h-4" />
                            Observations
                          </h4>
                          <div className="space-y-2">
                            {Object.entries(grSelectedSample.observations).map(([obsName, obsData]) => (
                              <div key={obsName} className="p-3 bg-muted rounded-lg">
                                <p className="font-medium text-sm mb-1">{obsName}</p>
                                <div className="text-xs text-muted-foreground space-y-1">
                                  <p>Shape: {JSON.stringify((obsData as any).shape)}</p>
                                  <p>Type: {(obsData as any).type}</p>
                                </div>
                              </div>
                            ))}
                          </div>
                        </div>

                        <div>
                          <h4 className="font-semibold mb-3 flex items-center gap-2">
                            <Zap className="w-4 h-4" />
                            Action Configuration
                          </h4>
                          <div className="p-3 bg-muted rounded-lg">
                            <p className="text-sm">
                              <span className="text-muted-foreground">Action Dimensions:</span>
                              <span className="font-medium ml-2">{grSelectedSample.actions_dim}</span>
                            </p>
                          </div>
                        </div>
                      </CardContent>
                    </Card>
                  ) : (
                    <Card>
                      <CardContent className="pt-6">
                        <p className="text-center text-muted-foreground">
                          Select a sample to view details
                        </p>
                      </CardContent>
                    </Card>
                  )}
                </TabsContent>

                {/* Stats Tab */}
                <TabsContent value="stats" className="space-y-4">
                  <Card>
                    <CardHeader>
                      <CardTitle className="text-lg">Dataset Statistics</CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-6">
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                        <div>
                          <h4 className="font-semibold mb-4">Frame Distribution</h4>
                          <div className="space-y-3">
                            {grMetadata.samples.map(sample => (
                              <div key={sample.id}>
                                <div className="flex justify-between text-sm mb-1">
                                  <span>{sample.episode_id}</span>
                                  <span className="font-medium">{sample.frames}</span>
                                </div>
                                <div className="w-full bg-muted rounded-full h-2">
                                  <div
                                    className="bg-primary h-2 rounded-full"
                                    style={{
                                      width: `${(sample.frames / Math.max(...grMetadata.samples.map(s => s.frames))) * 100}%`
                                    }}
                                  />
                                </div>
                              </div>
                            ))}
                          </div>
                        </div>

                        <div>
                          <h4 className="font-semibold mb-4">Robot Types</h4>
                          <div className="space-y-2">
                            {Array.from(new Set(grMetadata.samples.map(s => s.robot_type))).map(type => (
                              <div key={type} className="flex items-center justify-between">
                                <Badge variant="secondary">{type}</Badge>
                                <span className="text-sm text-muted-foreground">
                                  {grMetadata.samples.filter(s => s.robot_type === type).length} samples
                                </span>
                              </div>
                            ))}
                          </div>
                        </div>
                      </div>

                      <div className="bg-muted p-4 rounded-lg">
                        <h4 className="font-semibold mb-3">Summary</h4>
                        <div className="space-y-2 text-sm">
                          <p>
                            <span className="text-muted-foreground">Average Frames per Sample:</span>
                            <span className="font-medium ml-2">
                              {(grMetadata.samples.reduce((sum, s) => sum + s.frames, 0) / grMetadata.samples.length).toFixed(2)}
                            </span>
                          </p>
                          <p>
                            <span className="text-muted-foreground">Average Action Dimensions:</span>
                            <span className="font-medium ml-2">
                              {(grMetadata.samples.reduce((sum, s) => sum + s.actions_dim, 0) / grMetadata.samples.length).toFixed(2)}
                            </span>
                          </p>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                </TabsContent>
              </Tabs>
            </div>
          )}
        </TabsContent>
      </Tabs>
    </div>
  );
}
