import { useState, useEffect, useCallback, useRef } from 'react';
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
  CloudDownload,
  Heart,
  Loader2
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
import { ScrollArea } from '@/components/ui/scroll-area';
import { api } from '@/services/api';
import type { Dataset } from '@/types';

interface HFDataset {
  id: string;
  description?: string;
  downloads?: number;
  likes?: number;
  tags?: string[];
  lastModified?: string;
  private?: boolean;
}

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

export default function DatasetManager() {
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
  const [isHubDialogOpen, setIsHubDialogOpen] = useState(false);
  const [hubSearchQuery, setHubSearchQuery] = useState('');
  const [hubResults, setHubResults] = useState<HFDataset[]>([]);
  const [hubLoading, setHubLoading] = useState(false);
  const [hubError, setHubError] = useState<string | null>(null);
  const hubDebounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const popularChips = ['lerobot', 'aloha', 'pusht', 'robot manipulation'];

  const searchHubDatasets = useCallback(async (query: string) => {
    if (!query.trim()) {
      setHubResults([]);
      setHubLoading(false);
      return;
    }
    setHubLoading(true);
    setHubError(null);
    try {
      const res = await fetch(
        `https://huggingface.co/api/datasets?search=${encodeURIComponent(query)}&limit=20&full=true`
      );
      if (!res.ok) throw new Error('Request failed');
      const data: HFDataset[] = await res.json();
      setHubResults(data);
    } catch {
      setHubError('Failed to search HuggingFace Hub. Check network.');
      setHubResults([]);
    } finally {
      setHubLoading(false);
    }
  }, []);

  const handleHubSearchChange = useCallback((value: string) => {
    setHubSearchQuery(value);
    if (hubDebounceRef.current) clearTimeout(hubDebounceRef.current);
    hubDebounceRef.current = setTimeout(() => {
      searchHubDatasets(value);
    }, 500);
  }, [searchHubDatasets]);

  const handleImportFromHub = (hfDataset: HFDataset) => {
    setIsHubDialogOpen(false);
    const hasLerobot = hfDataset.tags?.includes('lerobot');
    setNewDataset({
      name: hfDataset.id,
      description: hfDataset.description?.slice(0, 500) || '',
      format: hasLerobot ? 'lerobot' : 'rlds',
      robotType: 'single_arm'
    });
    setIsCreateDialogOpen(true);
  };

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

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold">Datasets</h2>
          <p className="text-muted-foreground">Manage your embodied intelligence datasets</p>
        </div>
        <div className="flex gap-2">
          <Button variant="outline" onClick={() => setIsHubDialogOpen(true)}>
            <CloudDownload className="w-4 h-4 mr-2" />
            Import from Hub
          </Button>
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
                    onValueChange={v => setNewDataset({ ...newDataset, format: v as any })}
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
      </div>

      {/* Hub Import Dialog */}
      <Dialog open={isHubDialogOpen} onOpenChange={(open) => {
        setIsHubDialogOpen(open);
        if (!open) {
          setHubSearchQuery('');
          setHubResults([]);
          setHubError(null);
        }
      }}>
        <DialogContent className="max-w-2xl">
          <DialogHeader>
            <DialogTitle>Import Open Dataset from Hub</DialogTitle>
          </DialogHeader>
          <div className="space-y-4 pt-2">
            <div className="relative">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
              <Input
                placeholder="Search robot datasets (e.g. lerobot, aloha, pusht)..."
                className="pl-10"
                value={hubSearchQuery}
                onChange={e => handleHubSearchChange(e.target.value)}
              />
            </div>
            {!hubSearchQuery && (
              <div className="flex gap-2 flex-wrap">
                {popularChips.map(chip => (
                  <Button
                    key={chip}
                    variant="outline"
                    size="sm"
                    onClick={() => handleHubSearchChange(chip)}
                  >
                    {chip}
                  </Button>
                ))}
              </div>
            )}
            {hubLoading && (
              <div className="flex items-center justify-center py-8">
                <Loader2 className="w-6 h-6 animate-spin text-muted-foreground" />
              </div>
            )}
            {hubError && (
              <p className="text-sm text-destructive text-center py-4">{hubError}</p>
            )}
            {!hubLoading && !hubError && hubResults.length > 0 && (
              <ScrollArea className="h-[400px]">
                <div className="space-y-3 pr-4">
                  {hubResults.map(ds => (
                    <Card key={ds.id} className="p-4">
                      <div className="flex items-start justify-between gap-4">
                        <div className="flex-1 min-w-0">
                          <p className="font-bold text-sm truncate">{ds.id}</p>
                          {ds.description && (
                            <p className="text-xs text-muted-foreground mt-1 line-clamp-2">
                              {ds.description.slice(0, 120)}
                            </p>
                          )}
                          <div className="flex flex-wrap gap-1.5 mt-2">
                            {ds.downloads != null && (
                              <Badge variant="secondary" className="text-xs">
                                <Download className="w-3 h-3 mr-1" />
                                {ds.downloads.toLocaleString()}
                              </Badge>
                            )}
                            {ds.likes != null && (
                              <Badge variant="secondary" className="text-xs">
                                <Heart className="w-3 h-3 mr-1" />
                                {ds.likes.toLocaleString()}
                              </Badge>
                            )}
                            {ds.tags?.slice(0, 3).map(tag => (
                              <Badge key={tag} variant="outline" className="text-xs">{tag}</Badge>
                            ))}
                          </div>
                        </div>
                        <Button size="sm" onClick={() => handleImportFromHub(ds)}>
                          Import
                        </Button>
                      </div>
                    </Card>
                  ))}
                </div>
              </ScrollArea>
            )}
            {!hubLoading && !hubError && hubSearchQuery && hubResults.length === 0 && (
              <p className="text-sm text-muted-foreground text-center py-4">No datasets found.</p>
            )}
          </div>
        </DialogContent>
      </Dialog>

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
                    onValueChange={v => setSelectedDatasetForEdit({ ...selectedDatasetForEdit, format: v as any })}
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
    </div>
  );
}
