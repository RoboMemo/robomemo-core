import { useState, useEffect } from 'react';
import { 
  Sparkles, 
  RefreshCw, 
  ArrowRightLeft, 
  Palette,
  Play,
  Cpu,
  Layers
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Label } from '@/components/ui/label';
import { Switch } from '@/components/ui/switch';
import { Slider } from '@/components/ui/slider';
import { Progress } from '@/components/ui/progress';
import { api } from '@/services/api';
import type { Dataset, AugmentationModel } from '@/types';

interface AugmentationJob {
  id: string;
  type: string;
  status: 'queued' | 'running' | 'completed' | 'failed';
  progress: number;
  datasetId: string;
  modelId: string;
  createdAt: string;
}

export default function DataAugmentation() {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [models, setModels] = useState<AugmentationModel[]>([]);
  const [selectedDataset, setSelectedDataset] = useState<string>('');
  const [selectedModel, setSelectedModel] = useState<string>('');
  const [jobs, setJobs] = useState<AugmentationJob[]>([]);
  const [activeTab, setActiveTab] = useState('cross-embodiment');
  
  const [sourceRobot, setSourceRobot] = useState('');
  const [targetRobot, setTargetRobot] = useState('');
  
  const [targetContext, setTargetContext] = useState('');
  const [contextVariations, setContextVariations] = useState(5);
  
  const [drSettings, setDrSettings] = useState({
    lighting: true,
    texture: true,
    camera: true,
    object: true,
    background: true
  });
  const [drVariations, setDrVariations] = useState(10);

  useEffect(() => {
    loadData();
  }, []);

  const loadData = async () => {
    try {
      const [datasetsData, modelsData] = await Promise.all([
        api.getDatasets(),
        api.getAugmentationModels()
      ]);
      setDatasets(datasetsData);
      setModels(modelsData);
    } catch (error) {
      console.error('Failed to load data:', error);
    }
  };

  const startAugmentation = async () => {
    if (!selectedDataset || !selectedModel) {
      alert('Please select dataset and model');
      return;
    }

    const jobId = `job_${Date.now()}`;
    const newJob: AugmentationJob = {
      id: jobId,
      type: activeTab,
      status: 'queued',
      progress: 0,
      datasetId: selectedDataset,
      modelId: selectedModel,
      createdAt: new Date().toISOString()
    };

    setJobs(prev => [newJob, ...prev]);

    try {
      switch (activeTab) {
        case 'cross-embodiment':
          await api.crossEmbodimentTransfer({
            sourceDatasetId: selectedDataset,
            sourceRobotType: sourceRobot,
            targetRobotType: targetRobot,
            modelId: selectedModel
          });
          break;
        case 'cross-context':
          await api.crossContextGeneration({
            datasetId: selectedDataset,
            targetContext,
            modelId: selectedModel,
            variations: contextVariations
          });
          break;
        case 'domain-randomization':
          await api.generateAugmentedData(selectedModel, selectedDataset, {
            type: 'domain_randomization',
            parameters: drSettings,
            numVariations: drVariations
          });
          break;
        default:
          await api.generateAugmentedData(selectedModel, selectedDataset, {});
      }

      simulateJobProgress(jobId);
    } catch (error) {
      console.error('Failed to start augmentation:', error);
      setJobs(prev => prev.map(j => j.id === jobId ? { ...j, status: 'failed' } : j));
    }
  };

  const simulateJobProgress = (jobId: string) => {
    let progress = 0;
    const interval = setInterval(() => {
      progress += Math.random() * 15;
      if (progress >= 100) {
        progress = 100;
        clearInterval(interval);
        setJobs(prev => prev.map(j => 
          j.id === jobId ? { ...j, status: 'completed', progress: 100 } : j
        ));
      } else {
        setJobs(prev => prev.map(j => 
          j.id === jobId ? { ...j, status: 'running', progress } : j
        ));
      }
    }, 1000);
  };

  const selectedModelData = models.find(m => m.id === selectedModel);

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold">Data Augmentation</h2>
        <p className="text-muted-foreground">
          Use world models like Emu3.5 and Rynn-002 for cross-embodiment and cross-context generation
        </p>
      </div>

      {/* Model Selection */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg flex items-center gap-2">
            <Cpu className="w-5 h-5" />
            World Model
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label>Select Model</Label>
              <Select value={selectedModel} onValueChange={setSelectedModel}>
                <SelectTrigger>
                  <SelectValue placeholder="Select augmentation model" />
                </SelectTrigger>
                <SelectContent>
                  {models.map(model => (
                    <SelectItem key={model.id} value={model.id}>
                      {model.name} ({model.provider})
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-2">
              <Label>Target Dataset</Label>
              <Select value={selectedDataset} onValueChange={setSelectedDataset}>
                <SelectTrigger>
                  <SelectValue placeholder="Select dataset" />
                </SelectTrigger>
                <SelectContent>
                  {datasets.map(ds => (
                    <SelectItem key={ds.id} value={ds.id}>
                      {ds.name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </div>

          {selectedModelData && (
            <div className="p-4 bg-muted rounded-lg">
              <div className="flex items-start gap-4">
                <div className="w-12 h-12 rounded-lg bg-primary/10 flex items-center justify-center">
                  <Sparkles className="w-6 h-6 text-primary" />
                </div>
                <div className="flex-1">
                  <h4 className="font-medium">{selectedModelData.name}</h4>
                  <p className="text-sm text-muted-foreground">{selectedModelData.description}</p>
                  <div className="flex flex-wrap gap-2 mt-2">
                    {selectedModelData.capabilities.map(cap => (
                      <Badge key={cap} variant="secondary" className="text-xs">
                        {cap.replace('_', ' ')}
                      </Badge>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Augmentation Types */}
      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="cross-embodiment">
            <ArrowRightLeft className="w-4 h-4 mr-2" />
            Cross-Embodiment
          </TabsTrigger>
          <TabsTrigger value="cross-context">
            <Palette className="w-4 h-4 mr-2" />
            Cross-Context
          </TabsTrigger>
          <TabsTrigger value="domain-randomization">
            <RefreshCw className="w-4 h-4 mr-2" />
            Domain Randomization
          </TabsTrigger>
        </TabsList>

        <TabsContent value="cross-embodiment" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="text-lg">Cross-Embodiment Transfer</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <p className="text-sm text-muted-foreground">
                Transfer skills from one robot embodiment to another using world models.
                This enables training on one robot and deploying on another.
              </p>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label>Source Robot Type</Label>
                  <Select value={sourceRobot} onValueChange={setSourceRobot}>
                    <SelectTrigger>
                      <SelectValue placeholder="Select source robot" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="franka">Franka Emika Panda</SelectItem>
                      <SelectItem value="ur10">Universal Robots UR10</SelectItem>
                      <SelectItem value="baxter">Rethink Baxter</SelectItem>
                      <SelectItem value="humanoid">Humanoid</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-2">
                  <Label>Target Robot Type</Label>
                  <Select value={targetRobot} onValueChange={setTargetRobot}>
                    <SelectTrigger>
                      <SelectValue placeholder="Select target robot" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="franka">Franka Emika Panda</SelectItem>
                      <SelectItem value="ur10">Universal Robots UR10</SelectItem>
                      <SelectItem value="baxter">Rethink Baxter</SelectItem>
                      <SelectItem value="humanoid">Humanoid</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="cross-context" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="text-lg">Cross-Context Generation</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <p className="text-sm text-muted-foreground">
                Generate variations of your data in different contexts (lighting, background, textures)
                while preserving the underlying task structure.
              </p>
              <div className="space-y-4">
                <div className="space-y-2">
                  <Label>Target Context</Label>
                  <Select value={targetContext} onValueChange={setTargetContext}>
                    <SelectTrigger>
                      <SelectValue placeholder="Select target context" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="kitchen">Kitchen Environment</SelectItem>
                      <SelectItem value="factory">Factory Floor</SelectItem>
                      <SelectItem value="office">Office Space</SelectItem>
                      <SelectItem value="outdoor">Outdoor Scene</SelectItem>
                      <SelectItem value="lab">Research Lab</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-2">
                  <Label>Number of Variations: {contextVariations}</Label>
                  <Slider 
                    value={[contextVariations]} 
                    onValueChange={v => setContextVariations(v[0])}
                    min={1}
                    max={20}
                    step={1}
                  />
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="domain-randomization" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="text-lg">Domain Randomization</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <p className="text-sm text-muted-foreground">
                Randomize simulation parameters to improve policy robustness and sim-to-real transfer.
              </p>
              <div className="space-y-4">
                <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                  {Object.entries(drSettings).map(([key, value]) => (
                    <div key={key} className="flex items-center justify-between p-3 border rounded-lg">
                      <span className="capitalize">{key}</span>
                      <Switch 
                        checked={value} 
                        onCheckedChange={v => setDrSettings(prev => ({ ...prev, [key]: v }))}
                      />
                    </div>
                  ))}
                </div>
                <div className="space-y-2">
                  <Label>Number of Variations: {drVariations}</Label>
                  <Slider 
                    value={[drVariations]} 
                    onValueChange={v => setDrVariations(v[0])}
                    min={5}
                    max={100}
                    step={5}
                  />
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>

      {/* Start Button */}
      <Button 
        onClick={startAugmentation} 
        disabled={!selectedDataset || !selectedModel}
        className="w-full"
        size="lg"
      >
        <Play className="w-5 h-5 mr-2" />
        Start Augmentation
      </Button>

      {/* Jobs List */}
      {jobs.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="text-lg flex items-center gap-2">
              <Layers className="w-5 h-5" />
              Augmentation Jobs
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {jobs.map(job => {
                const dataset = datasets.find(d => d.id === job.datasetId);
                const model = models.find(m => m.id === job.modelId);
                
                return (
                  <div key={job.id} className="p-4 border rounded-lg">
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center gap-3">
                        <Badge variant={
                          job.status === 'completed' ? 'default' :
                          job.status === 'running' ? 'secondary' :
                          job.status === 'failed' ? 'destructive' :
                          'outline'
                        }>
                          {job.status}
                        </Badge>
                        <span className="font-medium">{job.type.replace('-', ' ')}</span>
                      </div>
                      <span className="text-sm text-muted-foreground">
                        {new Date(job.createdAt).toLocaleTimeString()}
                      </span>
                    </div>
                    <div className="text-sm text-muted-foreground mb-2">
                      {dataset?.name} → {model?.name}
                    </div>
                    <div className="flex items-center gap-3">
                      <Progress value={job.progress} className="flex-1" />
                      <span className="text-sm w-12 text-right">{job.progress.toFixed(0)}%</span>
                    </div>
                  </div>
                );
              })}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
