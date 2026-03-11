import { useState, useEffect, useRef } from 'react';
import { 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  Legend, 
  ResponsiveContainer,
  ScatterChart,
  Scatter
} from 'recharts';
import { 
  Play, 
  Pause, 
  SkipBack, 
  Camera,
  Grip,
  Activity,
  Layers,
  Maximize2
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Slider } from '@/components/ui/slider';
import { api } from '@/services/api';
import { useWebSocket } from '@/services/websocket';
import type { Dataset, Episode } from '@/types';

interface TrajectoryPoint {
  x: number;
  y: number;
  z: number;
  t: number;
}

export default function DataVisualization() {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [selectedDataset, setSelectedDataset] = useState<string>('');
  const [episodes, setEpisodes] = useState<Episode[]>([]);
  const [selectedEpisode, setSelectedEpisode] = useState<string>('');
  const [currentFrame, setCurrentFrame] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [sensorData, setSensorData] = useState<Record<string, any>>({});
  const [trajectoryData, setTrajectoryData] = useState<TrajectoryPoint[]>([]);
  const playbackInterval = useRef<ReturnType<typeof setInterval> | null>(null);
  const { isConnected, on } = useWebSocket('visualization');

  useEffect(() => {
    loadDatasets();
    
    const unsubscribe = on('sensor_data', (data) => {
      setSensorData(prev => ({
        ...prev,
        [data.sensorType || 'unknown']: data
      }));
    });

    return () => {
      unsubscribe?.();
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
    const mockTrajectory: TrajectoryPoint[] = Array.from({ length: 100 }, (_, i) => ({
      x: Math.sin(i * 0.1) * 0.3 + Math.random() * 0.05,
      y: Math.cos(i * 0.1) * 0.2 + Math.random() * 0.05,
      z: 0.1 + i * 0.001 + Math.random() * 0.02,
      t: i
    }));
    setTrajectoryData(mockTrajectory);
  }, [selectedEpisode]);

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

  const forceData = sensorData['force_torque']?.map((d: any) => ({
    time: new Date(d.timestamp).toLocaleTimeString(),
    fx: Math.sin(d.timestamp / 1000) * 10 + Math.random() * 2,
    fy: Math.cos(d.timestamp / 1000) * 8 + Math.random() * 2,
    fz: 20 + Math.random() * 5,
  })) || [];

  const jointData = Array.from({ length: 50 }, (_, i) => ({
    time: i,
    j1: Math.sin(i * 0.2) * 1.5,
    j2: Math.cos(i * 0.2) * 1.2,
    j3: Math.sin(i * 0.15) * 0.8,
    j4: Math.cos(i * 0.15) * 0.6,
    j5: Math.sin(i * 0.1) * 0.4,
    j6: Math.cos(i * 0.1) * 0.3,
    j7: Math.sin(i * 0.05) * 0.2,
  }));

  const tactileData = Array.from({ length: 64 }, (_, i) => ({
    x: i % 8,
    y: Math.floor(i / 8),
    pressure: Math.random() * 100
  }));

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold">Data Visualization</h2>
        <p className="text-muted-foreground">Visualize sensor data, trajectories, and episode playback</p>
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
          <Button variant="outline" size="icon" onClick={() => setCurrentFrame(0)}>
            <SkipBack className="w-4 h-4" />
          </Button>
          <Button variant="outline" size="icon" onClick={togglePlayback}>
            {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
          </Button>
          <div className="w-32">
            <Slider 
              value={[currentFrame]} 
              onValueChange={v => setCurrentFrame(v[0])}
              max={99}
              step={1}
            />
          </div>
          <span className="text-sm text-muted-foreground w-16">
            {currentFrame + 1}/100
          </span>
        </div>

        <Badge variant={isConnected ? 'default' : 'destructive'}>
          {isConnected ? 'Live' : 'Offline'}
        </Badge>
      </div>

      <Tabs defaultValue="video" className="space-y-4">
        <TabsList>
          <TabsTrigger value="video">
            <Camera className="w-4 h-4 mr-2" />
            Video
          </TabsTrigger>
          <TabsTrigger value="forces">
            <Grip className="w-4 h-4 mr-2" />
            Forces
          </TabsTrigger>
          <TabsTrigger value="joints">
            <Activity className="w-4 h-4 mr-2" />
            Joints
          </TabsTrigger>
          <TabsTrigger value="trajectory">
            <Layers className="w-4 h-4 mr-2" />
            Trajectory
          </TabsTrigger>
          <TabsTrigger value="tactile">
            <Maximize2 className="w-4 h-4 mr-2" />
            Tactile
          </TabsTrigger>
        </TabsList>

        <TabsContent value="video" className="space-y-4">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm flex items-center gap-2">
                  <Camera className="w-4 h-4" />
                  Left Wrist Camera
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="aspect-video bg-muted rounded-lg flex items-center justify-center">
                  <img 
                    src={`https://picsum.photos/seed/left${currentFrame}/640/480`}
                    alt="Left camera"
                    className="w-full h-full object-cover rounded-lg"
                  />
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm flex items-center gap-2">
                  <Camera className="w-4 h-4" />
                  Right Wrist Camera
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="aspect-video bg-muted rounded-lg flex items-center justify-center">
                  <img 
                    src={`https://picsum.photos/seed/right${currentFrame}/640/480`}
                    alt="Right camera"
                    className="w-full h-full object-cover rounded-lg"
                  />
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm flex items-center gap-2">
                  <Camera className="w-4 h-4" />
                  Chest Camera (RGB-D)
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="aspect-video bg-muted rounded-lg flex items-center justify-center">
                  <img 
                    src={`https://picsum.photos/seed/chest${currentFrame}/640/480`}
                    alt="Chest camera"
                    className="w-full h-full object-cover rounded-lg"
                  />
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm flex items-center gap-2">
                  <Camera className="w-4 h-4" />
                  Depth View
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="aspect-video bg-muted rounded-lg flex items-center justify-center">
                  <img 
                    src={`https://picsum.photos/seed/depth${currentFrame}/640/480?grayscale`}
                    alt="Depth"
                    className="w-full h-full object-cover rounded-lg"
                  />
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="forces">
          <Card>
            <CardHeader>
              <CardTitle className="text-lg flex items-center gap-2">
                <Grip className="w-5 h-5" />
                Force/Torque Sensor Data
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="h-80">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={forceData.length > 0 ? forceData : Array.from({ length: 50 }, (_, i) => ({
                    time: i,
                    fx: Math.sin(i * 0.2) * 10 + Math.random() * 2,
                    fy: Math.cos(i * 0.2) * 8 + Math.random() * 2,
                    fz: 20 + Math.random() * 5,
                  }))}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="time" />
                    <YAxis />
                    <Tooltip />
                    <Legend />
                    <Line type="monotone" dataKey="fx" stroke="#ef4444" name="Force X" dot={false} />
                    <Line type="monotone" dataKey="fy" stroke="#22c55e" name="Force Y" dot={false} />
                    <Line type="monotone" dataKey="fz" stroke="#3b82f6" name="Force Z" dot={false} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="joints">
          <Card>
            <CardHeader>
              <CardTitle className="text-lg flex items-center gap-2">
                <Activity className="w-5 h-5" />
                Joint Positions
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="h-80">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={jointData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="time" />
                    <YAxis />
                    <Tooltip />
                    <Legend />
                    <Line type="monotone" dataKey="j1" stroke="#8884d8" name="Joint 1" dot={false} />
                    <Line type="monotone" dataKey="j2" stroke="#82ca9d" name="Joint 2" dot={false} />
                    <Line type="monotone" dataKey="j3" stroke="#ffc658" name="Joint 3" dot={false} />
                    <Line type="monotone" dataKey="j4" stroke="#ff7300" name="Joint 4" dot={false} />
                    <Line type="monotone" dataKey="j5" stroke="#00C49F" name="Joint 5" dot={false} />
                    <Line type="monotone" dataKey="j6" stroke="#FFBB28" name="Joint 6" dot={false} />
                    <Line type="monotone" dataKey="j7" stroke="#FF8042" name="Joint 7" dot={false} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="trajectory">
          <Card>
            <CardHeader>
              <CardTitle className="text-lg flex items-center gap-2">
                <Layers className="w-5 h-5" />
                End Effector Trajectory
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                <div className="h-80">
                  <ResponsiveContainer width="100%" height="100%">
                    <ScatterChart>
                      <CartesianGrid />
                      <XAxis type="number" dataKey="x" name="X" domain={[-0.5, 0.5]} />
                      <YAxis type="number" dataKey="y" name="Y" domain={[-0.5, 0.5]} />
                      <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                      <Scatter name="Trajectory" data={trajectoryData} fill="#8884d8" />
                    </ScatterChart>
                  </ResponsiveContainer>
                  <p className="text-center text-sm text-muted-foreground mt-2">XY Plane</p>
                </div>
                <div className="h-80">
                  <ResponsiveContainer width="100%" height="100%">
                    <ScatterChart>
                      <CartesianGrid />
                      <XAxis type="number" dataKey="x" name="X" domain={[-0.5, 0.5]} />
                      <YAxis type="number" dataKey="z" name="Z" domain={[0, 0.3]} />
                      <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                      <Scatter name="Trajectory" data={trajectoryData} fill="#82ca9d" />
                    </ScatterChart>
                  </ResponsiveContainer>
                  <p className="text-center text-sm text-muted-foreground mt-2">XZ Plane</p>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="tactile">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Left Hand Tactile Sensor</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-8 gap-1">
                  {tactileData.map((cell, i) => (
                    <div
                      key={i}
                      className="aspect-square rounded"
                      style={{
                        backgroundColor: `rgba(239, 68, 68, ${cell.pressure / 100})`
                      }}
                    />
                  ))}
                </div>
                <div className="flex justify-between text-xs text-muted-foreground mt-2">
                  <span>0</span>
                  <span>Pressure</span>
                  <span>Max</span>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Right Hand Tactile Sensor</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-8 gap-1">
                  {tactileData.map((cell, i) => (
                    <div
                      key={i}
                      className="aspect-square rounded"
                      style={{
                        backgroundColor: `rgba(34, 197, 94, ${cell.pressure / 100})`
                      }}
                    />
                  ))}
                </div>
                <div className="flex justify-between text-xs text-muted-foreground mt-2">
                  <span>0</span>
                  <span>Pressure</span>
                  <span>Max</span>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
}
