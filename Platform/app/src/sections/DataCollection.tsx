import { useState, useEffect, useRef } from 'react';
import { 
  Play, 
  Pause, 
  Square, 
  Circle, 
  Camera, 
  Settings,
  RotateCcw,
  MonitorPlay,
  Radio,
  Grip,
  Hand
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Label } from '@/components/ui/label';
import { api } from '@/services/api';
import { useWebSocket } from '@/services/websocket';
import type { Dataset, Simulator } from '@/types';

interface RecordingState {
  isRecording: boolean;
  isPaused: boolean;
  episodeId: string | null;
  frameCount: number;
  startTime: number | null;
  duration: number;
}

export default function DataCollection() {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [simulators, setSimulators] = useState<Simulator[]>([]);
  const [selectedDataset, setSelectedDataset] = useState<string>('');
  const [selectedSimulator, setSelectedSimulator] = useState<string>('');
  const [robotIP, setRobotIP] = useState<string>('192.168.1.100');
  const [collectionMode, setCollectionMode] = useState<'real' | 'simulation'>('simulation');
  const [isReconnecting, setIsReconnecting] = useState(false);
  const [connectionStatus, setConnectionStatus] = useState<'disconnected' | 'connected' | 'connecting'>('disconnected');
  const [recording, setRecording] = useState<RecordingState>({
    isRecording: false,
    isPaused: false,
    episodeId: null,
    frameCount: 0,
    startTime: null,
    duration: 0
  });
  const [sensorData, setSensorData] = useState<Record<string, any>>({});
  const [activeSensors, setActiveSensors] = useState<string[]>(['rgb', 'depth', 'force_torque']);
  const durationInterval = useRef<ReturnType<typeof setInterval> | null>(null);
  
  const { isConnected: _isConnected, on } = useWebSocket('collector');

  useEffect(() => {
    loadData();
    
    const unsubscribe = on('sensor_data', (data) => {
      setSensorData(prev => ({
        ...prev,
        [data.sensorType]: data
      }));
      
      if (recording.isRecording && !recording.isPaused) {
        setRecording(prev => ({
          ...prev,
          frameCount: prev.frameCount + 1
        }));
      }
    });

    return () => {
      unsubscribe?.();
      if (durationInterval.current) {
        clearInterval(durationInterval.current);
      }
    };
  }, [recording.isRecording, recording.isPaused]);

  const loadData = async () => {
    try {
      const [datasetsData, simulatorsData] = await Promise.all([
        api.getDatasets(),
        api.getSimulators()
      ]);
      setDatasets(datasetsData);
      setSimulators(simulatorsData);
    } catch (error) {
      console.error('Failed to load data:', error);
    }
  };

  const handleReconnect = async () => {
    setIsReconnecting(true);
    setConnectionStatus('connecting');
    
    try {
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      console.log(`Attempting to reconnect to robot at ${robotIP}`);
      setConnectionStatus('connected');
      
      setTimeout(() => {
        setIsReconnecting(false);
      }, 500);
    } catch (error) {
      console.error('Failed to reconnect:', error);
      setConnectionStatus('disconnected');
      setIsReconnecting(false);
    }
  };

  const startRecording = async () => {
    if (!selectedDataset) {
      alert('Please select a dataset first');
      return;
    }

    const episodeId = `episode_${Date.now()}`;
    setRecording({
      isRecording: true,
      isPaused: false,
      episodeId,
      frameCount: 0,
      startTime: Date.now(),
      duration: 0
    });

    durationInterval.current = setInterval(() => {
      setRecording(prev => ({
        ...prev,
        duration: prev.startTime ? Math.floor((Date.now() - prev.startTime) / 1000) : 0
      }));
    }, 1000);
  };

  const pauseRecording = () => {
    setRecording(prev => ({ ...prev, isPaused: !prev.isPaused }));
  };

  const stopRecording = async () => {
    if (durationInterval.current) {
      clearInterval(durationInterval.current);
    }

    if (recording.episodeId) {
      try {
        console.log('Saving episode:', recording.episodeId);
      } catch (error) {
        console.error('Failed to save episode:', error);
      }
    }

    setRecording({
      isRecording: false,
      isPaused: false,
      episodeId: null,
      frameCount: 0,
      startTime: null,
      duration: 0
    });
  };

  const formatDuration = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  const toggleSensor = (sensor: string) => {
    setActiveSensors(prev => 
      prev.includes(sensor) 
        ? prev.filter(s => s !== sensor)
        : [...prev, sensor]
    );
  };

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold">Data Collection</h2>
        <p className="text-muted-foreground">Collect embodied intelligence data from real robots or simulators</p>
      </div>

      <Tabs value={collectionMode} onValueChange={(v) => setCollectionMode(v as any)}>
        <TabsList className="grid w-full max-w-md grid-cols-2">
          <TabsTrigger value="simulation">
            <MonitorPlay className="w-4 h-4 mr-2" />
            Simulation
          </TabsTrigger>
          <TabsTrigger value="real">
            <Radio className="w-4 h-4 mr-2" />
            Real Robot
          </TabsTrigger>
        </TabsList>

        <TabsContent value="simulation" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="text-lg flex items-center gap-2">
                <Settings className="w-5 h-5" />
                Simulator Configuration
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="space-y-2">
                  <Label>Simulator</Label>
                  <Select value={selectedSimulator} onValueChange={setSelectedSimulator}>
                    <SelectTrigger>
                      <SelectValue placeholder="Select simulator" />
                    </SelectTrigger>
                    <SelectContent>
                      {simulators.map(sim => (
                        <SelectItem key={sim.id} value={sim.id}>
                          {sim.name}
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
                        <SelectItem key={ds.id} value={ds.id}>
                          {ds.name}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-2">
                  <Label>Scene</Label>
                  <Select>
                    <SelectTrigger>
                      <SelectValue placeholder="Select scene" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="tabletop">Tabletop</SelectItem>
                      <SelectItem value="kitchen">Kitchen</SelectItem>
                      <SelectItem value="factory">Factory</SelectItem>
                      <SelectItem value="office">Office</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="real" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="text-lg flex items-center gap-2">
                <Radio className="w-5 h-5" />
                Robot Connection
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label>Dataset</Label>
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
                <div className="space-y-2">
                  <Label>Robot IP</Label>
                  <input 
                    type="text" 
                    placeholder="192.168.1.100"
                    value={robotIP}
                    onChange={(e) => setRobotIP(e.target.value)}
                    className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
                  />
                </div>
              </div>
              <div className="flex items-center gap-4">
                <Badge variant={connectionStatus === 'connected' ? 'default' : connectionStatus === 'connecting' ? 'secondary' : 'destructive'}>
                  {connectionStatus === 'connected' ? 'Connected' : connectionStatus === 'connecting' ? 'Connecting...' : 'Disconnected'}
                </Badge>
                <Button 
                  variant="outline" 
                  size="sm"
                  onClick={handleReconnect}
                  disabled={isReconnecting}
                >
                  <RotateCcw className={`w-4 h-4 mr-2 ${isReconnecting ? 'animate-spin' : ''}`} />
                  {isReconnecting ? 'Reconnecting...' : 'Reconnect'}
                </Button>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>

      {/* Sensor Configuration */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg">Active Sensors</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex flex-wrap gap-3">
            {[
              { id: 'rgb', name: 'RGB Camera', icon: Camera },
              { id: 'depth', name: 'Depth Camera', icon: Camera },
              { id: 'rgbd', name: 'RGB-D Camera', icon: Camera },
              { id: 'force_torque', name: 'F/T Sensor', icon: Grip },
              { id: 'tactile', name: 'Tactile', icon: Hand },
              { id: 'joint_pos', name: 'Joint Pos', icon: Settings },
              { id: 'imu', name: 'IMU', icon: RotateCcw },
            ].map(sensor => {
              const Icon = sensor.icon;
              const isActive = activeSensors.includes(sensor.id);
              return (
                <button
                  key={sensor.id}
                  onClick={() => toggleSensor(sensor.id)}
                  className={`flex items-center gap-2 px-4 py-2 rounded-lg border transition-colors ${
                    isActive 
                      ? 'bg-primary text-primary-foreground border-primary' 
                      : 'bg-background text-muted-foreground border-border hover:border-primary/50'
                  }`}
                >
                  <Icon className="w-4 h-4" />
                  {sensor.name}
                </button>
              );
            })}
          </div>
        </CardContent>
      </Card>

      {/* Recording Controls */}
      <Card className="border-primary/20">
        <CardContent className="p-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-6">
              <div className="text-center">
                <p className="text-3xl font-mono font-bold">{formatDuration(recording.duration)}</p>
                <p className="text-xs text-muted-foreground">Duration</p>
              </div>
              <div className="w-px h-12 bg-border" />
              <div className="text-center">
                <p className="text-3xl font-mono font-bold">{recording.frameCount}</p>
                <p className="text-xs text-muted-foreground">Frames</p>
              </div>
              <div className="w-px h-12 bg-border" />
              <div>
                <Badge variant={recording.isRecording ? 'default' : 'secondary'}>
                  {recording.isRecording ? (recording.isPaused ? 'Paused' : 'Recording') : 'Ready'}
                </Badge>
                {recording.episodeId && (
                  <p className="text-xs text-muted-foreground mt-1">
                    {recording.episodeId.slice(-12)}
                  </p>
                )}
              </div>
            </div>
            
            <div className="flex items-center gap-2">
              {!recording.isRecording ? (
                <Button 
                  size="lg" 
                  onClick={startRecording}
                  disabled={!selectedDataset}
                  className="bg-red-500 hover:bg-red-600"
                >
                  <Circle className="w-5 h-5 mr-2 fill-current" />
                  Record
                </Button>
              ) : (
                <>
                  <Button 
                    size="lg" 
                    variant="outline"
                    onClick={pauseRecording}
                  >
                    {recording.isPaused ? <Play className="w-5 h-5" /> : <Pause className="w-5 h-5" />}
                  </Button>
                  <Button 
                    size="lg" 
                    variant="destructive"
                    onClick={stopRecording}
                  >
                    <Square className="w-5 h-5 mr-2 fill-current" />
                    Stop
                  </Button>
                </>
              )}
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Sensor Visualization */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {activeSensors.includes('rgb') && (
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm flex items-center gap-2">
                <Camera className="w-4 h-4" />
                RGB Camera
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="aspect-video bg-muted rounded-lg flex items-center justify-center">
                {sensorData.rgb?.image ? (
                  <img 
                    src={sensorData.rgb.image} 
                    alt="RGB" 
                    className="w-full h-full object-cover rounded-lg"
                  />
                ) : (
                  <div className="text-center">
                    <Camera className="w-8 h-8 text-muted-foreground mx-auto mb-2" />
                    <p className="text-sm text-muted-foreground">Waiting for data...</p>
                  </div>
                )}
              </div>
              <div className="mt-2 text-xs text-muted-foreground">
                <p>Resolution: 640x480 @ 30fps</p>
                <p>Location: wrist</p>
              </div>
            </CardContent>
          </Card>
        )}

        {activeSensors.includes('depth') && (
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm flex items-center gap-2">
                <Camera className="w-4 h-4" />
                Depth Camera
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="aspect-video bg-muted rounded-lg flex items-center justify-center">
                {sensorData.depth?.image ? (
                  <img 
                    src={sensorData.depth.image} 
                    alt="Depth" 
                    className="w-full h-full object-cover rounded-lg"
                  />
                ) : (
                  <div className="text-center">
                    <Camera className="w-8 h-8 text-muted-foreground mx-auto mb-2" />
                    <p className="text-sm text-muted-foreground">Waiting for data...</p>
                  </div>
                )}
              </div>
              <div className="mt-2 text-xs text-muted-foreground">
                <p>Resolution: 640x480 @ 30fps</p>
                <p>Range: 0.1m - 10m</p>
              </div>
            </CardContent>
          </Card>
        )}

        {activeSensors.includes('force_torque') && (
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm flex items-center gap-2">
                <Grip className="w-4 h-4" />
                Force/Torque Sensor
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                <div>
                  <div className="flex justify-between text-xs mb-1">
                    <span>Force X</span>
                    <span>{(sensorData.force_torque?.force?.[0] || 0).toFixed(2)} N</span>
                  </div>
                  <div className="h-2 bg-muted rounded-full overflow-hidden">
                    <div 
                      className="h-full bg-primary transition-all"
                      style={{ width: `${Math.min(Math.abs(sensorData.force_torque?.force?.[0] || 0) / 100 * 50 + 50, 100)}%` }}
                    />
                  </div>
                </div>
                <div>
                  <div className="flex justify-between text-xs mb-1">
                    <span>Force Y</span>
                    <span>{(sensorData.force_torque?.force?.[1] || 0).toFixed(2)} N</span>
                  </div>
                  <div className="h-2 bg-muted rounded-full overflow-hidden">
                    <div 
                      className="h-full bg-primary transition-all"
                      style={{ width: `${Math.min(Math.abs(sensorData.force_torque?.force?.[1] || 0) / 100 * 50 + 50, 100)}%` }}
                    />
                  </div>
                </div>
                <div>
                  <div className="flex justify-between text-xs mb-1">
                    <span>Force Z</span>
                    <span>{(sensorData.force_torque?.force?.[2] || 0).toFixed(2)} N</span>
                  </div>
                  <div className="h-2 bg-muted rounded-full overflow-hidden">
                    <div 
                      className="h-full bg-primary transition-all"
                      style={{ width: `${Math.min(Math.abs(sensorData.force_torque?.force?.[2] || 0) / 100 * 50 + 50, 100)}%` }}
                    />
                  </div>
                </div>
              </div>
              <div className="mt-3 text-xs text-muted-foreground">
                <p>Frequency: 1000Hz</p>
                <p>Location: wrist</p>
              </div>
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  );
}
