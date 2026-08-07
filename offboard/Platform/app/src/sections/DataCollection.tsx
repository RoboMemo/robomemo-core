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
  Hand,
  Activity
} from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
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
  const [robotIP, setRobotIP] = useState<string>('192.168.1.12');
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
  const [imuHistory, setImuHistory] = useState<Array<{ t: number; ax: number; ay: number; az: number; gx: number; gy: number; gz: number }>>([]);
  const [streamStatus, setStreamStatus] = useState<{ state: string; episodeId?: string; dataset?: string } | null>(null);
  const [activeSensors, setActiveSensors] = useState<string[]>(['rgb', 'depth', 'force_torque']);
  const durationInterval = useRef<ReturnType<typeof setInterval> | null>(null);

  const { isConnected, on } = useWebSocket('collector');

  useEffect(() => {
    loadData();

    const unsubscribeSensor = on('sensor_data', (data) => {
      setSensorData(prev => ({
        ...prev,
        [data.sensorType]: data
      }));

      // X5 IMU stream: keep a short rolling window for the accel + gyro charts.
      if (data.sensorType === 'imu' && Array.isArray(data.accel_mps2)) {
        const [ax, ay, az] = data.accel_mps2;
        const [gx, gy, gz] = Array.isArray(data.gyro_rps) ? data.gyro_rps : [0, 0, 0];
        setImuHistory(prev => {
          const next = [...prev, { t: data.ts_ns ?? prev.length, ax, ay, az, gx, gy, gz }];
          return next.length > 150 ? next.slice(next.length - 150) : next;
        });
      }

      if (recording.isRecording && !recording.isPaused && data.sensorType === 'cam0') {
        setRecording(prev => ({
          ...prev,
          frameCount: prev.frameCount + 1
        }));
      }
    });

    const unsubscribeStatus = on('recording_status', (data: any) => {
      setStreamStatus({ state: data.state, episodeId: data.episodeId, dataset: data.dataset });
    });

    return () => {
      unsubscribeSensor?.();
      unsubscribeStatus?.();
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
    const dataset = selectedDataset || 'rdk_x5_live';
    try {
      const r = await api.startRecording({ dataset, ip: robotIP });
      setRecording({
        isRecording: true,
        isPaused: false,
        episodeId: r.episodeId,
        frameCount: 0,
        startTime: Date.now(),
        duration: 0,
      });
      durationInterval.current = setInterval(() => {
        setRecording(prev => ({
          ...prev,
          duration: prev.startTime ? Math.floor((Date.now() - prev.startTime) / 1000) : 0,
        }));
      }, 1000);
    } catch (e: any) {
      alert(`Failed to start recording: ${e?.message || e}`);
    }
  };

  const pauseRecording = () => {
    setRecording(prev => ({ ...prev, isPaused: !prev.isPaused }));
  };

  const stopRecording = async () => {
    if (durationInterval.current) {
      clearInterval(durationInterval.current);
    }
    try {
      await api.stopRecording();
    } catch (error) {
      console.error('Failed to stop recording:', error);
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

      {/* Live Preview — RDK X5 (4 RGB cameras + IMU), fed by record.py --stream */}
      <Card>
        <CardHeader className="pb-2">
          <div className="flex items-center justify-between">
            <CardTitle className="text-lg flex items-center gap-2">
              <Radio className="w-5 h-5" />
              Live Preview — RDK X5
            </CardTitle>
            <div className="flex items-center gap-2">
              <Badge variant={isConnected ? 'default' : 'secondary'}>
                {isConnected ? 'WS connected' : 'WS disconnected'}
              </Badge>
              {streamStatus?.state === 'recording' ? (
                <Badge variant="destructive">
                  <Circle className="w-3 h-3 mr-1 fill-current" />
                  Recording{streamStatus.episodeId ? ` · ${streamStatus.episodeId.slice(-8)}` : ''}
                </Badge>
              ) : streamStatus?.state === 'stopped' ? (
                <Badge variant="secondary">Stopped</Badge>
              ) : (
                <Badge variant="outline">Live preview</Badge>
              )}
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            {[0, 1, 2, 3].map((i) => {
              const key = `cam${i}`;
              const shot = sensorData[key];
              return (
                <Card key={key} className="overflow-hidden">
                  <CardHeader className="pb-1 pt-2">
                    <CardTitle className="text-xs flex items-center gap-2">
                      <Camera className="w-3.5 h-3.5" /> {key}
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="p-2">
                    <div className="aspect-video bg-muted rounded-md flex items-center justify-center overflow-hidden">
                      {shot?.image ? (
                        <img src={shot.image} alt={key} className="w-full h-full object-cover rounded-md" />
                      ) : (
                        <div className="text-center">
                          <Camera className="w-7 h-7 text-muted-foreground mx-auto mb-1" />
                          <p className="text-xs text-muted-foreground">Waiting for stream…</p>
                        </div>
                      )}
                    </div>
                  </CardContent>
                </Card>
              );
            })}
          </div>

          {/* IMU accel timeseries */}
          <Card className="mt-3">
            <CardHeader className="pb-1 pt-3">
              <CardTitle className="text-xs flex items-center gap-2">
                <Activity className="w-3.5 h-3.5" /> IMU accel (m/s²)
                <span className="ml-auto text-[10px] font-normal text-muted-foreground">
                  {imuHistory.length
                    ? `|a| ≈ ${Math.hypot(
                        imuHistory[imuHistory.length - 1].ax,
                        imuHistory[imuHistory.length - 1].ay,
                        imuHistory[imuHistory.length - 1].az,
                      ).toFixed(2)}`
                    : '—'}
                </span>
              </CardTitle>
            </CardHeader>
            <CardContent className="p-2">
              <div className="h-44">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={imuHistory}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="t" tick={false} height={0} />
                    <YAxis domain={[-20, 20]} width={32} />
                    <Tooltip />
                    <Legend />
                    <Line type="monotone" dataKey="ax" stroke="#ef4444" name="ax" dot={false} isAnimationActive={false} />
                    <Line type="monotone" dataKey="ay" stroke="#22c55e" name="ay" dot={false} isAnimationActive={false} />
                    <Line type="monotone" dataKey="az" stroke="#3b82f6" name="az" dot={false} isAnimationActive={false} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>

          {/* IMU gyro (angular velocity) timeseries */}
          <Card className="mt-3">
            <CardHeader className="pb-1 pt-3">
              <CardTitle className="text-xs flex items-center gap-2">
                <Activity className="w-3.5 h-3.5" /> IMU gyro (rad/s)
                <span className="ml-auto text-[10px] font-normal text-muted-foreground">
                  {imuHistory.length
                    ? `|ω| ≈ ${Math.hypot(
                        imuHistory[imuHistory.length - 1].gx,
                        imuHistory[imuHistory.length - 1].gy,
                        imuHistory[imuHistory.length - 1].gz,
                      ).toFixed(3)}`
                    : '—'}
                </span>
              </CardTitle>
            </CardHeader>
            <CardContent className="p-2">
              <div className="h-40">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={imuHistory}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="t" tick={false} height={0} />
                    <YAxis width={32} />
                    <Tooltip />
                    <Legend />
                    <Line type="monotone" dataKey="gx" stroke="#ef4444" name="gx" dot={false} isAnimationActive={false} />
                    <Line type="monotone" dataKey="gy" stroke="#22c55e" name="gy" dot={false} isAnimationActive={false} />
                    <Line type="monotone" dataKey="gz" stroke="#3b82f6" name="gz" dot={false} isAnimationActive={false} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>
        </CardContent>
      </Card>
    </div>
  );
}
