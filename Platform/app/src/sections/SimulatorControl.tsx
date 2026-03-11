import { useState, useEffect } from 'react';
import { 
  Play, 
  Square, 
  RotateCcw, 
  StepForward,
  Settings,
  Cpu,
  Monitor,
  Activity,
  Layers,
  Box,
  Grip
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Label } from '@/components/ui/label';
import { Slider } from '@/components/ui/slider';
import { api } from '@/services/api';
import type { Simulator } from '@/types';

export default function SimulatorControl() {
  const [simulators, setSimulators] = useState<Simulator[]>([]);
  const [selectedSimulator, setSelectedSimulator] = useState<string>('');
  const [scenes, setScenes] = useState<any[]>([]);
  const [robots, setRobots] = useState<any[]>([]);
  const [selectedScene, setSelectedScene] = useState<string>('');
  const [selectedRobot, setSelectedRobot] = useState<string>('');
  const [activeSensors, setActiveSensors] = useState<string[]>(['rgb', 'depth']);
  const [isRunning, setIsRunning] = useState(false);
  const [simulationSpeed, setSimulationSpeed] = useState(1.0);
  const [stats, setStats] = useState({
    fps: 60,
    physicsTime: 0.016,
    renderTime: 0.008,
    stepCount: 0
  });

  useEffect(() => {
    loadSimulators();
  }, []);

  useEffect(() => {
    if (selectedSimulator) {
      loadSimulatorDetails(selectedSimulator);
    }
  }, [selectedSimulator]);

  const loadSimulators = async () => {
    try {
      const data = await api.getSimulators();
      setSimulators(data);
      if (data.length > 0) {
        setSelectedSimulator(data[0].id);
      }
    } catch (error) {
      console.error('Failed to load simulators:', error);
    }
  };

  const loadSimulatorDetails = async (simulatorId: string) => {
    try {
      const [scenesData, robotsData] = await Promise.all([
        api.getSimulatorScenes(simulatorId),
        api.getSimulatorRobots(simulatorId)
      ]);
      setScenes(scenesData);
      setRobots(robotsData);
    } catch (error) {
      console.error('Failed to load simulator details:', error);
    }
  };

  const handleStart = async () => {
    if (!selectedSimulator || !selectedScene || !selectedRobot) {
      alert('Please select simulator, scene, and robot');
      return;
    }

    try {
      await api.startSimulator(selectedSimulator, {
        scene: selectedScene,
        robot: selectedRobot,
        sensors: activeSensors
      });
      setIsRunning(true);
      
      const interval = setInterval(() => {
        setStats(prev => ({
          fps: 58 + Math.random() * 4,
          physicsTime: 0.015 + Math.random() * 0.002,
          renderTime: 0.007 + Math.random() * 0.002,
          stepCount: prev.stepCount + 1
        }));
      }, 1000);
      
      return () => clearInterval(interval);
    } catch (error) {
      console.error('Failed to start simulator:', error);
    }
  };

  const handleStop = async () => {
    try {
      await api.stopSimulator(selectedSimulator);
      setIsRunning(false);
    } catch (error) {
      console.error('Failed to stop simulator:', error);
    }
  };

  const handleReset = async () => {
    try {
      await api.resetSimulator(selectedSimulator);
      setStats(prev => ({ ...prev, stepCount: 0 }));
    } catch (error) {
      console.error('Failed to reset simulator:', error);
    }
  };

  const handleStep = async () => {
    try {
      await api.stepSimulator(selectedSimulator);
      setStats(prev => ({ ...prev, stepCount: prev.stepCount + 1 }));
    } catch (error) {
      console.error('Failed to step simulator:', error);
    }
  };

  const toggleSensor = (sensor: string) => {
    setActiveSensors(prev => 
      prev.includes(sensor)
        ? prev.filter(s => s !== sensor)
        : [...prev, sensor]
    );
  };

  const currentSimulator = simulators.find(s => s.id === selectedSimulator);

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold">Simulator Control</h2>
        <p className="text-muted-foreground">Control Isaac Lab, MuJoCo, and Gazebo simulations</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Simulator Selection */}
        <Card>
          <CardHeader>
            <CardTitle className="text-lg flex items-center gap-2">
              <Cpu className="w-5 h-5" />
              Simulator
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <Label>Select Simulator</Label>
              <Select value={selectedSimulator} onValueChange={setSelectedSimulator}>
                <SelectTrigger>
                  <SelectValue />
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

            {currentSimulator && (
              <div className="space-y-2 text-sm">
                <div className="flex items-center justify-between">
                  <span className="text-muted-foreground">Status</span>
                  <Badge variant={currentSimulator.status === 'running' ? 'default' : 'secondary'}>
                    {currentSimulator.status}
                  </Badge>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-muted-foreground">Type</span>
                  <span className="font-medium">{currentSimulator.type}</span>
                </div>
                <p className="text-muted-foreground text-xs mt-2">
                  {currentSimulator.description}
                </p>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Scene & Robot */}
        <Card>
          <CardHeader>
            <CardTitle className="text-lg flex items-center gap-2">
              <Box className="w-5 h-5" />
              Configuration
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <Label>Scene</Label>
              <Select value={selectedScene} onValueChange={setSelectedScene}>
                <SelectTrigger>
                  <SelectValue placeholder="Select scene" />
                </SelectTrigger>
                <SelectContent>
                  {scenes.map(scene => (
                    <SelectItem key={scene.id} value={scene.id}>
                      {scene.name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2">
              <Label>Robot</Label>
              <Select value={selectedRobot} onValueChange={setSelectedRobot}>
                <SelectTrigger>
                  <SelectValue placeholder="Select robot" />
                </SelectTrigger>
                <SelectContent>
                  {robots.map(robot => (
                    <SelectItem key={robot.id} value={robot.id}>
                      {robot.name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2">
              <Label>Simulation Speed</Label>
              <div className="flex items-center gap-4">
                <Slider 
                  value={[simulationSpeed]} 
                  onValueChange={v => setSimulationSpeed(v[0])}
                  min={0.1}
                  max={2}
                  step={0.1}
                  className="flex-1"
                />
                <span className="text-sm w-12">{simulationSpeed.toFixed(1)}x</span>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Control Panel */}
        <Card>
          <CardHeader>
            <CardTitle className="text-lg flex items-center gap-2">
              <Activity className="w-5 h-5" />
              Control
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex gap-2">
              {!isRunning ? (
                <Button onClick={handleStart} className="flex-1">
                  <Play className="w-4 h-4 mr-2" />
                  Start
                </Button>
              ) : (
                <Button onClick={handleStop} variant="destructive" className="flex-1">
                  <Square className="w-4 h-4 mr-2 fill-current" />
                  Stop
                </Button>
              )}
              <Button onClick={handleReset} variant="outline">
                <RotateCcw className="w-4 h-4" />
              </Button>
              <Button onClick={handleStep} variant="outline" disabled={!isRunning}>
                <StepForward className="w-4 h-4" />
              </Button>
            </div>

            <div className="grid grid-cols-2 gap-4 pt-4 border-t">
              <div className="text-center">
                <p className="text-2xl font-mono font-bold">{stats.fps.toFixed(0)}</p>
                <p className="text-xs text-muted-foreground">FPS</p>
              </div>
              <div className="text-center">
                <p className="text-2xl font-mono font-bold">{stats.stepCount}</p>
                <p className="text-xs text-muted-foreground">Steps</p>
              </div>
              <div className="text-center">
                <p className="text-2xl font-mono font-bold">{(stats.physicsTime * 1000).toFixed(1)}</p>
                <p className="text-xs text-muted-foreground">Physics ms</p>
              </div>
              <div className="text-center">
                <p className="text-2xl font-mono font-bold">{(stats.renderTime * 1000).toFixed(1)}</p>
                <p className="text-xs text-muted-foreground">Render ms</p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Sensors */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg flex items-center gap-2">
            <Layers className="w-5 h-5" />
            Active Sensors
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex flex-wrap gap-3">
            {currentSimulator?.supportedSensors.map(sensor => {
              const isActive = activeSensors.includes(sensor);
              const icons: Record<string, any> = {
                rgb: Monitor,
                depth: Layers,
                segmentation: Box,
                force_torque: Grip,
                tactile: Activity,
                joint_pos: Settings,
                joint_vel: Settings,
                imu: Activity,
                lidar: Layers
              };
              const Icon = icons[sensor] || Activity;
              
              return (
                <button
                  key={sensor}
                  onClick={() => toggleSensor(sensor)}
                  disabled={isRunning}
                  className={`flex items-center gap-2 px-4 py-2 rounded-lg border transition-colors disabled:opacity-50 ${
                    isActive 
                      ? 'bg-primary text-primary-foreground border-primary' 
                      : 'bg-background text-muted-foreground border-border hover:border-primary/50'
                  }`}
                >
                  <Icon className="w-4 h-4" />
                  {sensor.replace('_', ' ').toUpperCase()}
                </button>
              );
            })}
          </div>
        </CardContent>
      </Card>

      {/* Visualization */}
      <Tabs defaultValue="viewport" className="space-y-4">
        <TabsList>
          <TabsTrigger value="viewport">
            <Monitor className="w-4 h-4 mr-2" />
            Viewport
          </TabsTrigger>
          <TabsTrigger value="sensors">
            <Activity className="w-4 h-4 mr-2" />
            Sensor Data
          </TabsTrigger>
          <TabsTrigger value="logs">
            <Layers className="w-4 h-4 mr-2" />
            Logs
          </TabsTrigger>
        </TabsList>

        <TabsContent value="viewport">
          <Card>
            <CardContent className="p-4">
              <div className="aspect-video bg-muted rounded-lg flex items-center justify-center">
                {isRunning ? (
                  <div className="relative w-full h-full">
                    <img 
                      src={`https://picsum.photos/seed/sim${Date.now()}/1280/720`}
                      alt="Simulation"
                      className="w-full h-full object-cover rounded-lg"
                    />
                    <div className="absolute top-4 left-4 bg-black/50 text-white px-3 py-1 rounded text-sm">
                      {currentSimulator?.name} - {selectedScene}
                    </div>
                    <div className="absolute top-4 right-4 bg-black/50 text-white px-3 py-1 rounded text-sm">
                      {stats.fps.toFixed(0)} FPS
                    </div>
                  </div>
                ) : (
                  <div className="text-center">
                    <Monitor className="w-16 h-16 text-muted-foreground mx-auto mb-4" />
                    <p className="text-muted-foreground">Simulation not running</p>
                    <Button onClick={handleStart} className="mt-4">
                      <Play className="w-4 h-4 mr-2" />
                      Start Simulation
                    </Button>
                  </div>
                )}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="sensors">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm">RGB Camera</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="aspect-video bg-muted rounded-lg">
                  {isRunning && (
                    <img 
                      src={`https://picsum.photos/seed/rgb${Date.now()}/640/480`}
                      alt="RGB"
                      className="w-full h-full object-cover rounded-lg"
                    />
                  )}
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm">Depth Camera</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="aspect-video bg-muted rounded-lg">
                  {isRunning && (
                    <img 
                      src={`https://picsum.photos/seed/depth${Date.now()}/640/480?grayscale`}
                      alt="Depth"
                      className="w-full h-full object-cover rounded-lg"
                    />
                  )}
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="logs">
          <Card>
            <CardContent className="p-4">
              <div className="bg-black text-green-400 font-mono text-sm p-4 rounded-lg h-64 overflow-auto">
                <p>[INFO] Simulator initialized</p>
                <p>[INFO] Loading scene: {selectedScene || 'none'}</p>
                <p>[INFO] Loading robot: {selectedRobot || 'none'}</p>
                {isRunning && (
                  <>
                    <p>[INFO] Simulation started</p>
                    <p>[INFO] Physics engine: {currentSimulator?.type}</p>
                    <p>[INFO] Active sensors: {activeSensors.join(', ')}</p>
                    <p>[DEBUG] Step {stats.stepCount}: physics={stats.physicsTime.toFixed(4)}s, render={stats.renderTime.toFixed(4)}s</p>
                  </>
                )}
                {!isRunning && stats.stepCount > 0 && (
                  <p>[INFO] Simulation stopped</p>
                )}
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
