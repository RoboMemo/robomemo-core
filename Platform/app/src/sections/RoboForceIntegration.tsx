import { useState, useEffect } from 'react';
import { Upload, CheckCircle2, AlertTriangle, Monitor, Cpu } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import api from '@/services/api';

const MOCK_EPISODES = [
  { name: 'screw_drive_01', description: 'M4 screw insertion at 5Nm torque', frameCount: 500, duration: 16.7, fps: 30, sensors: ['rgbd_front', 'rgbd_left', 'rgbd_right', 'ft_left', 'ft_right'] },
  { name: 'screw_drive_02', description: 'M6 screw tightening sequence', frameCount: 450, duration: 15.0, fps: 30, sensors: ['rgbd_front', 'rgbd_left', 'rgbd_right', 'ft_left', 'ft_right'] },
  { name: 'assembly_01', description: 'Bimanual bracket assembly', frameCount: 900, duration: 30.0, fps: 30, sensors: ['rgbd_front', 'rgbd_left', 'rgbd_right', 'ft_left', 'ft_right'] },
];

export default function RoboForceIntegration() {
  const [presets, setPresets] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [importName, setImportName] = useState('');
  const [importJson, setImportJson] = useState('');
  const [importResult, setImportResult] = useState<any>(null);
  const [validateResult, setValidateResult] = useState<any>(null);
  const [processing, setProcessing] = useState(false);

  useEffect(() => {
    loadPresets();
  }, []);

  const loadPresets = async () => {
    try {
      const data = await api.getRoboForceSensorPresets();
      setPresets(data);
    } catch (err) {
      console.error('Failed to load presets:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleImport = async () => {
    if (!importName) return;
    setProcessing(true);
    try {
      const episodes = importJson ? JSON.parse(importJson) : MOCK_EPISODES;
      const result = await api.importRoboForceData({
        datasetName: importName,
        episodes,
        preset: 'titan_standard',
      });
      setImportResult(result);
    } catch (err: any) {
      setImportResult({ error: err.message });
    } finally {
      setProcessing(false);
    }
  };

  const handleValidate = async () => {
    setProcessing(true);
    try {
      const data = importJson ? JSON.parse(importJson) : { episodes: MOCK_EPISODES, sensors: presets[0]?.sensors || [] };
      const result = await api.validateRoboForceData({ data, preset: 'titan_standard' });
      setValidateResult(result);
    } catch (err: any) {
      setValidateResult({ error: err.message });
    } finally {
      setProcessing(false);
    }
  };

  const useMockData = () => {
    setImportJson(JSON.stringify(MOCK_EPISODES, null, 2));
    setImportName('RoboForce Titan Screw Driving');
  };

  if (loading) return <div className="text-center py-12 text-muted-foreground">Loading...</div>;

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold">RoboForce Data Integration</h1>
        <p className="text-muted-foreground mt-1">Import and validate RoboForce Titan data</p>
      </div>

      {/* Sensor Preset Card */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2"><Cpu className="w-5 h-5" /> Sensor Configuration</CardTitle>
          <CardDescription>RoboForce Titan: RGBD×3 + F/T 6-axis×2</CardDescription>
        </CardHeader>
        <CardContent>
          {presets.length > 0 ? (
            <div className="space-y-3">
              {presets.map((preset, i) => (
                <div key={i}>
                  <h3 className="font-medium">{preset.name}</h3>
                  <p className="text-sm text-muted-foreground mb-2">{preset.description}</p>
                  <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-2">
                    {preset.sensors?.map((sensor: any, j: number) => (
                      <div key={j} className="flex items-center gap-2 p-2 bg-slate-50 rounded border">
                        <Monitor className="w-4 h-4 text-blue-500" />
                        <div>
                          <div className="text-sm font-medium">{sensor.name}</div>
                          <div className="text-xs text-muted-foreground">{sensor.type} · {sensor.location}</div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-muted-foreground">No sensor presets found</p>
          )}
        </CardContent>
      </Card>

      <Tabs defaultValue="import">
        <TabsList>
          <TabsTrigger value="import">Import</TabsTrigger>
          <TabsTrigger value="validate">Validate</TabsTrigger>
        </TabsList>

        <TabsContent value="import">
          <Card>
            <CardHeader>
              <CardTitle>Import RoboForce Data</CardTitle>
              <CardDescription>Import episodes from RoboForce data format</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex gap-2">
                <Button variant="outline" size="sm" onClick={useMockData}>
                  Use Mock Data (3 episodes)
                </Button>
              </div>
              <div>
                <Label>Dataset Name</Label>
                <Input placeholder="RoboForce Titan Dataset" value={importName} onChange={(e) => setImportName(e.target.value)} />
              </div>
              <div>
                <Label>Episodes JSON (or use mock data)</Label>
                <Textarea
                  placeholder="Paste RoboForce episodes JSON..."
                  value={importJson}
                  onChange={(e) => setImportJson(e.target.value)}
                  rows={8}
                  className="font-mono text-xs"
                />
              </div>
              <Button onClick={handleImport} disabled={processing || !importName}>
                <Upload className="w-4 h-4 mr-2" /> Import
              </Button>
              {importResult && (
                <div className={`p-3 rounded-lg text-sm ${importResult.error ? 'bg-red-50 text-red-700' : 'bg-green-50 text-green-700'}`}>
                  {importResult.error
                    ? `Error: ${importResult.error}`
                    : `✓ Imported ${importResult.episodeCount} episodes to dataset ${importResult.datasetId}`}
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="validate">
          <Card>
            <CardHeader>
              <CardTitle>Validate Data Format</CardTitle>
              <CardDescription>Check if data matches RoboForce Titan specifications</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <Button onClick={handleValidate} disabled={processing}>
                <CheckCircle2 className="w-4 h-4 mr-2" /> Validate
              </Button>
              {validateResult && (
                <div className="space-y-2">
                  <div className={`p-3 rounded-lg text-sm ${validateResult.valid ? 'bg-green-50 text-green-700' : 'bg-red-50 text-red-700'}`}>
                    {validateResult.valid ? '✓ Data format is valid' : '✗ Validation failed'}
                  </div>
                  {validateResult.errors?.length > 0 && (
                    <div className="space-y-1">
                      {validateResult.errors.map((e: string, i: number) => (
                        <div key={i} className="flex items-center gap-2 text-sm text-red-600">
                          <AlertTriangle className="w-4 h-4" /> {e}
                        </div>
                      ))}
                    </div>
                  )}
                  {validateResult.warnings?.length > 0 && (
                    <div className="space-y-1">
                      {validateResult.warnings.map((w: string, i: number) => (
                        <div key={i} className="flex items-center gap-2 text-sm text-yellow-600">
                          <AlertTriangle className="w-4 h-4" /> {w}
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
