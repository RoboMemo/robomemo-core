import { useState, useEffect } from 'react';
import { Upload, CheckCircle2, AlertTriangle, Monitor, Cpu } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';
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

  // Upload state
  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const [uploadDatasetName, setUploadDatasetName] = useState('');
  const [uploadEpisodeName, setUploadEpisodeName] = useState('');
  const [uploadDescription, setUploadDescription] = useState('');
  const [uploadOrderTitle, setUploadOrderTitle] = useState('');
  const [uploadResult, setUploadResult] = useState<any>(null);
  const [uploading, setUploading] = useState(false);

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

  const handleUpload = async () => {
    if (!uploadFile || !uploadDatasetName || !uploadEpisodeName) return;
    setUploading(true);
    setUploadResult(null);
    try {
      const formData = new FormData();
      formData.append('video', uploadFile);
      formData.append('datasetName', uploadDatasetName);
      formData.append('episodeName', uploadEpisodeName);
      formData.append('description', uploadDescription);
      formData.append('orderTitle', uploadOrderTitle || `VQA - ${uploadEpisodeName}`);
      formData.append('preset', 'titan_standard');

      const result = await api.uploadRoboForceVideo(formData);
      setUploadResult(result);
    } catch (err: any) {
      setUploadResult({ error: err.message });
    } finally {
      setUploading(false);
    }
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

      <Tabs defaultValue="upload">
        <TabsList>
          <TabsTrigger value="upload">Upload Video</TabsTrigger>
          <TabsTrigger value="import">Import JSON</TabsTrigger>
          <TabsTrigger value="validate">Validate</TabsTrigger>
        </TabsList>

        <TabsContent value="upload">
          <Card>
            <CardHeader>
              <CardTitle>Upload RoboForce Video</CardTitle>
              <CardDescription>Upload video file and auto-create order + VQA task</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <Label>Video File</Label>
                <Input
                  type="file"
                  accept="video/*"
                  onChange={(e) => setUploadFile(e.target.files?.[0] || null)}
                />
              </div>
              <div>
                <Label>Dataset Name</Label>
                <Input
                  placeholder="RoboForce Titan Dataset"
                  value={uploadDatasetName}
                  onChange={(e) => setUploadDatasetName(e.target.value)}
                />
              </div>
              <div>
                <Label>Episode Name</Label>
                <Input
                  placeholder="screw_drive_01"
                  value={uploadEpisodeName}
                  onChange={(e) => setUploadEpisodeName(e.target.value)}
                />
              </div>
              <div>
                <Label>Description (optional)</Label>
                <Textarea
                  placeholder="M4 screw insertion at 5Nm torque"
                  value={uploadDescription}
                  onChange={(e) => setUploadDescription(e.target.value)}
                  rows={2}
                />
              </div>
              <div>
                <Label>Order Title (optional)</Label>
                <Input
                  placeholder="Auto-generated if empty"
                  value={uploadOrderTitle}
                  onChange={(e) => setUploadOrderTitle(e.target.value)}
                />
              </div>
              <Button
                onClick={handleUpload}
                disabled={uploading || !uploadFile || !uploadDatasetName || !uploadEpisodeName}
              >
                <Upload className="w-4 h-4 mr-2" />
                {uploading ? 'Uploading...' : 'Upload & Create Order'}
              </Button>
              {uploadResult && (
                <div
                  className={`p-3 rounded-lg text-sm ${
                    uploadResult.error ? 'bg-red-50 text-red-700' : 'bg-green-50 text-green-700'
                  }`}
                >
                  {uploadResult.error ? (
                    `Error: ${uploadResult.error}`
                  ) : (
                    <div className="space-y-1">
                      <div>✓ Upload successful!</div>
                      <div className="text-xs">Dataset: {uploadResult.datasetId}</div>
                      <div className="text-xs">Episode: {uploadResult.episodeId}</div>
                      <div className="text-xs">Order: {uploadResult.orderId}</div>
                      <div className="text-xs">Task: {uploadResult.taskId}</div>
                    </div>
                  )}
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

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
