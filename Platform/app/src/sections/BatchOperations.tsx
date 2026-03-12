import { useState, useEffect } from 'react';
import { Upload, Download, Zap, Users, FileJson, Loader2 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import api from '@/services/api';
import type { Dataset, User } from '@/types';

export default function BatchOperations() {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [users, setUsers] = useState<User[]>([]);
  const [loading, setLoading] = useState(true);
  const [importJson, setImportJson] = useState('');
  const [importDataset, setImportDataset] = useState('');
  const [importResult, setImportResult] = useState<any>(null);
  const [assignEpIds, setAssignEpIds] = useState('');
  const [assignUsers, setAssignUsers] = useState<string[]>([]);
  const [assignType, setAssignType] = useState('annotation');
  const [assignResult, setAssignResult] = useState<any>(null);
  const [exportDataset, setExportDataset] = useState('');
  const [exportResult, setExportResult] = useState<any>(null);
  const [autoEpIds, setAutoEpIds] = useState('');
  const [autoResult, setAutoResult] = useState<any>(null);
  const [processing, setProcessing] = useState(false);

  useEffect(() => {
    loadData();
  }, []);

  const loadData = async () => {
    try {
      const [ds, us] = await Promise.all([
        api.getDatasets().catch(() => []),
        api.getUsers().catch(() => []),
      ]);
      setDatasets(ds);
      setUsers(us);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleImport = async () => {
    if (!importDataset || !importJson) return;
    setProcessing(true);
    try {
      const episodes = JSON.parse(importJson);
      const result = await api.batchImportEpisodes(importDataset, episodes);
      setImportResult(result);
    } catch (err: any) {
      setImportResult({ error: err.message });
    } finally {
      setProcessing(false);
    }
  };

  const handleAssign = async () => {
    if (!assignEpIds || assignUsers.length === 0) return;
    setProcessing(true);
    try {
      const episodeIds = assignEpIds.split(',').map(s => s.trim()).filter(Boolean);
      const result = await api.batchAssignTasks({
        episodeIds,
        assignees: assignUsers,
        type: assignType,
        datasetId: importDataset || undefined,
      });
      setAssignResult(result);
    } catch (err: any) {
      setAssignResult({ error: err.message });
    } finally {
      setProcessing(false);
    }
  };

  const handleExport = async () => {
    setProcessing(true);
    try {
      const result = await api.batchExport({
        datasetId: exportDataset || undefined,
        format: 'json',
        includeAnnotations: true,
      });
      setExportResult(result);
    } catch (err: any) {
      setExportResult({ error: err.message });
    } finally {
      setProcessing(false);
    }
  };

  const handleAutoAnnotate = async () => {
    if (!autoEpIds) return;
    setProcessing(true);
    try {
      const episodeIds = autoEpIds.split(',').map(s => s.trim()).filter(Boolean);
      const result = await api.batchAutoAnnotate(episodeIds);
      setAutoResult(result);
    } catch (err: any) {
      setAutoResult({ error: err.message });
    } finally {
      setProcessing(false);
    }
  };

  if (loading) return <div className="text-center py-12 text-muted-foreground">Loading...</div>;

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold">Batch Operations</h1>
        <p className="text-muted-foreground mt-1">Import, assign, export, and auto-annotate in bulk</p>
      </div>

      <Tabs defaultValue="import">
        <TabsList className="grid grid-cols-4 w-full">
          <TabsTrigger value="import"><Upload className="w-4 h-4 mr-1" /> Import</TabsTrigger>
          <TabsTrigger value="assign"><Users className="w-4 h-4 mr-1" /> Assign</TabsTrigger>
          <TabsTrigger value="export"><Download className="w-4 h-4 mr-1" /> Export</TabsTrigger>
          <TabsTrigger value="auto"><Zap className="w-4 h-4 mr-1" /> Auto-Annotate</TabsTrigger>
        </TabsList>

        {/* Import Tab */}
        <TabsContent value="import">
          <Card>
            <CardHeader><CardTitle>Batch Import Episodes</CardTitle></CardHeader>
            <CardContent className="space-y-4">
              <div>
                <Label>Dataset</Label>
                <Select value={importDataset} onValueChange={setImportDataset}>
                  <SelectTrigger><SelectValue placeholder="Select dataset" /></SelectTrigger>
                  <SelectContent>
                    {datasets.map(d => <SelectItem key={d.id} value={d.id}>{d.name}</SelectItem>)}
                  </SelectContent>
                </Select>
              </div>
              <div>
                <Label>Episodes JSON</Label>
                <Textarea
                  placeholder={`[{"name": "ep_001", "skill": "pick", "frameCount": 300}, ...]`}
                  value={importJson}
                  onChange={(e) => setImportJson(e.target.value)}
                  rows={8}
                />
              </div>
              <Button onClick={handleImport} disabled={processing || !importDataset || !importJson}>
                {processing ? <><Loader2 className="w-4 h-4 mr-2 animate-spin" /> Importing...</> : <><Upload className="w-4 h-4 mr-2" /> Import</>}
              </Button>
              {importResult && (
                <div className={`p-3 rounded-lg text-sm ${importResult.error ? 'bg-red-50 text-red-700' : 'bg-green-50 text-green-700'}`}>
                  {importResult.error ? `Error: ${importResult.error}` : `Imported ${importResult.imported} episodes`}
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* Assign Tab */}
        <TabsContent value="assign">
          <Card>
            <CardHeader><CardTitle>Batch Task Assignment</CardTitle></CardHeader>
            <CardContent className="space-y-4">
              <div>
                <Label>Episode IDs (comma-separated)</Label>
                <Textarea
                  placeholder="ep_001, ep_002, ep_003"
                  value={assignEpIds}
                  onChange={(e) => setAssignEpIds(e.target.value)}
                  rows={3}
                />
              </div>
              <div>
                <Label>Assign To</Label>
                <div className="flex flex-wrap gap-2 mt-1">
                  {users.map(u => (
                    <Badge
                      key={u.id}
                      variant={assignUsers.includes(u.id) ? 'default' : 'outline'}
                      className="cursor-pointer"
                      onClick={() => setAssignUsers(prev =>
                        prev.includes(u.id) ? prev.filter(id => id !== u.id) : [...prev, u.id]
                      )}
                    >
                      {u.name}
                    </Badge>
                  ))}
                </div>
              </div>
              <div>
                <Label>Task Type</Label>
                <Select value={assignType} onValueChange={setAssignType}>
                  <SelectTrigger><SelectValue /></SelectTrigger>
                  <SelectContent>
                    <SelectItem value="annotation">Annotation</SelectItem>
                    <SelectItem value="review">Review</SelectItem>
                    <SelectItem value="vqa">VQA</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <Button onClick={handleAssign} disabled={processing || !assignEpIds || assignUsers.length === 0}>
                {processing ? <><Loader2 className="w-4 h-4 mr-2 animate-spin" /> Assigning...</> : <><Users className="w-4 h-4 mr-2" /> Assign Tasks</>}
              </Button>
              {assignResult && (
                <div className={`p-3 rounded-lg text-sm ${assignResult.error ? 'bg-red-50 text-red-700' : 'bg-green-50 text-green-700'}`}>
                  {assignResult.error ? `Error: ${assignResult.error}` : `Created ${assignResult.tasksCreated} tasks`}
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* Export Tab */}
        <TabsContent value="export">
          <Card>
            <CardHeader><CardTitle>Batch Export</CardTitle></CardHeader>
            <CardContent className="space-y-4">
              <div>
                <Label>Dataset (empty = all)</Label>
                <Select value={exportDataset} onValueChange={setExportDataset}>
                  <SelectTrigger><SelectValue placeholder="All datasets" /></SelectTrigger>
                  <SelectContent>
                    <SelectItem value="">All</SelectItem>
                    {datasets.map(d => <SelectItem key={d.id} value={d.id}>{d.name}</SelectItem>)}
                  </SelectContent>
                </Select>
              </div>
              <Button onClick={handleExport} disabled={processing}>
                {processing ? <><Loader2 className="w-4 h-4 mr-2 animate-spin" /> Exporting...</> : <><Download className="w-4 h-4 mr-2" /> Export</>}
              </Button>
              {exportResult && (
                <div className="p-3 bg-green-50 rounded-lg text-sm text-green-700">
                  Exported {exportResult.datasets} datasets, {exportResult.episodes} episodes, {exportResult.annotations} annotations
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* Auto-Annotate Tab */}
        <TabsContent value="auto">
          <Card>
            <CardHeader><CardTitle>Batch Auto-Annotation</CardTitle></CardHeader>
            <CardContent className="space-y-4">
              <div>
                <Label>Episode IDs (comma-separated)</Label>
                <Textarea
                  placeholder="ep_001, ep_002, ep_003"
                  value={autoEpIds}
                  onChange={(e) => setAutoEpIds(e.target.value)}
                  rows={3}
                />
              </div>
              <Button onClick={handleAutoAnnotate} disabled={processing || !autoEpIds}>
                {processing ? <><Loader2 className="w-4 h-4 mr-2 animate-spin" /> Processing...</> : <><Zap className="w-4 h-4 mr-2" /> Auto-Annotate</>}
              </Button>
              {autoResult && (
                <div className={`p-3 rounded-lg text-sm ${autoResult.error ? 'bg-red-50 text-red-700' : 'bg-green-50 text-green-700'}`}>
                  {autoResult.error ? `Error: ${autoResult.error}` : `Job ${autoResult.id}: ${autoResult.status}`}
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
