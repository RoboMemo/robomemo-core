import { useState, useEffect } from 'react';
import { 
  Database, 
  Layers, 
  Tag, 
  Cpu,
  HardDrive,
  TrendingUp,
  Activity,
  Calendar
} from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  Legend, 
  ResponsiveContainer,
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell
} from 'recharts';
import { api } from '@/services/api';
import type { PlatformStats, TimelineEntry } from '@/types';

const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884d8', '#82ca9d'];

export default function PlatformStats() {
  const [stats, setStats] = useState<PlatformStats | null>(null);
  const [timeline, setTimeline] = useState<TimelineEntry[]>([]);
  const [datasetStats, setDatasetStats] = useState<any>(null);
  const [annotationStats, setAnnotationStats] = useState<any>(null);
  const [simulatorStats, setSimulatorStats] = useState<any>(null);

  useEffect(() => {
    loadStats();
    const interval = setInterval(loadStats, 30000);
    return () => clearInterval(interval);
  }, []);

  const loadStats = async () => {
    try {
      const [statsData, timelineData, dsStats, annStats, simStats] = await Promise.all([
        api.getStats(),
        api.getTimeline(7),
        api.getDatasetStats(),
        api.getAnnotationStats(),
        api.getSimulatorStats()
      ]);
      setStats(statsData);
      setTimeline(timelineData);
      setDatasetStats(dsStats);
      setAnnotationStats(annStats);
      setSimulatorStats(simStats);
    } catch (error) {
      console.error('Failed to load stats:', error);
    }
  };

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const formatData = (data: Record<string, number>) => {
    return Object.entries(data).map(([name, value]) => ({ name, value }));
  };

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold">Platform Statistics</h2>
        <p className="text-muted-foreground">Overview of your embodied data platform</p>
      </div>

      {/* Overview Cards */}
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <Database className="w-8 h-8 text-blue-500" />
              <div className="text-right">
                <p className="text-2xl font-bold">{stats?.datasets || 0}</p>
                <p className="text-xs text-muted-foreground">Datasets</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <Layers className="w-8 h-8 text-green-500" />
              <div className="text-right">
                <p className="text-2xl font-bold">{stats?.episodes || 0}</p>
                <p className="text-xs text-muted-foreground">Episodes</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <Activity className="w-8 h-8 text-yellow-500" />
              <div className="text-right">
                <p className="text-2xl font-bold">{stats?.frames || 0}</p>
                <p className="text-xs text-muted-foreground">Frames</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <Tag className="w-8 h-8 text-purple-500" />
              <div className="text-right">
                <p className="text-2xl font-bold">{stats?.annotations || 0}</p>
                <p className="text-xs text-muted-foreground">Annotations</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <Cpu className="w-8 h-8 text-red-500" />
              <div className="text-right">
                <p className="text-2xl font-bold">{stats?.simulators || 0}</p>
                <p className="text-xs text-muted-foreground">Simulators</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <HardDrive className="w-8 h-8 text-orange-500" />
              <div className="text-right">
                <p className="text-2xl font-bold">{formatFileSize(stats?.totalStorage || 0)}</p>
                <p className="text-xs text-muted-foreground">Storage</p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Charts */}
      <Tabs defaultValue="timeline" className="space-y-4">
        <TabsList>
          <TabsTrigger value="timeline">
            <TrendingUp className="w-4 h-4 mr-2" />
            Timeline
          </TabsTrigger>
          <TabsTrigger value="datasets">
            <Database className="w-4 h-4 mr-2" />
            Datasets
          </TabsTrigger>
          <TabsTrigger value="annotations">
            <Tag className="w-4 h-4 mr-2" />
            Annotations
          </TabsTrigger>
          <TabsTrigger value="simulators">
            <Cpu className="w-4 h-4 mr-2" />
            Simulators
          </TabsTrigger>
        </TabsList>

        <TabsContent value="timeline">
          <Card>
            <CardHeader>
              <CardTitle className="text-lg flex items-center gap-2">
                <Calendar className="w-5 h-5" />
                Activity Timeline (Last 7 Days)
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="h-80">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={timeline}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="date" />
                    <YAxis />
                    <Tooltip />
                    <Legend />
                    <Line type="monotone" dataKey="datasets" stroke="#8884d8" name="Datasets" />
                    <Line type="monotone" dataKey="episodes" stroke="#82ca9d" name="Episodes" />
                    <Line type="monotone" dataKey="annotations" stroke="#ffc658" name="Annotations" />
                    <Line type="monotone" dataKey="collections" stroke="#ff7300" name="Collections" />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="datasets">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Datasets by Format</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                      <Pie
                        data={datasetStats?.byFormat ? formatData(datasetStats.byFormat) : []}
                        cx="50%"
                        cy="50%"
                        labelLine={false}
                        label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                        outerRadius={80}
                        fill="#8884d8"
                        dataKey="value"
                      >
                        {datasetStats?.byFormat && Object.entries(datasetStats.byFormat).map((_, index) => (
                          <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                        ))}
                      </Pie>
                      <Tooltip />
                    </PieChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Datasets by Robot Type</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={datasetStats?.byRobotType ? formatData(datasetStats.byRobotType) : []}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="name" />
                      <YAxis />
                      <Tooltip />
                      <Bar dataKey="value" fill="#8884d8" />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="annotations">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Annotations by Type</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={annotationStats?.byType ? formatData(annotationStats.byType) : []}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="name" />
                      <YAxis />
                      <Tooltip />
                      <Bar dataKey="value" fill="#82ca9d" />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Verification Status</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                      <Pie
                        data={[
                          { name: 'Verified', value: annotationStats?.verified || 0 },
                          { name: 'Unverified', value: annotationStats?.unverified || 0 }
                        ]}
                        cx="50%"
                        cy="50%"
                        labelLine={false}
                        label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                        outerRadius={80}
                        fill="#8884d8"
                        dataKey="value"
                      >
                        <Cell fill="#22c55e" />
                        <Cell fill="#ef4444" />
                      </Pie>
                      <Tooltip />
                    </PieChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="simulators">
          <Card>
            <CardHeader>
              <CardTitle className="text-lg">Simulator Status</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="p-4 bg-muted rounded-lg text-center">
                  <p className="text-3xl font-bold text-green-500">{simulatorStats?.running || 0}</p>
                  <p className="text-sm text-muted-foreground">Running</p>
                </div>
                <div className="p-4 bg-muted rounded-lg text-center">
                  <p className="text-3xl font-bold text-blue-500">{simulatorStats?.available || 0}</p>
                  <p className="text-sm text-muted-foreground">Available</p>
                </div>
                <div className="p-4 bg-muted rounded-lg text-center">
                  <p className="text-3xl font-bold">{simulatorStats?.total || 0}</p>
                  <p className="text-sm text-muted-foreground">Total</p>
                </div>
              </div>
              
              {simulatorStats?.byType && (
                <div className="mt-6">
                  <h4 className="text-sm font-medium mb-4">By Type</h4>
                  <div className="h-64">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={formatData(simulatorStats.byType)}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="name" />
                        <YAxis />
                        <Tooltip />
                        <Bar dataKey="value" fill="#8884d8" />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>

      {/* Recent Activity */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg">Recent Datasets</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-2">
            {datasetStats?.recent?.map((dataset: any) => (
              <div key={dataset.id} className="flex items-center justify-between p-3 border rounded-lg">
                <div className="flex items-center gap-3">
                  <Database className="w-5 h-5 text-muted-foreground" />
                  <div>
                    <p className="font-medium">{dataset.name}</p>
                    <p className="text-sm text-muted-foreground">
                      {dataset.format} • {dataset.robotType}
                    </p>
                  </div>
                </div>
                <div className="text-right">
                  <Badge variant="secondary">{dataset.episodeCount} episodes</Badge>
                  <p className="text-xs text-muted-foreground mt-1">
                    {new Date(dataset.createdAt).toLocaleDateString()}
                  </p>
                </div>
              </div>
            ))}
            {!datasetStats?.recent?.length && (
              <p className="text-center text-muted-foreground py-4">No recent datasets</p>
            )}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
