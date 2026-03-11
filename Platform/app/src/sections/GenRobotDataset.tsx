import { useState, useEffect } from 'react';
import { 
  RefreshCw, 
  Database, 
  Layers,
  Eye,
  BarChart3,
  Zap
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';

interface Sample {
  id: string;
  episode_id: string;
  frames: number;
  task: string;
  robot_type: string;
  actions_dim: number;
  observations: Record<string, any>;
}

interface GenRobotMetadata {
  name: string;
  source: string;
  description: string;
  samples: Sample[];
}

export default function GenRobotDataset() {
  const [metadata, setMetadata] = useState<GenRobotMetadata | null>(null);
  const [selectedSample, setSelectedSample] = useState<Sample | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadDataset();
  }, []);

  const loadDataset = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const response = await fetch('/api/datasets/genrobot');
      if (!response.ok) {
        throw new Error('Failed to load GenRobot dataset');
      }
      const data = await response.json();
      setMetadata(data);
      if (data.samples.length > 0) {
        setSelectedSample(data.samples[0]);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
      console.error('Failed to load dataset:', err);
    } finally {
      setIsLoading(false);
    }
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-12">
        <div className="animate-spin w-8 h-8 border-2 border-primary border-t-transparent rounded-full mr-4" />
        <p className="text-muted-foreground">Loading GenRobot dataset...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="space-y-4">
        <Card className="border-destructive">
          <CardContent className="pt-6">
            <p className="text-destructive font-medium">Error: {error}</p>
            <p className="text-sm text-muted-foreground mt-2">
              Please run `python download_data.py` in the backend directory first.
            </p>
            <Button onClick={loadDataset} className="mt-4">
              <RefreshCw className="w-4 h-4 mr-2" />
              Retry
            </Button>
          </CardContent>
        </Card>
      </div>
    );
  }

  if (!metadata) {
    return <div className="text-center py-12 text-muted-foreground">No data available</div>;
  }

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold">GenRobot Open Dataset</h2>
        <p className="text-muted-foreground">Advanced robot learning dataset visualization</p>
      </div>

      {/* Overview */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg flex items-center gap-2">
            <Database className="w-5 h-5" />
            Dataset Overview
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="space-y-2">
              <p className="text-sm text-muted-foreground">Total Samples</p>
              <p className="text-2xl font-bold">{metadata.samples.length}</p>
            </div>
            <div className="space-y-2">
              <p className="text-sm text-muted-foreground">Total Frames</p>
              <p className="text-2xl font-bold">
                {metadata.samples.reduce((sum, s) => sum + s.frames, 0)}
              </p>
            </div>
            <div className="space-y-2">
              <p className="text-sm text-muted-foreground">Data Source</p>
              <p className="text-sm font-medium truncate">GenRobot AI</p>
            </div>
            <div className="space-y-2">
              <p className="text-sm text-muted-foreground">Status</p>
              <Badge className="w-fit">Ready</Badge>
            </div>
          </div>
          <p className="text-sm text-muted-foreground mt-4">
            {metadata.description}
          </p>
        </CardContent>
      </Card>

      <Tabs defaultValue="samples" className="w-full">
        <TabsList className="grid w-full max-w-md grid-cols-3">
          <TabsTrigger value="samples">
            <Layers className="w-4 h-4 mr-2" />
            Samples
          </TabsTrigger>
          <TabsTrigger value="details">
            <Eye className="w-4 h-4 mr-2" />
            Details
          </TabsTrigger>
          <TabsTrigger value="stats">
            <BarChart3 className="w-4 h-4 mr-2" />
            Stats
          </TabsTrigger>
        </TabsList>

        {/* Samples Tab */}
        <TabsContent value="samples" className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {metadata.samples.map(sample => (
              <Card
                key={sample.id}
                className={`cursor-pointer transition-all hover:shadow-lg ${
                  selectedSample?.id === sample.id ? 'ring-2 ring-primary' : ''
                }`}
                onClick={() => setSelectedSample(sample)}
              >
                <CardContent className="pt-6">
                  <div className="space-y-3">
                    <div>
                      <p className="text-sm text-muted-foreground">Episode ID</p>
                      <p className="font-medium">{sample.episode_id}</p>
                    </div>
                    <div>
                      <p className="text-sm text-muted-foreground">Task</p>
                      <p className="font-medium">{sample.task}</p>
                    </div>
                    <div className="grid grid-cols-2 gap-4 text-sm">
                      <div>
                        <p className="text-muted-foreground">Frames</p>
                        <p className="font-semibold text-lg">{sample.frames}</p>
                      </div>
                      <div>
                        <p className="text-muted-foreground">Action Dims</p>
                        <p className="font-semibold text-lg">{sample.actions_dim}</p>
                      </div>
                    </div>
                    <Badge variant="secondary">{sample.robot_type}</Badge>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>

        {/* Details Tab */}
        <TabsContent value="details" className="space-y-4">
          {selectedSample ? (
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Sample: {selectedSample.id}</CardTitle>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <p className="text-sm text-muted-foreground">Episode ID</p>
                    <p className="font-medium">{selectedSample.episode_id}</p>
                  </div>
                  <div>
                    <p className="text-sm text-muted-foreground">Task</p>
                    <p className="font-medium">{selectedSample.task}</p>
                  </div>
                  <div>
                    <p className="text-sm text-muted-foreground">Total Frames</p>
                    <p className="font-medium text-xl">{selectedSample.frames}</p>
                  </div>
                  <div>
                    <p className="text-sm text-muted-foreground">Robot Type</p>
                    <p className="font-medium">{selectedSample.robot_type}</p>
                  </div>
                </div>

                <div>
                  <h4 className="font-semibold mb-3 flex items-center gap-2">
                    <Eye className="w-4 h-4" />
                    Observations
                  </h4>
                  <div className="space-y-2">
                    {Object.entries(selectedSample.observations).map(([obsName, obsData]) => (
                      <div key={obsName} className="p-3 bg-muted rounded-lg">
                        <p className="font-medium text-sm mb-1">{obsName}</p>
                        <div className="text-xs text-muted-foreground space-y-1">
                          <p>Shape: {JSON.stringify(obsData.shape)}</p>
                          <p>Type: {obsData.type}</p>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                <div>
                  <h4 className="font-semibold mb-3 flex items-center gap-2">
                    <Zap className="w-4 h-4" />
                    Action Configuration
                  </h4>
                  <div className="p-3 bg-muted rounded-lg">
                    <p className="text-sm">
                      <span className="text-muted-foreground">Action Dimensions:</span>
                      <span className="font-medium ml-2">{selectedSample.actions_dim}</span>
                    </p>
                  </div>
                </div>
              </CardContent>
            </Card>
          ) : (
            <Card>
              <CardContent className="pt-6">
                <p className="text-center text-muted-foreground">
                  Select a sample to view details
                </p>
              </CardContent>
            </Card>
          )}
        </TabsContent>

        {/* Stats Tab */}
        <TabsContent value="stats" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="text-lg">Dataset Statistics</CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <h4 className="font-semibold mb-4">Frame Distribution</h4>
                  <div className="space-y-3">
                    {metadata.samples.map(sample => (
                      <div key={sample.id}>
                        <div className="flex justify-between text-sm mb-1">
                          <span>{sample.episode_id}</span>
                          <span className="font-medium">{sample.frames}</span>
                        </div>
                        <div className="w-full bg-muted rounded-full h-2">
                          <div
                            className="bg-primary h-2 rounded-full"
                            style={{
                              width: `${(sample.frames / Math.max(...metadata.samples.map(s => s.frames))) * 100}%`
                            }}
                          />
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                <div>
                  <h4 className="font-semibold mb-4">Robot Types</h4>
                  <div className="space-y-2">
                    {Array.from(new Set(metadata.samples.map(s => s.robot_type))).map(type => (
                      <div key={type} className="flex items-center justify-between">
                        <Badge variant="secondary">{type}</Badge>
                        <span className="text-sm text-muted-foreground">
                          {metadata.samples.filter(s => s.robot_type === type).length} samples
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              <div className="bg-muted p-4 rounded-lg">
                <h4 className="font-semibold mb-3">Summary</h4>
                <div className="space-y-2 text-sm">
                  <p>
                    <span className="text-muted-foreground">Average Frames per Sample:</span>
                    <span className="font-medium ml-2">
                      {(metadata.samples.reduce((sum, s) => sum + s.frames, 0) / metadata.samples.length).toFixed(2)}
                    </span>
                  </p>
                  <p>
                    <span className="text-muted-foreground">Average Action Dimensions:</span>
                    <span className="font-medium ml-2">
                      {(metadata.samples.reduce((sum, s) => sum + s.actions_dim, 0) / metadata.samples.length).toFixed(2)}
                    </span>
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
