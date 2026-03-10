import type { 
  Dataset, Episode, Annotation, Collection, 
  Simulator, SensorConfig, AugmentationModel, ExportFormat,
  PlatformStats, TimelineEntry,
  StructuredVQAAnalysis, VQAAnnotationRecord, VLMProvider
} from '@/types';

const API_BASE = 'http://localhost:3001/api';

class ApiService {
  private async fetch<T>(endpoint: string, options?: RequestInit): Promise<T> {
    const response = await fetch(`${API_BASE}${endpoint}`, {
      headers: {
        'Content-Type': 'application/json',
      },
      ...options,
    });
    
    if (!response.ok) {
      const error = await response.json().catch(() => ({ error: 'Unknown error' }));
      throw new Error(error.error || `HTTP ${response.status}`);
    }
    
    return response.json();
  }

  // Datasets
  async getDatasets(): Promise<Dataset[]> {
    return this.fetch('/datasets');
  }

  async getDataset(id: string): Promise<Dataset> {
    return this.fetch(`/datasets/${id}`);
  }

  async createDataset(dataset: Partial<Dataset>): Promise<Dataset> {
    return this.fetch('/datasets', {
      method: 'POST',
      body: JSON.stringify(dataset),
    });
  }

  async updateDataset(id: string, updates: Partial<Dataset>): Promise<Dataset> {
    return this.fetch(`/datasets/${id}`, {
      method: 'PUT',
      body: JSON.stringify(updates),
    });
  }

  async deleteDataset(id: string): Promise<void> {
    return this.fetch(`/datasets/${id}`, { method: 'DELETE' });
  }

  async getDatasetEpisodes(datasetId: string): Promise<Episode[]> {
    return this.fetch(`/datasets/${datasetId}/episodes`);
  }

  // Collections
  async getCollections(): Promise<Collection[]> {
    return this.fetch('/collections');
  }

  async getCollection(id: string): Promise<Collection> {
    return this.fetch(`/collections/${id}`);
  }

  async createCollection(collection: Partial<Collection>): Promise<Collection> {
    return this.fetch('/collections', {
      method: 'POST',
      body: JSON.stringify(collection),
    });
  }

  async addEpisodeToCollection(collectionId: string, episodeId: string): Promise<Collection> {
    return this.fetch(`/collections/${collectionId}/episodes`, {
      method: 'POST',
      body: JSON.stringify({ episodeId }),
    });
  }

  // Annotations
  async getAnnotationsByFrame(frameId: string): Promise<Annotation[]> {
    return this.fetch(`/annotations/frame/${frameId}`);
  }

  async getAnnotationsByEpisode(episodeId: string): Promise<Annotation[]> {
    return this.fetch(`/annotations/episode/${episodeId}`);
  }

  async createAnnotation(annotation: Partial<Annotation>): Promise<Annotation> {
    return this.fetch('/annotations', {
      method: 'POST',
      body: JSON.stringify(annotation),
    });
  }

  async updateAnnotation(id: string, updates: Partial<Annotation>): Promise<Annotation> {
    return this.fetch(`/annotations/${id}`, {
      method: 'PUT',
      body: JSON.stringify(updates),
    });
  }

  async deleteAnnotation(id: string): Promise<void> {
    return this.fetch(`/annotations/${id}`, { method: 'DELETE' });
  }

  async batchCreateAnnotations(annotations: Partial<Annotation>[]): Promise<Annotation[]> {
    return this.fetch('/annotations/batch', {
      method: 'POST',
      body: JSON.stringify({ annotations }),
    });
  }

  // Simulators
  async getSimulators(): Promise<Simulator[]> {
    return this.fetch('/simulators');
  }

  async getSimulator(id: string): Promise<Simulator> {
    return this.fetch(`/simulators/${id}`);
  }

  async startSimulator(id: string, config: { scene: string; robot: string; sensors: string[] }): Promise<Simulator> {
    return this.fetch(`/simulators/${id}/start`, {
      method: 'POST',
      body: JSON.stringify(config),
    });
  }

  async stopSimulator(id: string): Promise<Simulator> {
    return this.fetch(`/simulators/${id}/stop`, { method: 'POST' });
  }

  async resetSimulator(id: string): Promise<Simulator> {
    return this.fetch(`/simulators/${id}/reset`, { method: 'POST' });
  }

  async stepSimulator(id: string, action?: any): Promise<any> {
    return this.fetch(`/simulators/${id}/step`, {
      method: 'POST',
      body: JSON.stringify({ action }),
    });
  }

  async getSimulatorScenes(id: string): Promise<any[]> {
    return this.fetch(`/simulators/${id}/scenes`);
  }

  async getSimulatorRobots(id: string): Promise<any[]> {
    return this.fetch(`/simulators/${id}/robots`);
  }

  // Sensors
  async getSensorConfigs(): Promise<SensorConfig[]> {
    return this.fetch('/sensors/configs');
  }

  async createSensorConfig(config: Partial<SensorConfig>): Promise<SensorConfig> {
    return this.fetch('/sensors/configs', {
      method: 'POST',
      body: JSON.stringify(config),
    });
  }

  async getSensorTypes(): Promise<any[]> {
    return this.fetch('/sensors/types');
  }

  async getDefaultSensorConfigs(): Promise<Record<string, SensorConfig>> {
    return this.fetch('/sensors/defaults');
  }

  // Augmentation
  async getAugmentationModels(): Promise<AugmentationModel[]> {
    return this.fetch('/augmentation/models');
  }

  async generateAugmentedData(modelId: string, datasetId: string, config: any): Promise<any> {
    return this.fetch('/augmentation/generate', {
      method: 'POST',
      body: JSON.stringify({ modelId, datasetId, config }),
    });
  }

  async getAugmentationJobStatus(jobId: string): Promise<any> {
    return this.fetch(`/augmentation/jobs/${jobId}`);
  }

  async crossEmbodimentTransfer(params: any): Promise<any> {
    return this.fetch('/augmentation/cross-embodiment', {
      method: 'POST',
      body: JSON.stringify(params),
    });
  }

  async crossContextGeneration(params: any): Promise<any> {
    return this.fetch('/augmentation/cross-context', {
      method: 'POST',
      body: JSON.stringify(params),
    });
  }

  // Export
  async getExportFormats(): Promise<ExportFormat[]> {
    return this.fetch('/export/formats');
  }

  async exportDataset(datasetId: string, format: string, options?: any): Promise<any> {
    return this.fetch(`/export/dataset/${datasetId}`, {
      method: 'POST',
      body: JSON.stringify({ format, options }),
    });
  }

  // Stats
  async getStats(): Promise<PlatformStats> {
    return this.fetch('/stats');
  }

  async getDatasetStats(): Promise<any> {
    return this.fetch('/stats/datasets');
  }

  async getAnnotationStats(): Promise<any> {
    return this.fetch('/stats/annotations');
  }

  async getSimulatorStats(): Promise<any> {
    return this.fetch('/stats/simulators');
  }

  async getTimeline(days?: number): Promise<TimelineEntry[]> {
    const query = days ? `?days=${days}` : '';
    return this.fetch(`/stats/timeline${query}`);
  }

  // Auto-Annotation (VLM-based)
  async getAutoAnnotationModels(): Promise<any[]> {
    return this.fetch('/autoannotation/models');
  }

  async segmentVideo(videoPath: string, modelId: string, options?: any): Promise<any> {
    return this.fetch('/autoannotation/segment', {
      method: 'POST',
      body: JSON.stringify({ videoPath, modelId, options }),
    });
  }

  async segmentVideoLocal(videoPath: string): Promise<any> {
    return this.fetch('/vlm/local/segment', {
      method: 'POST',
      body: JSON.stringify({ videoPath }),
    });
  }

  async queryVideo(videoPath: string, query: string, modelId: string, frameRange?: any, options?: any): Promise<any> {
    return this.fetch('/autoannotation/query', {
      method: 'POST',
      body: JSON.stringify({ videoPath, query, modelId, frameRange, options }),
    });
  }

  async queryVideoLocal(videoPath: string, query: string): Promise<any> {
    return this.fetch('/vlm/local/query', {
      method: 'POST',
      body: JSON.stringify({ videoPath, query }),
    });
  }

  async batchAutoAnnotate(episodeIds: string[], modelId: string, annotationType: string, options?: any): Promise<any> {
    return this.fetch('/autoannotation/batch', {
      method: 'POST',
      body: JSON.stringify({ episodeIds, modelId, annotationType, options }),
    });
  }

  async getAutoAnnotationJobStatus(jobId: string): Promise<any> {
    return this.fetch(`/autoannotation/jobs/${jobId}`);
  }

  async searchVideos(datasetId: string, query: string, modelId: string, topK?: number): Promise<any> {
    return this.fetch('/autoannotation/search', {
      method: 'POST',
      body: JSON.stringify({ datasetId, query, modelId, topK }),
    });
  }

  async summarizeVideo(videoPath: string, modelId: string, summaryType: string, options?: any): Promise<any> {
    return this.fetch('/autoannotation/summarize', {
      method: 'POST',
      body: JSON.stringify({ videoPath, modelId, summaryType, options }),
    });
  }

  async summarizeVideoLocal(videoPath: string, summaryType: string): Promise<any> {
    return this.fetch('/vlm/local/summarize', {
      method: 'POST',
      body: JSON.stringify({ videoPath, summaryType }),
    });
  }

  async compareVideos(videoPaths: string[], modelId: string, comparisonType: string): Promise<any> {
    return this.fetch('/autoannotation/compare', {
      method: 'POST',
      body: JSON.stringify({ videoPaths, modelId, comparisonType }),
    });
  }

  // ========== Structured VQA API ==========

  async getVLMProviders(): Promise<VLMProvider[]> {
    return this.fetch('/vlm/providers');
  }

  async getLocalVLMModels(): Promise<{
    available: boolean;
    models: { name: string; size: number; modified_at?: string }[];
    error?: string;
  }> {
    return this.fetch('/vlm/local-models');
  }

  async runStructuredVQAAnalysis(params: {
    videoPath: string;
    provider: string;
    apiKey?: string;
    numFrames?: number;
    model?: string;
  }): Promise<{ success: boolean; annotationId: string; analysis: StructuredVQAAnalysis }> {
    return this.fetch('/vlm/structured-analysis', {
      method: 'POST',
      body: JSON.stringify(params),
    });
  }

  async getStructuredAnalyses(): Promise<VQAAnnotationRecord[]> {
    return this.fetch('/vlm/structured-analyses');
  }

  async getStructuredAnalysis(id: string): Promise<VQAAnnotationRecord> {
    return this.fetch(`/vlm/structured-analysis/${id}`);
  }
}

export const api = new ApiService();
export default api;
