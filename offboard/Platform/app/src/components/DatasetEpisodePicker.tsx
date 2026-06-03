import { useState, useEffect } from 'react';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { api } from '@/services/api';
import type { Dataset, Episode } from '@/types';

interface DatasetEpisodePickerProps {
  onDatasetChange: (id: string, dataset: Dataset) => void;
  onEpisodeChange?: (id: string, episode: Episode) => void;
  showEpisode?: boolean;
  className?: string;
}

export default function DatasetEpisodePicker({
  onDatasetChange,
  onEpisodeChange,
  showEpisode = true,
  className,
}: DatasetEpisodePickerProps) {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [episodes, setEpisodes] = useState<Episode[]>([]);
  const [selectedDatasetId, setSelectedDatasetId] = useState('');
  const [selectedEpisodeId, setSelectedEpisodeId] = useState('');

  useEffect(() => {
    api.getDatasets().then(setDatasets).catch(console.error);
  }, []);

  const handleDatasetChange = async (id: string) => {
    setSelectedDatasetId(id);
    setSelectedEpisodeId('');
    setEpisodes([]);
    const ds = datasets.find(d => d.id === id);
    if (ds) onDatasetChange(id, ds);
    if (showEpisode) {
      try {
        const eps = await api.getDatasetEpisodes(id);
        setEpisodes(eps);
      } catch (e) {
        console.error(e);
      }
    }
  };

  const handleEpisodeChange = (id: string) => {
    setSelectedEpisodeId(id);
    const ep = episodes.find(e => e.id === id);
    if (ep && onEpisodeChange) onEpisodeChange(id, ep);
  };

  return (
    <div className={`flex flex-wrap gap-4 ${className ?? ''}`}>
      <div className="space-y-2 min-w-[200px]">
        <Label>数据集</Label>
        <Select value={selectedDatasetId} onValueChange={handleDatasetChange}>
          <SelectTrigger>
            <SelectValue placeholder="选择数据集..." />
          </SelectTrigger>
          <SelectContent>
            {datasets.map(ds => (
              <SelectItem key={ds.id} value={ds.id}>{ds.name}</SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>
      {showEpisode && (
        <div className="space-y-2 min-w-[200px]">
          <Label>Episode</Label>
          <Select
            value={selectedEpisodeId}
            onValueChange={handleEpisodeChange}
            disabled={!selectedDatasetId || episodes.length === 0}
          >
            <SelectTrigger>
              <SelectValue placeholder="选择 Episode..." />
            </SelectTrigger>
            <SelectContent>
              {episodes.map(ep => (
                <SelectItem key={ep.id} value={ep.id}>{(ep as any).name || ep.id}</SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      )}
    </div>
  );
}
