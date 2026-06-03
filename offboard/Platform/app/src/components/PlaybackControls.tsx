import { Play, Pause, SkipBack } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Slider } from '@/components/ui/slider';

interface PlaybackControlsProps {
  isPlaying: boolean;
  currentFrame: number;
  totalFrames: number;
  onPlay: () => void;
  onPause: () => void;
  onSeek: (frame: number) => void;
}

export default function PlaybackControls({
  isPlaying,
  currentFrame,
  totalFrames,
  onPlay,
  onPause,
  onSeek,
}: PlaybackControlsProps) {
  return (
    <div className="flex items-center gap-3">
      <Button variant="ghost" size="icon" onClick={() => onSeek(0)}>
        <SkipBack className="w-4 h-4" />
      </Button>
      <Button variant="ghost" size="icon" onClick={isPlaying ? onPause : onPlay}>
        {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
      </Button>
      <div className="flex-1">
        <Slider
          min={0}
          max={Math.max(totalFrames - 1, 0)}
          step={1}
          value={[currentFrame]}
          onValueChange={([v]) => onSeek(v)}
        />
      </div>
      <span className="text-sm text-muted-foreground whitespace-nowrap">
        {currentFrame} / {totalFrames}
      </span>
    </div>
  );
}
