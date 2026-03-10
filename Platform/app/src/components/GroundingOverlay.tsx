/**
 * GroundingOverlay — Renders bounding boxes on a frame image
 * Used to visually anchor VQA annotations to specific regions in video frames.
 */
import { useRef, useState } from 'react';
import { Badge } from '@/components/ui/badge';
import { Eye, Maximize2 } from 'lucide-react';
import type { GroundingBBox, VisualGrounding } from '@/types';

// ─── Color palette for bounding boxes ─────────────────────────────────────────
const BBOX_COLORS = [
  'rgba(59, 130, 246, 0.8)',   // blue
  'rgba(239, 68, 68, 0.8)',    // red
  'rgba(16, 185, 129, 0.8)',   // green
  'rgba(245, 158, 11, 0.8)',   // amber
  'rgba(168, 85, 247, 0.8)',   // purple
  'rgba(236, 72, 153, 0.8)',   // pink
];

// ─── BBox overlay on a single frame ───────────────────────────────────────────
interface FrameOverlayProps {
  /** URL/path to the frame image */
  frameUrl: string;
  /** Bounding boxes to render (normalized 0-1 coords) */
  bboxes?: GroundingBBox[];
  /** Frame index for display */
  frameIndex?: number;
  /** Timestamp string */
  timestamp?: string;
  /** Size variant */
  size?: 'sm' | 'md' | 'lg';
  /** Click handler */
  onClick?: () => void;
}

export function FrameOverlay({
  frameUrl,
  bboxes = [],
  frameIndex,
  timestamp,
  size = 'md',
  onClick,
}: FrameOverlayProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [hovered, setHovered] = useState<number | null>(null);

  const sizeClasses = {
    sm: 'w-32 h-20',
    md: 'w-48 h-32',
    lg: 'w-72 h-48',
  };

  return (
    <div
      ref={containerRef}
      className={`relative ${sizeClasses[size]} rounded-lg overflow-hidden border bg-black cursor-pointer group`}
      onClick={onClick}
      onMouseLeave={() => setHovered(null)}
    >
      {/* Frame image */}
      <img
        src={frameUrl}
        alt={`Frame ${frameIndex ?? ''}`}
        className="absolute inset-0 w-full h-full object-cover"
        loading="lazy"
      />

      {/* Bounding boxes */}
      {bboxes.map((bbox, i) => (
        <div
          key={i}
          className="absolute transition-all duration-150"
          style={{
            left: `${bbox.x * 100}%`,
            top: `${bbox.y * 100}%`,
            width: `${bbox.width * 100}%`,
            height: `${bbox.height * 100}%`,
            border: `2px solid ${BBOX_COLORS[i % BBOX_COLORS.length]}`,
            backgroundColor: hovered === i
              ? `${BBOX_COLORS[i % BBOX_COLORS.length].replace('0.8', '0.15')}`
              : 'transparent',
          }}
          onMouseEnter={() => setHovered(i)}
        >
          {/* Label */}
          {bbox.label && (
            <span
              className="absolute -top-5 left-0 text-[10px] px-1 py-0.5 rounded whitespace-nowrap font-medium text-white"
              style={{ backgroundColor: BBOX_COLORS[i % BBOX_COLORS.length] }}
            >
              {bbox.label}
            </span>
          )}
        </div>
      ))}

      {/* Frame info badge */}
      <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/70 to-transparent p-1.5 flex items-end justify-between">
        <div className="flex items-center gap-1">
          {frameIndex !== undefined && (
            <span className="text-[10px] font-mono text-white/80">#{frameIndex}</span>
          )}
          {timestamp && (
            <span className="text-[10px] font-mono text-white/60">{timestamp}</span>
          )}
        </div>
        {bboxes.length > 0 && (
          <Badge variant="secondary" className="text-[9px] h-4 px-1 bg-black/40 text-white/80">
            {bboxes.length} region{bboxes.length > 1 ? 's' : ''}
          </Badge>
        )}
      </div>

      {/* Hover expand icon */}
      <div className="absolute top-1 right-1 opacity-0 group-hover:opacity-100 transition-opacity">
        <Maximize2 className="w-3.5 h-3.5 text-white drop-shadow" />
      </div>
    </div>
  );
}

// ─── Grounding Badge — inline indicator for visual evidence ───────────────────
interface GroundingBadgeProps {
  grounding: VisualGrounding;
  /** Frame image URLs (from metadata) for thumbnail display */
  frameImageUrls?: string[];
  /** Category color class */
  colorClass?: string;
  /** Compact mode (just icon + frame count) */
  compact?: boolean;
  /** Click to expand/view */
  onClickView?: (grounding: VisualGrounding) => void;
}

export function GroundingBadge({
  grounding,
  frameImageUrls,
  colorClass = 'text-blue-500',
  compact = false,
  onClickView,
}: GroundingBadgeProps) {
  if (!grounding) return null;

  const frameCount = grounding.frame_indices?.length ?? 0;
  const hasBboxes = (grounding.bboxes?.length ?? 0) > 0;
  const conf = Math.round(grounding.confidence * 100);

  return (
    <button
      className={`inline-flex items-center gap-1 px-1.5 py-0.5 rounded-full text-[10px] font-medium
        border transition-colors hover:bg-accent/50
        ${hasBboxes ? 'border-green-300 dark:border-green-800' : 'border-muted'}
        ${colorClass}`}
      onClick={(e) => {
        e.stopPropagation();
        onClickView?.(grounding);
      }}
      title={grounding.description}
    >
      <Eye className="w-2.5 h-2.5" />
      {!compact && (
        <>
          <span>
            {frameCount} frame{frameCount !== 1 ? 's' : ''}
          </span>
          {hasBboxes && (
            <span className="text-green-600 dark:text-green-400">
              +{grounding.bboxes!.length} bbox
            </span>
          )}
          <span className={conf >= 80 ? 'text-green-600' : conf >= 60 ? 'text-yellow-600' : 'text-red-500'}>
            {conf}%
          </span>
        </>
      )}
      {compact && <span>{frameCount}f</span>}
    </button>
  );
}

// ─── Grounding Detail Panel — expanded view of visual evidence ────────────────
interface GroundingDetailProps {
  grounding: VisualGrounding;
  frameImageUrls?: string[];
  onClose?: () => void;
}

export function GroundingDetail({
  grounding,
  frameImageUrls = [],
  onClose,
}: GroundingDetailProps) {
  if (!grounding) return null;

  return (
    <div className="p-3 rounded-lg border bg-card space-y-2.5">
      {/* Description */}
      <div className="flex items-start justify-between gap-2">
        <p className="text-xs text-muted-foreground leading-relaxed flex-1">
          {grounding.description}
        </p>
        <Badge
          variant="outline"
          className={`text-[10px] shrink-0 ${
            grounding.confidence >= 0.8
              ? 'text-green-600 border-green-300'
              : grounding.confidence >= 0.6
              ? 'text-yellow-600 border-yellow-300'
              : 'text-red-500 border-red-300'
          }`}
        >
          {Math.round(grounding.confidence * 100)}% conf
        </Badge>
      </div>

      {/* Frame thumbnails with bboxes */}
      {grounding.frame_indices?.length > 0 && (
        <div className="flex gap-2 overflow-x-auto pb-1">
          {grounding.frame_indices.map((frameIdx, i) => {
            const url = frameImageUrls?.[frameIdx];
            // Distribute bboxes across frames or show all on each
            const frameBboxes = grounding.bboxes ?? [];

            if (url) {
              return (
                <FrameOverlay
                  key={frameIdx}
                  frameUrl={url}
                  bboxes={frameBboxes}
                  frameIndex={frameIdx}
                  timestamp={grounding.timestamps?.[i]}
                  size="sm"
                />
              );
            }

            // Fallback: text-only frame reference
            return (
              <div
                key={frameIdx}
                className="w-32 h-20 rounded border bg-muted flex flex-col items-center justify-center text-xs text-muted-foreground shrink-0"
              >
                <span className="font-mono font-bold">#{frameIdx}</span>
                {grounding.timestamps?.[i] && (
                  <span className="text-[10px]">{grounding.timestamps[i]}</span>
                )}
                {frameBboxes.length > 0 && (
                  <span className="text-[10px] text-green-600 mt-0.5">
                    {frameBboxes.length} bbox
                  </span>
                )}
              </div>
            );
          })}
        </div>
      )}

      {/* Timestamps inline */}
      {grounding.timestamps?.length > 0 && (
        <div className="flex flex-wrap gap-1">
          {grounding.timestamps.map((ts, i) => (
            <Badge key={i} variant="secondary" className="text-[10px] font-mono">
              {ts}
            </Badge>
          ))}
        </div>
      )}
    </div>
  );
}

// ─── Temporal Consistency Alert ───────────────────────────────────────────────
interface ConsistencyAlertProps {
  conflicts: Array<{
    category_a: string;
    category_b: string;
    claim_a: string;
    claim_b: string;
    timestamp_a: string;
    timestamp_b: string;
    description: string;
  }>;
}

export function TemporalConsistencyAlert({ conflicts }: ConsistencyAlertProps) {
  if (!conflicts || conflicts.length === 0) return null;

  return (
    <div className="p-2.5 rounded-lg bg-yellow-50 dark:bg-yellow-950 border border-yellow-200 dark:border-yellow-800">
      <div className="flex items-center gap-1.5 mb-2">
        <span className="text-yellow-600 text-xs font-semibold">
          ⚠️ Temporal Consistency Issues ({conflicts.length})
        </span>
      </div>
      <div className="space-y-1.5">
        {conflicts.map((c, i) => (
          <div key={i} className="text-xs text-yellow-700 dark:text-yellow-300 p-1.5 rounded bg-yellow-100 dark:bg-yellow-900">
            <div className="flex gap-2 mb-0.5">
              <Badge variant="outline" className="text-[10px]">{c.category_a}</Badge>
              <span>↔</span>
              <Badge variant="outline" className="text-[10px]">{c.category_b}</Badge>
            </div>
            <p>{c.description}</p>
          </div>
        ))}
      </div>
    </div>
  );
}
