"use client";

import { useState, useMemo, useCallback, useRef, useEffect } from "react";
import type { TopicInfo } from "@/lib/api";

interface TopicMapProps {
  topics: TopicInfo[];
  onBack?: () => void;
}

const TOPIC_COLORS = [
  { main: "#f59e0b", dark: "#92400e", light: "#fcd34d" },
  { main: "#ef4444", dark: "#991b1b", light: "#fca5a5" },
  { main: "#8b5cf6", dark: "#5b21b6", light: "#c4b5fd" },
  { main: "#3b82f6", dark: "#1e40af", light: "#93c5fd" },
  { main: "#06b6d4", dark: "#155e75", light: "#67e8f9" },
  { main: "#10b981", dark: "#065f46", light: "#6ee7b7" },
  { main: "#f97316", dark: "#9a3412", light: "#fdba74" },
  { main: "#ec4899", dark: "#9d174d", light: "#f9a8d4" },
  { main: "#84cc16", dark: "#3f6212", light: "#bef264" },
  { main: "#6366f1", dark: "#3730a3", light: "#a5b4fc" },
];

function getColorSet(index: number) {
  return TOPIC_COLORS[index % TOPIC_COLORS.length];
}

const PANEL_WIDTH = 384;
const ZOOM_DURATION = 700;
const ZOOM_LEVEL = 2.5; // Reduced zoom level

type ColorSet = { main: string; dark: string; light: string };

// Text that stays readable regardless of zoom
interface ScaledTextProps {
  x: number;
  y: number;
  scale: number;
  children: React.ReactNode;
  fontSize: number;
  fill: string;
  fontWeight?: string;
  opacity?: number;
  stroke?: string;
  strokeWidth?: number;
}

function ScaledText({ x, y, scale, children, fontSize, fill, fontWeight = "400", opacity = 1, stroke, strokeWidth }: ScaledTextProps) {
  // Counter-scale to keep text readable
  const counterScale = 1 / scale;
  return (
    <text
      x={x}
      y={y}
      textAnchor="middle"
      dominantBaseline="middle"
      fill={fill}
      fontSize={fontSize * counterScale}
      fontWeight={fontWeight}
      opacity={opacity}
      stroke={stroke}
      strokeWidth={strokeWidth ? strokeWidth * counterScale : undefined}
      strokeLinejoin={stroke ? "round" : undefined}
      style={{ pointerEvents: "none" }}
    >
      {children}
    </text>
  );
}

// Bubble component
interface BubbleProps {
  topic: TopicInfo;
  x: number;
  y: number;
  radius: number;
  colorSet: ColorSet;
  isZoomed: boolean;
  isOther: boolean;
  zoomProgress: number;
  currentScale: number;
  onClick: () => void;
  canClick: boolean;
  showSubtopicIndicator: boolean;
}

function Bubble({ topic, x, y, radius, colorSet, isZoomed, isOther, zoomProgress, currentScale, onClick, canClick, showSubtopicIndicator }: BubbleProps) {
  const [hovered, setHovered] = useState(false);
  
  const opacity = isOther ? Math.max(0.06, 1 - zoomProgress * 0.94) : 1;
  const showLabel = zoomProgress < 0.4;
  const label = topic.generated_name.length > 40 ? topic.generated_name.slice(0, 40) + "…" : topic.generated_name;
  
  return (
    <g style={{ opacity }} onClick={(e) => { e.stopPropagation(); if (canClick) onClick(); }}
      onMouseEnter={() => setHovered(true)} onMouseLeave={() => setHovered(false)}>
      
      {hovered && canClick && (
        <circle cx={x} cy={y} r={radius + 4} fill="none" stroke={colorSet.light} strokeWidth={2 / currentScale} opacity="0.6" />
      )}
      
      <circle cx={x} cy={y} r={radius} fill={colorSet.main}
        stroke={isZoomed ? colorSet.light : "transparent"} strokeWidth={isZoomed ? 2 / currentScale : 0}
        style={{ cursor: canClick ? "pointer" : "default" }} />
      
      {showLabel && (
        <>
          <ScaledText x={x} y={y - 2} scale={currentScale} fontSize={13} fill="#000" fontWeight="700" stroke="#000" strokeWidth={4}>{label}</ScaledText>
          <ScaledText x={x} y={y - 2} scale={currentScale} fontSize={13} fill="#fff" fontWeight="700">{label}</ScaledText>
          <ScaledText x={x} y={y + 14} scale={currentScale} fontSize={11} fill="#000" fontWeight="600" stroke="#000" strokeWidth={3}>{topic.count.toLocaleString()}</ScaledText>
          <ScaledText x={x} y={y + 14} scale={currentScale} fontSize={11} fill="#fff" fontWeight="600" opacity={0.9}>{topic.count.toLocaleString()}</ScaledText>
          {showSubtopicIndicator && topic.children?.length > 0 && (
            <>
              <ScaledText x={x} y={y + 28} scale={currentScale} fontSize={10} fill="#000" stroke="#000" strokeWidth={3}>▼ {topic.children.length}</ScaledText>
              <ScaledText x={x} y={y + 28} scale={currentScale} fontSize={10} fill={colorSet.light} fontWeight="500">▼ {topic.children.length}</ScaledText>
            </>
          )}
        </>
      )}
    </g>
  );
}

// Subtopic bubble
interface SubtopicBubbleProps {
  topic: TopicInfo;
  x: number;
  y: number;
  radius: number;
  colorSet: ColorSet;
  zoomProgress: number;
  currentScale: number;
  isSelected: boolean;
  onClick: () => void;
}

function SubtopicBubble({ topic, x, y, radius, colorSet, zoomProgress, currentScale, isSelected, onClick }: SubtopicBubbleProps) {
  const [hovered, setHovered] = useState(false);
  
  const opacity = Math.max(0, (zoomProgress - 0.5) * 2);
  if (opacity < 0.05) return null;
  
  const label = topic.generated_name.length > 35 ? topic.generated_name.slice(0, 35) + "…" : topic.generated_name;
  
  return (
    <g style={{ opacity }} onClick={(e) => { e.stopPropagation(); onClick(); }}
      onMouseEnter={() => setHovered(true)} onMouseLeave={() => setHovered(false)}>
      
      {(hovered || isSelected) && (
        <circle cx={x} cy={y} r={radius + 3} fill="none" stroke={colorSet.light} strokeWidth={2 / currentScale} opacity="0.6" />
      )}
      
      <circle cx={x} cy={y} r={radius} fill={colorSet.dark}
        stroke={isSelected ? "#fff" : "transparent"} strokeWidth={isSelected ? 2 / currentScale : 0}
        style={{ cursor: "pointer" }} />
      
      {opacity > 0.5 && (
        <>
          <ScaledText x={x} y={y - 2} scale={currentScale} fontSize={12} fill="#000" fontWeight="700" stroke="#000" strokeWidth={3}>{label}</ScaledText>
          <ScaledText x={x} y={y - 2} scale={currentScale} fontSize={12} fill="#fff" fontWeight="700">{label}</ScaledText>
          <ScaledText x={x} y={y + 12} scale={currentScale} fontSize={10} fill="#fff" fontWeight="600" opacity={0.85}>{topic.count.toLocaleString()}</ScaledText>
        </>
      )}
    </g>
  );
}

export function TopicMap({ topics, onBack }: TopicMapProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [dimensions, setDimensions] = useState({ width: 800, height: 600 });
  const [zoomedIndex, setZoomedIndex] = useState<number | null>(null);
  const [selectedSubtopic, setSelectedSubtopic] = useState<TopicInfo | null>(null);
  const [zoomProgress, setZoomProgress] = useState(0);
  const [isAnimating, setIsAnimating] = useState(false);
  const animationRef = useRef<number | null>(null);

  // Content area (the SVG container already excludes the panel via CSS)
  const centerX = dimensions.width / 2;
  const centerY = dimensions.height / 2;

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    const observer = new ResizeObserver((entries) => {
      const entry = entries[0];
      if (entry) setDimensions({ width: entry.contentRect.width, height: entry.contentRect.height });
    });
    observer.observe(container);
    return () => observer.disconnect();
  }, []);

  // Root bubbles - fill the entire content area
  const rootBubbles = useMemo(() => {
    const validTopics = topics.filter(t => t.viz_x != null && t.viz_y != null);
    if (!validTopics.length) return [];
    
    const minX = Math.min(...validTopics.map(t => t.viz_x!));
    const maxX = Math.max(...validTopics.map(t => t.viz_x!));
    const minY = Math.min(...validTopics.map(t => t.viz_y!));
    const maxY = Math.max(...validTopics.map(t => t.viz_y!));
    const maxCount = Math.max(...topics.map(t => t.count));
    const rangeX = maxX - minX || 1;
    const rangeY = maxY - minY || 1;
    
    // Use most of the available space
    const padding = 80;
    const availableWidth = dimensions.width - padding * 2;
    const availableHeight = dimensions.height - padding * 2;

    return topics.map((topic, i) => {
      let x: number, y: number;
      if (topic.viz_x == null || topic.viz_y == null) {
        const angle = (2 * Math.PI * i) / topics.length - Math.PI / 2;
        const r = Math.min(availableWidth, availableHeight) * 0.4;
        x = centerX + Math.cos(angle) * r;
        y = centerY + Math.sin(angle) * r;
      } else {
        // Map to available space, centered
        x = padding + ((topic.viz_x - minX) / rangeX) * availableWidth;
        y = padding + ((topic.viz_y - minY) / rangeY) * availableHeight;
      }
      const baseR = 45, scaleR = 40;
      const radius = baseR + (Math.log(topic.count + 1) / Math.log(maxCount + 1)) * scaleR;
      return { topic, x, y, radius, colorSet: getColorSet(i), index: i };
    });
  }, [topics, dimensions.width, dimensions.height, centerX, centerY]);

  const zoomedBubble = zoomedIndex !== null ? rootBubbles[zoomedIndex] : null;
  const zoomedTopic = zoomedBubble?.topic || null;
  
  // Subtopics positioned inside parent
  const subtopicBubbles = useMemo(() => {
    if (!zoomedBubble || !zoomedTopic?.children?.length) return [];
    const children = zoomedTopic.children;
    const parentX = zoomedBubble.x;
    const parentY = zoomedBubble.y;
    const parentR = zoomedBubble.radius;
    const innerRadius = parentR * 0.85;
    
    const withCoords = children.filter(c => c.viz_x != null && c.viz_y != null);
    const useCircular = withCoords.length < children.length * 0.5;
    let minX = 0, maxX = 1, minY = 0, maxY = 1;
    if (!useCircular && withCoords.length > 1) {
      minX = Math.min(...withCoords.map(c => c.viz_x!));
      maxX = Math.max(...withCoords.map(c => c.viz_x!));
      minY = Math.min(...withCoords.map(c => c.viz_y!));
      maxY = Math.max(...withCoords.map(c => c.viz_y!));
    }
    const maxCount = Math.max(...children.map(c => c.count));
    const rangeX = maxX - minX || 1, rangeY = maxY - minY || 1;

    return children.map((child, i) => {
      let dx: number, dy: number;
      if (useCircular || child.viz_x == null || child.viz_y == null) {
        const angle = (2 * Math.PI * i) / children.length - Math.PI / 2;
        dx = Math.cos(angle) * innerRadius * 0.6;
        dy = Math.sin(angle) * innerRadius * 0.6;
      } else {
        const normX = (child.viz_x - minX) / rangeX - 0.5;
        const normY = (child.viz_y - minY) / rangeY - 0.5;
        dx = normX * innerRadius * 1.4;
        dy = normY * innerRadius * 1.4;
      }
      const x = parentX + dx;
      const y = parentY + dy;
      const baseR = 12, scaleR = 10;
      const radius = baseR + (Math.log(child.count + 1) / Math.log(maxCount + 1)) * scaleR;
      return { topic: child, x, y, radius };
    });
  }, [zoomedBubble, zoomedTopic]);

  // Current scale for counter-scaling text
  const currentScale = 1 + (ZOOM_LEVEL - 1) * zoomProgress;

  // Camera transform - zoom centered on the bubble
  const cameraTransform = useMemo(() => {
    if (!zoomedBubble || zoomProgress === 0) {
      return { scale: 1, translateX: 0, translateY: 0 };
    }
    
    // Zoom in, keeping the clicked bubble at the center of the view
    const scale = currentScale;
    
    // Translate so bubble ends up at view center
    // At scale 1, we want translateX = 0. At full zoom, we want bubble at center.
    const translateX = (centerX - zoomedBubble.x) * zoomProgress * scale;
    const translateY = (centerY - zoomedBubble.y) * zoomProgress * scale;
    
    return { scale, translateX, translateY };
  }, [zoomedBubble, zoomProgress, currentScale, centerX, centerY]);

  const animateTo = useCallback((targetProgress: number) => {
    if (animationRef.current) cancelAnimationFrame(animationRef.current);
    setIsAnimating(true);
    const startProgress = zoomProgress;
    const startTime = performance.now();
    const animate = (currentTime: number) => {
      const elapsed = currentTime - startTime;
      const t = Math.min(elapsed / ZOOM_DURATION, 1);
      const eased = 1 - Math.pow(1 - t, 3);
      setZoomProgress(startProgress + (targetProgress - startProgress) * eased);
      if (t < 1) animationRef.current = requestAnimationFrame(animate);
      else { setIsAnimating(false); animationRef.current = null; }
    };
    animationRef.current = requestAnimationFrame(animate);
  }, [zoomProgress]);

  const handleBubbleClick = useCallback((index: number) => {
    const topic = topics[index];
    setZoomedIndex(index);
    setSelectedSubtopic(null);
    if (topic.children?.length) {
      animateTo(1);
    }
  }, [topics, animateTo]);

  const handleSubtopicClick = useCallback((topic: TopicInfo) => setSelectedSubtopic(topic), []);

  const handleZoomOut = useCallback(() => {
    setSelectedSubtopic(null);
    animateTo(0);
    setTimeout(() => { setZoomedIndex(null); }, ZOOM_DURATION + 50);
  }, [animateTo]);

  const handleBackgroundClick = useCallback(() => {
    if (selectedSubtopic) setSelectedSubtopic(null);
    else if (zoomProgress > 0.5) handleZoomOut();
  }, [selectedSubtopic, zoomProgress, handleZoomOut]);

  const hasVizData = topics.some(t => t.viz_x != null && t.viz_y != null);
  if (!hasVizData) {
    return (
      <div className="fixed inset-0 z-50 bg-zinc-950 flex items-center justify-center">
        <div className="text-center text-zinc-500">
          <div className="text-6xl mb-6">🗺️</div>
          <p className="text-xl">No visualization data</p>
          {onBack && <button onClick={onBack} className="mt-6 px-4 py-2 bg-zinc-800 rounded-lg">Go Back</button>}
        </div>
      </div>
    );
  }

  const currentColorSet = zoomedBubble?.colorSet || null;
  const detailTopic = selectedSubtopic || zoomedTopic;
  const isZoomed = zoomProgress > 0.5;

  return (
    <div className="fixed inset-0 z-50 bg-zinc-950 overflow-hidden flex">
      {/* Canvas - takes remaining space */}
      <div ref={containerRef} className="flex-1 relative" onClick={handleBackgroundClick}>
        <svg width="100%" height="100%" className="absolute inset-0 overflow-visible">
          <defs>
            <pattern id="grid" width="50" height="50" patternUnits="userSpaceOnUse">
              <path d="M 50 0 L 0 0 0 50" fill="none" stroke="#1f1f23" strokeWidth="1" />
            </pattern>
          </defs>
          <rect width="100%" height="100%" fill="url(#grid)" />
          
          {/* Camera transform - zoom centered on clicked bubble */}
          <g style={{
            transformOrigin: `${centerX}px ${centerY}px`,
            transform: `translate(${cameraTransform.translateX}px, ${cameraTransform.translateY}px) scale(${cameraTransform.scale})`,
          }}>
            {rootBubbles.map((b) => (
              <Bubble key={b.topic.id} topic={b.topic} x={b.x} y={b.y} radius={b.radius} colorSet={b.colorSet}
                isZoomed={b.index === zoomedIndex} isOther={zoomedIndex !== null && b.index !== zoomedIndex}
                zoomProgress={zoomProgress} currentScale={currentScale} onClick={() => handleBubbleClick(b.index)}
                canClick={!isAnimating && zoomProgress < 0.3} showSubtopicIndicator={true} />
            ))}
            
            {zoomedBubble && currentColorSet && subtopicBubbles.map((b) => (
              <SubtopicBubble key={b.topic.id} topic={b.topic} x={b.x} y={b.y} radius={b.radius}
                colorSet={currentColorSet} zoomProgress={zoomProgress} currentScale={currentScale}
                isSelected={selectedSubtopic?.id === b.topic.id} onClick={() => handleSubtopicClick(b.topic)} />
            ))}
          </g>
        </svg>
        
        {/* Top bar overlay */}
        <div className="absolute top-0 left-0 right-0 flex items-center justify-between px-5 py-4 pointer-events-none">
          <div className="flex items-center gap-3 pointer-events-auto">
            <button onClick={isZoomed ? handleZoomOut : undefined} disabled={isAnimating}
              className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-all ${!isZoomed ? "bg-amber-600 text-white" : "bg-zinc-800/90 text-zinc-300 hover:bg-zinc-700"}`}>
              🗺️ Topics
            </button>
            {isZoomed && zoomedTopic && (
              <>
                <span className="text-zinc-600">›</span>
                <span className="px-3 py-1.5 rounded-lg text-sm font-medium text-white" style={{ backgroundColor: currentColorSet?.main }}>
                  {zoomedTopic.generated_name.length > 30 ? zoomedTopic.generated_name.slice(0, 30) + "…" : zoomedTopic.generated_name}
                </span>
              </>
            )}
          </div>
          {onBack && (
            <button onClick={onBack} className="px-4 py-1.5 text-sm bg-zinc-800/90 hover:bg-zinc-700 text-zinc-300 rounded-lg pointer-events-auto">
              Exit Map
            </button>
          )}
        </div>
        
        {/* Bottom hint */}
        <div className="absolute bottom-6 left-6 text-sm text-zinc-600">
          {zoomProgress < 0.1 && "Click a bubble to zoom in"}
          {zoomProgress > 0.9 && !selectedSubtopic && "Click outside to zoom out"}
        </div>
      </div>

      {/* Panel - fixed width on right */}
      <div className="w-96 bg-zinc-900 border-l border-zinc-800 flex flex-col flex-shrink-0">
        {detailTopic ? (
          <>
            <div className="p-5 border-b border-zinc-800" style={{ background: currentColorSet ? `linear-gradient(135deg, ${currentColorSet.dark}50 0%, transparent 100%)` : undefined }}>
              <button onClick={() => setSelectedSubtopic(null)} className="absolute top-4 right-4 w-8 h-8 flex items-center justify-center text-zinc-500 hover:text-white hover:bg-zinc-800 rounded-full">×</button>
              <h2 className="text-lg font-semibold text-white pr-8 leading-tight">{detailTopic.generated_name}</h2>
              <div className="flex gap-3 mt-2 text-sm text-zinc-400">
                <span>{detailTopic.count.toLocaleString()} comments</span>
                {detailTopic.children?.length > 0 && <span>• {detailTopic.children.length} subtopics</span>}
              </div>
            </div>
            <div className="flex-1 overflow-y-auto p-5 space-y-6">
              {detailTopic.top_words?.length > 0 && (
                <section>
                  <h3 className="text-xs font-semibold text-zinc-500 uppercase tracking-wider mb-3">Keywords</h3>
                  <div className="flex flex-wrap gap-2">
                    {detailTopic.top_words.map((word, i) => <span key={i} className="px-2.5 py-1 bg-zinc-800 hover:bg-zinc-700 rounded-full text-sm text-zinc-300">{word}</span>)}
                  </div>
                </section>
              )}
              {(detailTopic.persistence != null || detailTopic.variance != null) && (
                <section>
                  <h3 className="text-xs font-semibold text-zinc-500 uppercase tracking-wider mb-3">Metrics</h3>
                  <div className="grid grid-cols-3 gap-2">
                    {detailTopic.persistence != null && <div className="p-2.5 bg-zinc-800/50 rounded-lg text-center"><div className="text-base font-bold text-amber-400">{detailTopic.persistence.toFixed(2)}</div><div className="text-xs text-zinc-500">Pers.</div></div>}
                    {detailTopic.variance != null && <div className="p-2.5 bg-zinc-800/50 rounded-lg text-center"><div className="text-base font-bold text-blue-400">{detailTopic.variance.toFixed(3)}</div><div className="text-xs text-zinc-500">Var.</div></div>}
                    {detailTopic.mean_distance != null && <div className="p-2.5 bg-zinc-800/50 rounded-lg text-center"><div className="text-base font-bold text-emerald-400">{detailTopic.mean_distance.toFixed(2)}</div><div className="text-xs text-zinc-500">Dist.</div></div>}
                  </div>
                </section>
              )}
              {detailTopic.example_comments?.length > 0 && (
                <section>
                  <h3 className="text-xs font-semibold text-zinc-500 uppercase tracking-wider mb-3">Sample Comments</h3>
                  <div className="space-y-2">
                    {detailTopic.example_comments.slice(0, 8).map((comment, i) => (
                      <div key={i} className="p-3 bg-zinc-800/30 rounded-lg text-sm text-zinc-400 border-l-2" style={{ borderColor: currentColorSet?.main || "#525252" }}>
                        {comment.length > 200 ? comment.slice(0, 200) + "…" : comment}
                      </div>
                    ))}
                  </div>
                </section>
              )}
            </div>
          </>
        ) : (
          <div className="flex-1 flex items-center justify-center text-zinc-600">
            <div className="text-center px-8"><div className="text-4xl mb-4">👈</div><p>Click a topic to see details</p></div>
          </div>
        )}
      </div>
    </div>
  );
}
