"use client";

import { useState, useEffect } from "react";
import { getExtractionResult, type ExtractionResult, type TopicInfo } from "@/lib/api";
import { TopicMap } from "./TopicMap";

type ViewMode = "list" | "map";

interface TopicTreeProps {
  extractionId: number;
  onRestart: () => void;
}

function TopicCard({ topic, depth = 0 }: { topic: TopicInfo; depth?: number }) {
  const [expanded, setExpanded] = useState(depth === 0);

  const hasChildren = topic.children && topic.children.length > 0;
  const hasComments = topic.example_comments && topic.example_comments.length > 0;
  const isExpandable = hasChildren || hasComments;

  return (
    <div
      className={`border-l-2 ${
        depth === 0 ? "border-amber-500" : "border-zinc-600"
      } pl-4`}
    >
      <div
        className={`p-4 rounded-lg ${
          depth === 0 ? "bg-zinc-800" : "bg-zinc-850"
        }`}
      >
        {/* Header */}
        <div className="flex items-start justify-between">
          <div className="flex-1">
            <div className="flex items-center gap-2">
              {isExpandable && (
                <button
                  onClick={() => setExpanded(!expanded)}
                  className="text-zinc-400 hover:text-zinc-200"
                >
                  {expanded ? "▼" : "▶"}
                </button>
              )}
              <h3 className={`font-medium text-zinc-100 ${depth === 0 ? "text-lg" : "text-base"}`}>
                {topic.generated_name}
              </h3>
            </div>
            {topic.parent_name && (
              <div className="text-xs text-zinc-500 mt-1">
                Part of: {topic.parent_name}
              </div>
            )}
          </div>
          <div className="flex gap-2 flex-wrap">
            <span className="px-2 py-1 bg-zinc-700 rounded text-sm text-zinc-300">
              {topic.count} comments
            </span>
            {topic.persistence != null && (
              <span
                className="px-2 py-1 rounded text-sm bg-zinc-700 text-zinc-300"
                title="Persistence: cluster stability in HDBSCAN hierarchy"
              >
                pers: {topic.persistence.toFixed(3)}
              </span>
            )}
            {topic.variance != null && (
              <span
                className="px-2 py-1 rounded text-sm bg-zinc-700 text-zinc-300"
                title="Variance: average variance across embedding dimensions"
              >
                var: {topic.variance.toFixed(4)}
              </span>
            )}
            {topic.mean_distance != null && (
              <span
                className="px-2 py-1 rounded text-sm bg-zinc-700 text-zinc-300"
                title="Mean distance: average distance to cluster centroid"
              >
                dist: {topic.mean_distance.toFixed(3)}
              </span>
            )}
          </div>
        </div>

        {/* Keywords */}
        {topic.top_words.length > 0 && (
          <div className="flex flex-wrap gap-1 mt-3">
            {topic.top_words.slice(0, 8).map((word, i) => (
              <span
                key={i}
                className="px-2 py-0.5 bg-zinc-700 rounded text-xs text-zinc-300"
              >
                {word}
              </span>
            ))}
          </div>
        )}

        {/* Example comments */}
        {expanded && hasComments && (
          <div className="mt-4 space-y-2">
            <div className="text-xs text-zinc-500 font-medium">
              Example comments ({topic.example_comments.length} samples):
            </div>
            {topic.example_comments.slice(0, depth === 0 ? 3 : 5).map((comment, i) => (
              <div
                key={i}
                className="p-2 bg-zinc-900 rounded text-sm text-zinc-400 italic"
              >
                "{comment.length > 200 ? comment.slice(0, 200) + "..." : comment}"
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Children */}
      {expanded && hasChildren && (
        <div className="mt-2 ml-4 space-y-2">
          {topic.children.map((child, i) => (
            <TopicCard key={i} topic={child} depth={depth + 1} />
          ))}
        </div>
      )}
    </div>
  );
}

export function TopicTree({ extractionId, onRestart }: TopicTreeProps) {
  const [result, setResult] = useState<ExtractionResult | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [showOutliersModal, setShowOutliersModal] = useState(false);
  const [viewMode, setViewMode] = useState<ViewMode>("list");

  // Check if map view is available (has viz coordinates)
  const hasMapData = result?.topics?.some(t => t.viz_x != null && t.viz_y != null) ?? false;

  useEffect(() => {
    const loadResult = async () => {
      try {
        const data = await getExtractionResult(extractionId);
        setResult(data);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to load results");
      } finally {
        setLoading(false);
      }
    };

    loadResult();
  }, [extractionId]);

  if (loading) {
    return (
      <div className="text-center py-12">
        <div className="animate-spin h-8 w-8 border-2 border-amber-500 border-t-transparent rounded-full mx-auto mb-4" />
        <p className="text-zinc-400">Loading results...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="text-center py-12">
        <div className="text-red-400 mb-4">{error}</div>
        <button
          onClick={onRestart}
          className="px-4 py-2 bg-zinc-700 hover:bg-zinc-600 text-zinc-100 rounded-lg"
        >
          Start Over
        </button>
      </div>
    );
  }

  if (!result) return null;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-semibold text-zinc-100">
            Topic Analysis Results
          </h2>
          <p className="text-zinc-400">
            {result.num_topics} topics found from {result.num_comments?.toLocaleString()} comments
          </p>
        </div>
        <div className="flex items-center gap-3">
          {/* View mode toggle */}
          {hasMapData && (
            <div className="flex bg-zinc-800 rounded-lg p-1">
              <button
                onClick={() => setViewMode("list")}
                className={`px-3 py-1.5 rounded text-sm font-medium transition-colors ${
                  viewMode === "list"
                    ? "bg-amber-600 text-white"
                    : "text-zinc-400 hover:text-zinc-200"
                }`}
              >
                📋 List
              </button>
              <button
                onClick={() => setViewMode("map")}
                className={`px-3 py-1.5 rounded text-sm font-medium transition-colors ${
                  viewMode === "map"
                    ? "bg-amber-600 text-white"
                    : "text-zinc-400 hover:text-zinc-200"
                }`}
              >
                🗺️ Map
              </button>
            </div>
          )}
          <button
            onClick={onRestart}
            className="px-4 py-2 bg-zinc-700 hover:bg-zinc-600 text-zinc-100 rounded-lg transition-colors"
          >
            New Analysis
          </button>
        </div>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-5 gap-4">
        <div className="p-4 bg-zinc-800 rounded-lg text-center">
          <div className="text-3xl font-bold text-amber-400">{result.num_topics}</div>
          <div className="text-sm text-zinc-500">Topics</div>
        </div>
        <div className="p-4 bg-zinc-800 rounded-lg text-center">
          <div className="text-3xl font-bold text-blue-400">
            {result.num_subtopics || 0}
          </div>
          <div className="text-sm text-zinc-500">Sub-topics</div>
        </div>
        <div className="p-4 bg-zinc-800 rounded-lg text-center">
          <div className="text-3xl font-bold text-green-400">
            {result.num_comments?.toLocaleString() || 0}
          </div>
          <div className="text-sm text-zinc-500">Comments</div>
        </div>
        <button
          onClick={() => result.outliers?.examples?.length && setShowOutliersModal(true)}
          className={`p-4 bg-zinc-800 rounded-lg text-center transition-colors ${
            result.outliers?.examples?.length 
              ? "hover:bg-zinc-700 cursor-pointer" 
              : "cursor-default"
          }`}
          title={result.outliers?.examples?.length ? "Click to explore outliers" : undefined}
        >
          <div className={`text-3xl font-bold ${
            result.outliers && result.outliers.percentage > 20 
              ? "text-red-400" 
              : "text-zinc-400"
          }`}>
            {result.outliers ? `${result.outliers.percentage.toFixed(1)}%` : "-"}
          </div>
          <div className="text-sm text-zinc-500">
            Outliers ({result.outliers?.count?.toLocaleString() || 0})
            {result.outliers?.examples?.length ? " 🔍" : ""}
          </div>
        </button>
        <div className="p-4 bg-zinc-800 rounded-lg text-center">
          <div className="text-3xl font-bold text-purple-400">
            {result.duration_seconds 
              ? result.duration_seconds < 60 
                ? `${Math.round(result.duration_seconds)}s`
                : `${Math.floor(result.duration_seconds / 60)}m${Math.round(result.duration_seconds % 60)}s`
              : "-"}
          </div>
          <div className="text-sm text-zinc-500">Duration</div>
        </div>
      </div>

      {/* Outliers Modal */}
      {showOutliersModal && result.outliers?.examples && (
        <div 
          className="fixed inset-0 bg-black/70 flex items-center justify-center z-50 p-4"
          onClick={() => setShowOutliersModal(false)}
        >
          <div 
            className="bg-zinc-900 rounded-xl max-w-3xl w-full max-h-[80vh] overflow-hidden flex flex-col"
            onClick={(e) => e.stopPropagation()}
          >
            {/* Modal Header */}
            <div className="p-4 border-b border-zinc-700 flex items-center justify-between">
              <div>
                <h3 className="text-xl font-semibold text-zinc-100">
                  Outlier Comments
                </h3>
                <p className="text-sm text-zinc-400">
                  {result.outliers.examples.length} random samples from {result.outliers.count.toLocaleString()} outliers
                </p>
              </div>
              <button
                onClick={() => setShowOutliersModal(false)}
                className="text-zinc-400 hover:text-zinc-200 text-2xl leading-none"
              >
                ×
              </button>
            </div>
            
            {/* Modal Body */}
            <div className="p-4 overflow-y-auto space-y-3">
              {result.outliers.examples.slice(0, 50).map((comment, i) => (
                <div
                  key={i}
                  className="p-3 bg-zinc-800 rounded-lg text-sm text-zinc-300"
                >
                  <span className="text-zinc-500 mr-2">#{i + 1}</span>
                  {comment.length > 300 ? comment.slice(0, 300) + "..." : comment}
                </div>
              ))}
            </div>
            
            {/* Modal Footer */}
            <div className="p-4 border-t border-zinc-700 text-center">
              <button
                onClick={() => setShowOutliersModal(false)}
                className="px-6 py-2 bg-zinc-700 hover:bg-zinc-600 text-zinc-100 rounded-lg transition-colors"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Topics - List View */}
      {viewMode === "list" && (
        <div className="space-y-4">
          {result.topics.map((topic, i) => (
            <TopicCard key={i} topic={topic} />
          ))}
        </div>
      )}

      {/* Topics - Map View (renders as full-page overlay) */}
      {viewMode === "map" && (
        <TopicMap topics={result.topics} onBack={() => setViewMode("list")} />
      )}

      {/* Generated timestamp */}
      {result.generated_at && (
        <div className="text-center text-sm text-zinc-500">
          Generated: {new Date(result.generated_at).toLocaleString()}
        </div>
      )}
    </div>
  );
}

