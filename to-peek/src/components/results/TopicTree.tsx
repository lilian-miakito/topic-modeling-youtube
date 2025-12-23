"use client";

import { useState, useEffect } from "react";
import { getExtractionResult, type ExtractionResult, type TopicInfo } from "@/lib/api";

interface TopicTreeProps {
  extractionId: number;
  onRestart: () => void;
}

function TopicCard({ topic, depth = 0 }: { topic: TopicInfo; depth?: number }) {
  const [expanded, setExpanded] = useState(depth === 0);

  const hasChildren = topic.children && topic.children.length > 0;

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
              {hasChildren && (
                <button
                  onClick={() => setExpanded(!expanded)}
                  className="text-zinc-400 hover:text-zinc-200"
                >
                  {expanded ? "▼" : "▶"}
                </button>
              )}
              <h3 className="text-lg font-medium text-zinc-100">
                {topic.generated_name}
              </h3>
            </div>
            {topic.parent_name && (
              <div className="text-xs text-zinc-500 mt-1">
                Part of: {topic.parent_name}
              </div>
            )}
          </div>
          <div className="flex gap-2">
            <span className="px-2 py-1 bg-zinc-700 rounded text-sm text-zinc-300">
              {topic.count} comments
            </span>
            {topic.silhouette !== undefined && (
              <span
                className={`px-2 py-1 rounded text-sm ${
                  topic.silhouette >= 0.15
                    ? "bg-green-900/50 text-green-400"
                    : "bg-amber-900/50 text-amber-400"
                }`}
              >
                sil: {topic.silhouette.toFixed(2)}
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
        {expanded && topic.example_comments.length > 0 && (
          <div className="mt-4 space-y-2">
            <div className="text-xs text-zinc-500 font-medium">
              Example comments:
            </div>
            {topic.example_comments.slice(0, 3).map((comment, i) => (
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
        <button
          onClick={onRestart}
          className="px-4 py-2 bg-zinc-700 hover:bg-zinc-600 text-zinc-100 rounded-lg transition-colors"
        >
          New Analysis
        </button>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-3 gap-4">
        <div className="p-4 bg-zinc-800 rounded-lg text-center">
          <div className="text-3xl font-bold text-amber-400">{result.num_topics}</div>
          <div className="text-sm text-zinc-500">Topics</div>
        </div>
        <div className="p-4 bg-zinc-800 rounded-lg text-center">
          <div className="text-3xl font-bold text-blue-400">
            {result.num_hierarchical || 0}
          </div>
          <div className="text-sm text-zinc-500">Hierarchical</div>
        </div>
        <div className="p-4 bg-zinc-800 rounded-lg text-center">
          <div className="text-3xl font-bold text-green-400">
            {result.num_comments?.toLocaleString() || 0}
          </div>
          <div className="text-sm text-zinc-500">Comments</div>
        </div>
      </div>

      {/* Topics */}
      <div className="space-y-4">
        {result.topics.map((topic, i) => (
          <TopicCard key={i} topic={topic} />
        ))}
      </div>

      {/* Generated timestamp */}
      {result.generated_at && (
        <div className="text-center text-sm text-zinc-500">
          Generated: {new Date(result.generated_at).toLocaleString()}
        </div>
      )}
    </div>
  );
}

