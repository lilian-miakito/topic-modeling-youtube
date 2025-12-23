"use client";

import { useState, useEffect } from "react";
import {
  startExtraction,
  getExtractionStatus,
  type ChannelInfo,
  type ExtractionStatus,
} from "@/lib/api";

interface ExtractionProps {
  channel: ChannelInfo;
  selectedVideoIds: number[];
  onComplete: (extractionId: number) => void;
  onBack: () => void;
}

export function Extraction({
  channel,
  selectedVideoIds,
  onComplete,
  onBack,
}: ExtractionProps) {
  const [extractionId, setExtractionId] = useState<number | null>(null);
  const [status, setStatus] = useState<ExtractionStatus | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Get videos with comments only
  const videosWithComments = channel.videos.filter(
    (v) => selectedVideoIds.includes(v.id) && v.has_comments
  );

  // Poll for status
  useEffect(() => {
    if (!extractionId) return;

    const interval = setInterval(async () => {
      try {
        const s = await getExtractionStatus(extractionId);
        setStatus(s);

        if (s.status === "completed") {
          clearInterval(interval);
          onComplete(extractionId);
        } else if (s.status === "failed") {
          clearInterval(interval);
          setError(s.error_message || "Extraction failed");
        }
      } catch (err) {
        console.error("Failed to get status:", err);
      }
    }, 1000);

    return () => clearInterval(interval);
  }, [extractionId, onComplete]);

  const handleStart = async () => {
    setError(null);

    try {
      const result = await startExtraction(
        channel.id,
        videosWithComments.map((v) => v.id)
      );
      setExtractionId(result.id);
      setStatus(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to start extraction");
    }
  };

  const progressPercent = status ? Math.round(status.progress * 100) : 0;

  return (
    <div className="space-y-6">
      <div className="text-center">
        <h2 className="text-2xl font-semibold text-zinc-100 mb-2">
          Topic Extraction
        </h2>
        <p className="text-zinc-400">
          Analyze {videosWithComments.length} videos from {channel.name}
        </p>
      </div>

      {/* Stats */}
      <div className="p-4 bg-zinc-800 rounded-lg">
        <div className="grid grid-cols-2 gap-4 text-center">
          <div>
            <div className="text-2xl font-bold text-zinc-100">
              {videosWithComments.length}
            </div>
            <div className="text-xs text-zinc-500">Videos to analyze</div>
          </div>
          <div>
            <div className="text-2xl font-bold text-zinc-100">
              {videosWithComments.reduce((sum, v) => sum + v.comment_count, 0).toLocaleString()}
            </div>
            <div className="text-xs text-zinc-500">Total comments</div>
          </div>
        </div>
      </div>

      {/* Progress */}
      {status && status.status === "running" && (
        <div className="space-y-3">
          <div className="flex justify-between text-sm">
            <span className="text-zinc-400">
              {status.current_step || "Processing..."}
            </span>
            <span className="text-zinc-300">{progressPercent}%</span>
          </div>
          <div className="h-3 bg-zinc-700 rounded-full overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-amber-600 to-amber-400 transition-all duration-500"
              style={{ width: `${progressPercent}%` }}
            />
          </div>
          {status.num_comments && (
            <div className="text-center text-zinc-500 text-sm">
              Analyzing {status.num_comments.toLocaleString()} comments
            </div>
          )}
        </div>
      )}

      {/* Pipeline steps */}
      {!extractionId && (
        <div className="space-y-2">
          <h3 className="text-sm font-medium text-zinc-400 mb-3">
            Pipeline Steps:
          </h3>
          {[
            "Load comments from database",
            "Generate embeddings (cached)",
            "Detect corpus-specific stopwords",
            "Cluster with UMAP + HDBSCAN",
            "Calculate silhouette scores",
            "Extract semantic words (centroid + MMR)",
            "Split low-quality clusters",
            "Name topics with LLM",
          ].map((step, i) => (
            <div
              key={i}
              className="flex items-center gap-3 text-sm text-zinc-400"
            >
              <span className="w-6 h-6 rounded-full bg-zinc-700 flex items-center justify-center text-xs">
                {i + 1}
              </span>
              {step}
            </div>
          ))}
        </div>
      )}

      {error && (
        <div className="p-3 bg-red-900/30 border border-red-700 rounded-lg text-red-300 text-sm">
          {error}
        </div>
      )}

      {/* Actions */}
      <div className="flex gap-4">
        <button
          onClick={onBack}
          disabled={status?.status === "running"}
          className="flex-1 py-3 bg-zinc-700 hover:bg-zinc-600 disabled:opacity-50 
                   text-zinc-100 font-medium rounded-lg transition-colors"
        >
          ← Back
        </button>

        {!extractionId && (
          <button
            onClick={handleStart}
            disabled={videosWithComments.length === 0}
            className="flex-1 py-3 bg-amber-600 hover:bg-amber-500 disabled:bg-zinc-700 
                     disabled:text-zinc-500 text-white font-medium rounded-lg 
                     transition-colors"
          >
            Start Extraction
          </button>
        )}

        {status?.status === "running" && (
          <div className="flex-1 py-3 bg-zinc-700 text-zinc-300 font-medium 
                        rounded-lg text-center flex items-center justify-center gap-2">
            <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24">
              <circle
                className="opacity-25"
                cx="12"
                cy="12"
                r="10"
                stroke="currentColor"
                strokeWidth="4"
                fill="none"
              />
              <path
                className="opacity-75"
                fill="currentColor"
                d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
              />
            </svg>
            Extracting...
          </div>
        )}
      </div>
    </div>
  );
}

