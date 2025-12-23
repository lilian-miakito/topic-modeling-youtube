"use client";

import { useState, useEffect } from "react";
import { listExtractions } from "@/lib/api";

interface ExtractionHistoryProps {
  channelId: number;
  onSelectExtraction: (extractionId: number) => void;
}

interface ExtractionItem {
  id: number;
  channel_id: number;
  status: string;
  progress: number;
  num_comments?: number;
  num_topics?: number;
  created_at?: string;
}

export function ExtractionHistory({
  channelId,
  onSelectExtraction,
}: ExtractionHistoryProps) {
  const [extractions, setExtractions] = useState<ExtractionItem[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const load = async () => {
      try {
        const data = await listExtractions(channelId);
        // Only show completed extractions
        const completed = data.extractions.filter(
          (e) => e.status === "completed"
        );
        setExtractions(completed);
      } catch (err) {
        console.error("Failed to load extractions:", err);
      } finally {
        setLoading(false);
      }
    };

    load();
  }, [channelId]);

  if (loading) {
    return null;
  }

  if (extractions.length === 0) {
    return null;
  }

  return (
    <div className="mb-6 p-4 bg-zinc-800/50 rounded-lg border border-zinc-700">
      <h3 className="text-sm font-medium text-zinc-300 mb-3 flex items-center gap-2">
        <span>📊</span>
        Previous Analyses ({extractions.length})
      </h3>
      <div className="space-y-2">
        {extractions.slice(0, 5).map((ext) => (
          <button
            key={ext.id}
            onClick={() => onSelectExtraction(ext.id)}
            className="w-full p-3 bg-zinc-700/50 hover:bg-zinc-700 rounded-lg text-left transition-colors group"
          >
            <div className="flex items-center justify-between">
              <div>
                <span className="text-zinc-200 font-medium">
                  {ext.num_topics || "?"} topics
                </span>
                <span className="text-zinc-500 mx-2">•</span>
                <span className="text-zinc-400">
                  {ext.num_comments?.toLocaleString() || "?"} comments
                </span>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-xs text-zinc-500">
                  {ext.created_at
                    ? new Date(ext.created_at).toLocaleString()
                    : ""}
                </span>
                <span className="text-amber-500 opacity-0 group-hover:opacity-100 transition-opacity">
                  View →
                </span>
              </div>
            </div>
          </button>
        ))}
      </div>
      {extractions.length > 5 && (
        <p className="text-xs text-zinc-500 mt-2 text-center">
          +{extractions.length - 5} more analyses
        </p>
      )}
    </div>
  );
}

