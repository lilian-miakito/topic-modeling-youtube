"use client";

import { useState, useEffect, useCallback } from "react";
import {
  fetchComments,
  getFetchStatus,
  stopFetch,
  getChannel,
  type ChannelInfo,
  type FetchStatus,
} from "@/lib/api";

interface CommentsFetchProps {
  channel: ChannelInfo;
  selectedVideoIds: number[];
  onComplete: (updatedChannel: ChannelInfo) => void;
  onBack: () => void;
}

export function CommentsFetch({
  channel,
  selectedVideoIds,
  onComplete,
  onBack,
}: CommentsFetchProps) {
  const [status, setStatus] = useState<FetchStatus | null>(null);
  const [fetching, setFetching] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Calculate videos needing fetch
  const videosToFetch = channel.videos.filter(
    (v) => selectedVideoIds.includes(v.id) && !v.has_comments
  );
  const videosReady = channel.videos.filter(
    (v) => selectedVideoIds.includes(v.id) && v.has_comments
  );

  // Poll for status
  useEffect(() => {
    if (!fetching) return;

    const interval = setInterval(async () => {
      try {
        const s = await getFetchStatus();
        setStatus(s);

        if (!s.active) {
          setFetching(false);
          // Reload channel to get updated video info
          const updated = await getChannel(channel.id);
          onComplete(updated);
        }
      } catch (err) {
        console.error("Failed to get status:", err);
      }
    }, 1000);

    return () => clearInterval(interval);
  }, [fetching, channel.id, onComplete]);

  const startFetch = async () => {
    setFetching(true);
    setError(null);

    try {
      await fetchComments(videosToFetch.map((v) => v.id));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to start fetch");
      setFetching(false);
    }
  };

  const handleStop = async () => {
    try {
      await stopFetch();
    } catch (err) {
      console.error("Failed to stop:", err);
    }
  };

  const handleSkip = async () => {
    // Just proceed with videos that have comments
    onComplete(channel);
  };

  // All selected videos have comments - auto-continue
  if (videosToFetch.length === 0) {
    return (
      <div className="space-y-6 text-center">
        <div className="p-8">
          <div className="text-5xl mb-4">✅</div>
          <h2 className="text-2xl font-semibold text-zinc-100 mb-2">
            All Comments Ready
          </h2>
          <p className="text-zinc-400">
            All {videosReady.length} selected videos have comments loaded
          </p>
        </div>

        <div className="flex gap-4">
          <button
            onClick={onBack}
            className="flex-1 py-3 bg-zinc-700 hover:bg-zinc-600 text-zinc-100 
                     font-medium rounded-lg transition-colors"
          >
            ← Back
          </button>
          <button
            onClick={() => onComplete(channel)}
            className="flex-1 py-3 bg-amber-600 hover:bg-amber-500 text-white 
                     font-medium rounded-lg transition-colors"
          >
            Continue to Extraction →
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="text-center">
        <h2 className="text-2xl font-semibold text-zinc-100 mb-2">
          Fetch Comments
        </h2>
        <p className="text-zinc-400">
          {videosToFetch.length} videos need comments downloaded
        </p>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-2 gap-4">
        <div className="p-4 bg-zinc-800 rounded-lg text-center">
          <div className="text-3xl font-bold text-green-400">
            {videosReady.length}
          </div>
          <div className="text-sm text-zinc-500">Ready</div>
        </div>
        <div className="p-4 bg-zinc-800 rounded-lg text-center">
          <div className="text-3xl font-bold text-amber-400">
            {videosToFetch.length}
          </div>
          <div className="text-sm text-zinc-500">Need Download</div>
        </div>
      </div>

      {/* Progress */}
      {fetching && status && (
        <div className="space-y-3">
          <div className="flex justify-between text-sm">
            <span className="text-zinc-400">
              {status.current_video || "Starting..."}
            </span>
            <span className="text-zinc-300">
              {status.videos_completed}/{status.videos_total}
            </span>
          </div>
          <div className="h-2 bg-zinc-700 rounded-full overflow-hidden">
            <div
              className="h-full bg-amber-500 transition-all duration-300"
              style={{
                width: `${
                  status.videos_total
                    ? (status.videos_completed / status.videos_total) * 100
                    : 0
                }%`,
              }}
            />
          </div>
          <div className="text-center text-zinc-400 text-sm">
            {status.comments_extracted.toLocaleString()} comments extracted
          </div>
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
          disabled={fetching}
          className="flex-1 py-3 bg-zinc-700 hover:bg-zinc-600 disabled:opacity-50 
                   text-zinc-100 font-medium rounded-lg transition-colors"
        >
          ← Back
        </button>

        {fetching ? (
          <button
            onClick={handleStop}
            className="flex-1 py-3 bg-red-600 hover:bg-red-500 text-white 
                     font-medium rounded-lg transition-colors"
          >
            Stop
          </button>
        ) : (
          <>
            <button
              onClick={handleSkip}
              className="flex-1 py-3 bg-zinc-700 hover:bg-zinc-600 text-zinc-100 
                       font-medium rounded-lg transition-colors"
            >
              Skip ({videosReady.length} ready)
            </button>
            <button
              onClick={startFetch}
              className="flex-1 py-3 bg-amber-600 hover:bg-amber-500 text-white 
                       font-medium rounded-lg transition-colors"
            >
              Download All
            </button>
          </>
        )}
      </div>
    </div>
  );
}

