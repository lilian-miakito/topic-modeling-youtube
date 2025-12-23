"use client";

import { useState } from "react";
import { searchChannel, listChannels, type ChannelInfo } from "@/lib/api";

interface ChannelInputProps {
  onChannelSelected: (channel: ChannelInfo) => void;
}

export function ChannelInput({ onChannelSelected }: ChannelInputProps) {
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [existingChannels, setExistingChannels] = useState<
    Array<{ id: number; handle: string; name: string; video_count: number }>
  >([]);
  const [showExisting, setShowExisting] = useState(false);

  const loadExistingChannels = async () => {
    try {
      const data = await listChannels();
      setExistingChannels(data.channels);
      setShowExisting(true);
    } catch (err) {
      console.error("Failed to load channels:", err);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim()) return;

    setLoading(true);
    setError(null);

    try {
      const channel = await searchChannel(input.trim());
      onChannelSelected(channel);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to find channel");
    } finally {
      setLoading(false);
    }
  };

  const selectExisting = async (channelId: number) => {
    setLoading(true);
    try {
      const { getChannel } = await import("@/lib/api");
      const channel = await getChannel(channelId);
      onChannelSelected(channel);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load channel");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <div className="text-center">
        <h2 className="text-2xl font-semibold text-zinc-100 mb-2">
          Select a YouTube Channel
        </h2>
        <p className="text-zinc-400">
          Enter a channel handle (e.g., @Fireship) or paste a channel URL
        </p>
      </div>

      <form onSubmit={handleSubmit} className="space-y-4">
        <div className="relative">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="@channel or https://youtube.com/@channel"
            className="w-full px-4 py-3 bg-zinc-800 border border-zinc-700 rounded-lg 
                     text-zinc-100 placeholder-zinc-500 focus:outline-none 
                     focus:ring-2 focus:ring-amber-500 focus:border-transparent"
            disabled={loading}
          />
        </div>

        {error && (
          <div className="p-3 bg-red-900/30 border border-red-700 rounded-lg text-red-300 text-sm">
            {error}
          </div>
        )}

        <button
          type="submit"
          disabled={loading || !input.trim()}
          className="w-full py-3 bg-amber-600 hover:bg-amber-500 disabled:bg-zinc-700 
                   disabled:text-zinc-500 text-white font-medium rounded-lg 
                   transition-colors"
        >
          {loading ? (
            <span className="flex items-center justify-center gap-2">
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
              Searching...
            </span>
          ) : (
            "Search Channel"
          )}
        </button>
      </form>

      <div className="relative">
        <div className="absolute inset-0 flex items-center">
          <div className="w-full border-t border-zinc-700" />
        </div>
        <div className="relative flex justify-center text-sm">
          <span className="px-2 bg-zinc-900 text-zinc-500">or</span>
        </div>
      </div>

      <button
        onClick={loadExistingChannels}
        className="w-full py-2 text-zinc-400 hover:text-zinc-300 transition-colors"
      >
        Load from existing channels
      </button>

      {showExisting && existingChannels.length > 0 && (
        <div className="space-y-2 max-h-60 overflow-y-auto">
          {existingChannels.map((ch) => (
            <button
              key={ch.id}
              onClick={() => selectExisting(ch.id)}
              className="w-full p-3 bg-zinc-800 hover:bg-zinc-700 rounded-lg 
                       text-left transition-colors flex justify-between items-center"
            >
              <div>
                <div className="text-zinc-100 font-medium">{ch.name}</div>
                <div className="text-zinc-500 text-sm">{ch.handle}</div>
              </div>
              <div className="text-zinc-500 text-sm">
                {ch.video_count} videos
              </div>
            </button>
          ))}
        </div>
      )}

      {showExisting && existingChannels.length === 0 && (
        <div className="text-center text-zinc-500 py-4">
          No channels loaded yet
        </div>
      )}
    </div>
  );
}

