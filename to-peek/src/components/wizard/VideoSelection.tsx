"use client";

import { useState, useMemo } from "react";
import type { ChannelInfo, VideoInfo } from "@/lib/api";

interface VideoSelectionProps {
  channel: ChannelInfo;
  onVideosSelected: (videoIds: number[]) => void;
  onBack: () => void;
}

export function VideoSelection({ channel, onVideosSelected, onBack }: VideoSelectionProps) {
  const [selectedIds, setSelectedIds] = useState<Set<number>>(
    new Set(channel.videos.map((v) => v.id))
  );
  const [filter, setFilter] = useState<"all" | "with" | "without">("all");
  const [search, setSearch] = useState("");

  const filteredVideos = useMemo(() => {
    let videos = channel.videos;

    // Filter by comment status
    if (filter === "with") {
      videos = videos.filter((v) => v.has_comments);
    } else if (filter === "without") {
      videos = videos.filter((v) => !v.has_comments);
    }

    // Filter by search
    if (search.trim()) {
      const q = search.toLowerCase();
      videos = videos.filter((v) => v.title.toLowerCase().includes(q));
    }

    return videos;
  }, [channel.videos, filter, search]);

  const stats = useMemo(() => {
    const withComments = channel.videos.filter((v) => v.has_comments).length;
    return {
      total: channel.videos.length,
      withComments,
      withoutComments: channel.videos.length - withComments,
      selected: selectedIds.size,
      selectedWithComments: channel.videos.filter(
        (v) => selectedIds.has(v.id) && v.has_comments
      ).length,
    };
  }, [channel.videos, selectedIds]);

  const toggleVideo = (id: number) => {
    const newSet = new Set(selectedIds);
    if (newSet.has(id)) {
      newSet.delete(id);
    } else {
      newSet.add(id);
    }
    setSelectedIds(newSet);
  };

  const selectAll = () => {
    setSelectedIds(new Set(filteredVideos.map((v) => v.id)));
  };

  const selectNone = () => {
    const filteredIds = new Set(filteredVideos.map((v) => v.id));
    setSelectedIds(new Set([...selectedIds].filter((id) => !filteredIds.has(id))));
  };

  const handleContinue = () => {
    onVideosSelected([...selectedIds]);
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-semibold text-zinc-100">{channel.name}</h2>
          <p className="text-zinc-400">{channel.handle}</p>
        </div>
        <button
          onClick={onBack}
          className="px-3 py-1 text-zinc-400 hover:text-zinc-200 transition-colors"
        >
          ← Back
        </button>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-4 gap-4">
        <div className="p-3 bg-zinc-800 rounded-lg text-center">
          <div className="text-2xl font-bold text-zinc-100">{stats.total}</div>
          <div className="text-xs text-zinc-500">Total Videos</div>
        </div>
        <div className="p-3 bg-zinc-800 rounded-lg text-center">
          <div className="text-2xl font-bold text-green-400">{stats.withComments}</div>
          <div className="text-xs text-zinc-500">With Comments</div>
        </div>
        <div className="p-3 bg-zinc-800 rounded-lg text-center">
          <div className="text-2xl font-bold text-amber-400">{stats.withoutComments}</div>
          <div className="text-xs text-zinc-500">Without Comments</div>
        </div>
        <div className="p-3 bg-zinc-800 rounded-lg text-center">
          <div className="text-2xl font-bold text-blue-400">{stats.selected}</div>
          <div className="text-xs text-zinc-500">Selected</div>
        </div>
      </div>

      {/* Filters */}
      <div className="flex gap-4 items-center">
        <input
          type="text"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder="Search videos..."
          className="flex-1 px-3 py-2 bg-zinc-800 border border-zinc-700 rounded-lg 
                   text-zinc-100 placeholder-zinc-500 focus:outline-none 
                   focus:ring-1 focus:ring-amber-500"
        />
        <select
          value={filter}
          onChange={(e) => setFilter(e.target.value as "all" | "with" | "without")}
          className="px-3 py-2 bg-zinc-800 border border-zinc-700 rounded-lg 
                   text-zinc-100 focus:outline-none focus:ring-1 focus:ring-amber-500"
        >
          <option value="all">All videos</option>
          <option value="with">With comments</option>
          <option value="without">Without comments</option>
        </select>
        <button
          onClick={selectAll}
          className="px-3 py-2 text-sm text-zinc-400 hover:text-zinc-200"
        >
          Select all
        </button>
        <button
          onClick={selectNone}
          className="px-3 py-2 text-sm text-zinc-400 hover:text-zinc-200"
        >
          Select none
        </button>
      </div>

      {/* Video list */}
      <div className="max-h-96 overflow-y-auto space-y-2 pr-2">
        {filteredVideos.map((video) => (
          <label
            key={video.id}
            className={`flex items-center gap-3 p-3 rounded-lg cursor-pointer transition-colors
                      ${selectedIds.has(video.id) ? "bg-zinc-700" : "bg-zinc-800 hover:bg-zinc-750"}`}
          >
            <input
              type="checkbox"
              checked={selectedIds.has(video.id)}
              onChange={() => toggleVideo(video.id)}
              className="w-4 h-4 rounded border-zinc-600 text-amber-500 
                       focus:ring-amber-500 focus:ring-offset-zinc-900"
            />
            <div className="flex-1 min-w-0">
              <div className="text-zinc-100 truncate">{video.title}</div>
              <div className="text-xs text-zinc-500">{video.youtube_id}</div>
            </div>
            {video.has_comments ? (
              <span className="px-2 py-1 text-xs bg-green-900/50 text-green-400 rounded">
                {video.comment_count} comments
              </span>
            ) : (
              <span className="px-2 py-1 text-xs bg-zinc-700 text-zinc-400 rounded">
                No comments
              </span>
            )}
          </label>
        ))}
      </div>

      {/* Continue button */}
      <button
        onClick={handleContinue}
        disabled={selectedIds.size === 0}
        className="w-full py-3 bg-amber-600 hover:bg-amber-500 disabled:bg-zinc-700 
                 disabled:text-zinc-500 text-white font-medium rounded-lg 
                 transition-colors"
      >
        Continue with {selectedIds.size} videos
      </button>
    </div>
  );
}

