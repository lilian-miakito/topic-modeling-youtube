/**
 * API client for To-Peek backend
 */

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api/v1";

// =============================================================================
// Types
// =============================================================================

export interface VideoInfo {
  id: number;
  youtube_id: string;
  title: string;
  url: string;
  has_comments: boolean;
  comment_count: number;
}

export interface ChannelInfo {
  id: number;
  handle: string;
  name: string;
  channel_id?: string;
  description?: string;
  subscriber_count?: number;
  video_count: number;
  videos: VideoInfo[];
}

export interface FetchStatus {
  active: boolean;
  channel_id?: number;
  channel_name?: string;
  videos_total: number;
  videos_completed: number;
  comments_extracted: number;
  current_video?: string;
}

export interface ExtractionStatus {
  id: number;
  status: string;
  progress: number;
  current_step?: string;
  error_message?: string;
  num_comments?: number;
  num_topics?: number;
}

export interface TopicInfo {
  id: number | string;
  depth: number;
  parent_id?: number | string;
  parent_name?: string;
  generated_name: string;
  count: number;
  persistence?: number;
  variance?: number;
  max_distance?: number;
  mean_distance?: number;
  top_words: string[];
  example_comments: string[];
  is_hierarchical: boolean;
  children: TopicInfo[];
}

export interface OutliersInfo {
  count: number;
  percentage: number;
  examples: string[];
}

export interface ExtractionResult {
  id: number;
  status: string;
  generated_at?: string;
  duration_seconds?: number;
  num_comments?: number;
  num_topics?: number;
  num_hierarchical?: number;
  num_subtopics?: number;
  outliers?: OutliersInfo;
  topics: TopicInfo[];
}

// =============================================================================
// API Functions
// =============================================================================

async function fetchApi<T>(endpoint: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${endpoint}`, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...options?.headers,
    },
  });

  if (!res.ok) {
    const error = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(error.detail || "API Error");
  }

  return res.json();
}

// YouTube endpoints
export async function searchChannel(channel: string): Promise<ChannelInfo> {
  return fetchApi("/youtube/channel", {
    method: "POST",
    body: JSON.stringify({ channel }),
  });
}

export async function getChannel(channelId: number): Promise<ChannelInfo> {
  return fetchApi(`/youtube/channel/${channelId}`);
}

export async function getChannelVideos(channelId: number) {
  return fetchApi<{
    channel_id: number;
    channel_name: string;
    total_videos: number;
    videos_with_comments: number;
    videos: VideoInfo[];
  }>(`/youtube/channel/${channelId}/videos`);
}

export async function listChannels() {
  return fetchApi<{
    channels: Array<{
      id: number;
      handle: string;
      name: string;
      subscriber_count?: number;
      video_count: number;
      videos_with_comments: number;
    }>;
    total: number;
  }>("/youtube/channels");
}

export async function fetchComments(videoIds: number[], maxWorkers = 4) {
  return fetchApi<{ success: boolean; message: string; video_count: number }>(
    "/youtube/videos/fetch-comments",
    {
      method: "POST",
      body: JSON.stringify({ video_ids: videoIds, max_workers: maxWorkers }),
    }
  );
}

export async function getFetchStatus(): Promise<FetchStatus> {
  return fetchApi("/youtube/fetch-status");
}

export async function stopFetch() {
  return fetchApi<{ success: boolean; message: string }>("/youtube/fetch-stop", {
    method: "POST",
  });
}

// Extraction endpoints
export async function startExtraction(
  channelId: number,
  videoIds: number[],
  config?: Record<string, unknown>
): Promise<ExtractionStatus> {
  return fetchApi("/extract/start", {
    method: "POST",
    body: JSON.stringify({
      channel_id: channelId,
      video_ids: videoIds,
      config,
    }),
  });
}

export async function getExtractionStatus(extractionId: number): Promise<ExtractionStatus> {
  return fetchApi(`/extract/status/${extractionId}`);
}

export async function getExtractionResult(extractionId: number): Promise<ExtractionResult> {
  return fetchApi(`/extract/result/${extractionId}`);
}

export async function listExtractions(channelId?: number) {
  const params = channelId ? `?channel_id=${channelId}` : "";
  return fetchApi<{
    extractions: Array<{
      id: number;
      channel_id: number;
      status: string;
      progress: number;
      num_comments?: number;
      num_topics?: number;
      created_at?: string;
    }>;
    total: number;
  }>(`/extract/list${params}`);
}

