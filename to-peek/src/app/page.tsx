"use client";

import { useState, useCallback } from "react";
import { ChannelInput } from "@/components/wizard/ChannelInput";
import { VideoSelection } from "@/components/wizard/VideoSelection";
import { CommentsFetch } from "@/components/wizard/CommentsFetch";
import { Extraction } from "@/components/wizard/Extraction";
import { ExtractionHistory } from "@/components/wizard/ExtractionHistory";
import { TopicTree } from "@/components/results/TopicTree";
import type { ChannelInfo } from "@/lib/api";

type WizardStep = "channel" | "videos" | "fetch" | "extraction" | "results";

export default function Home() {
  const [step, setStep] = useState<WizardStep>("channel");
  const [channel, setChannel] = useState<ChannelInfo | null>(null);
  const [selectedVideoIds, setSelectedVideoIds] = useState<number[]>([]);
  const [extractionId, setExtractionId] = useState<number | null>(null);

  const handleChannelSelected = useCallback((ch: ChannelInfo) => {
    setChannel(ch);
    setStep("videos");
  }, []);

  const handleVideosSelected = useCallback((ids: number[]) => {
    setSelectedVideoIds(ids);
    setStep("fetch");
  }, []);

  const handleFetchComplete = useCallback((updatedChannel: ChannelInfo) => {
    setChannel(updatedChannel);
    setStep("extraction");
  }, []);

  const handleExtractionComplete = useCallback((id: number) => {
    setExtractionId(id);
    setStep("results");
  }, []);

  const handleSelectPastExtraction = useCallback((id: number) => {
    setExtractionId(id);
    setStep("results");
  }, []);

  const handleRestart = useCallback(() => {
    setStep("channel");
    setChannel(null);
    setSelectedVideoIds([]);
    setExtractionId(null);
  }, []);

  const goBack = useCallback((toStep: WizardStep) => {
    setStep(toStep);
  }, []);

  // Step indicator
  const steps: { key: WizardStep; label: string }[] = [
    { key: "channel", label: "Channel" },
    { key: "videos", label: "Videos" },
    { key: "fetch", label: "Comments" },
    { key: "extraction", label: "Extract" },
    { key: "results", label: "Results" },
  ];

  const currentIndex = steps.findIndex((s) => s.key === step);

  return (
    <div className="min-h-screen bg-zinc-900">
      {/* Header */}
      <header className="border-b border-zinc-800">
        <div className="max-w-4xl mx-auto px-6 py-4 flex items-center justify-between">
          <h1 className="text-xl font-bold text-zinc-100">
            <span className="text-amber-500">To</span>-Peek
          </h1>
          <span className="text-sm text-zinc-500">
            YouTube Topic Analyzer
          </span>
        </div>
      </header>

      {/* Progress indicator */}
      <div className="border-b border-zinc-800">
        <div className="max-w-4xl mx-auto px-6 py-3">
          <div className="flex items-center justify-between">
            {steps.map((s, i) => (
              <div
                key={s.key}
                className={`flex items-center ${
                  i < steps.length - 1 ? "flex-1" : ""
                }`}
              >
                <div
                  className={`flex items-center justify-center w-8 h-8 rounded-full text-sm font-medium
                    ${
                      i < currentIndex
                        ? "bg-amber-600 text-white"
                        : i === currentIndex
                        ? "bg-amber-500 text-white"
                        : "bg-zinc-700 text-zinc-400"
                    }`}
                >
                  {i < currentIndex ? "✓" : i + 1}
                </div>
                <span
                  className={`ml-2 text-sm hidden sm:block ${
                    i <= currentIndex ? "text-zinc-200" : "text-zinc-500"
                  }`}
                >
                  {s.label}
                </span>
                {i < steps.length - 1 && (
                  <div
                    className={`flex-1 h-0.5 mx-4 ${
                      i < currentIndex ? "bg-amber-600" : "bg-zinc-700"
                    }`}
                  />
                )}
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Main content */}
      <main className="max-w-4xl mx-auto px-6 py-8">
        <div className="bg-zinc-850 rounded-xl p-6 shadow-xl border border-zinc-800">
          {step === "channel" && (
            <ChannelInput onChannelSelected={handleChannelSelected} />
          )}

          {step === "videos" && channel && (
            <>
              <ExtractionHistory
                channelId={channel.id}
                onSelectExtraction={handleSelectPastExtraction}
              />
              <VideoSelection
                channel={channel}
                onVideosSelected={handleVideosSelected}
                onBack={() => goBack("channel")}
                onChannelDeleted={() => {
                  setChannel(null);
                  setStep("channel");
                }}
              />
            </>
          )}

          {step === "fetch" && channel && (
            <CommentsFetch
              channel={channel}
              selectedVideoIds={selectedVideoIds}
              onComplete={handleFetchComplete}
              onBack={() => goBack("videos")}
            />
          )}

          {step === "extraction" && channel && (
            <Extraction
              channel={channel}
              selectedVideoIds={selectedVideoIds}
              onComplete={handleExtractionComplete}
              onBack={() => goBack("fetch")}
            />
          )}

          {step === "results" && extractionId && (
            <TopicTree
              extractionId={extractionId}
              onRestart={handleRestart}
            />
          )}
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t border-zinc-800 mt-auto">
        <div className="max-w-4xl mx-auto px-6 py-4 text-center text-sm text-zinc-500">
          Powered by BERTopic + DSPy
        </div>
      </footer>
    </div>
  );
}
