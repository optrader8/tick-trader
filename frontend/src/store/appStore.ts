/**
 * Global state management using Zustand
 */

import { create } from 'zustand';
import { FileMetadata, Analysis, SystemStatus } from '../types';

interface AppState {
  // Files
  files: FileMetadata[];
  setFiles: (files: FileMetadata[]) => void;
  addFile: (file: FileMetadata) => void;
  removeFile: (fileId: string) => void;

  // Analyses
  analyses: Map<string, Analysis>;
  setAnalysis: (jobId: string, analysis: Analysis) => void;
  updateAnalysis: (jobId: string, updates: Partial<Analysis>) => void;
  removeAnalysis: (jobId: string) => void;

  // System
  systemStatus: SystemStatus | null;
  setSystemStatus: (status: SystemStatus) => void;

  // UI State
  selectedFileId: string | null;
  setSelectedFileId: (fileId: string | null) => void;

  activeAnalysisId: string | null;
  setActiveAnalysisId: (jobId: string | null) => void;
}

export const useStore = create<AppState>((set) => ({
  // Files
  files: [],
  setFiles: (files) => set({ files }),
  addFile: (file) => set((state) => ({ files: [...state.files, file] })),
  removeFile: (fileId) =>
    set((state) => ({ files: state.files.filter((f) => f.id !== fileId) })),

  // Analyses
  analyses: new Map(),
  setAnalysis: (jobId, analysis) =>
    set((state) => {
      const newAnalyses = new Map(state.analyses);
      newAnalyses.set(jobId, analysis);
      return { analyses: newAnalyses };
    }),
  updateAnalysis: (jobId, updates) =>
    set((state) => {
      const newAnalyses = new Map(state.analyses);
      const existing = newAnalyses.get(jobId);
      if (existing) {
        newAnalyses.set(jobId, { ...existing, ...updates });
      }
      return { analyses: newAnalyses };
    }),
  removeAnalysis: (jobId) =>
    set((state) => {
      const newAnalyses = new Map(state.analyses);
      newAnalyses.delete(jobId);
      return { analyses: newAnalyses };
    }),

  // System
  systemStatus: null,
  setSystemStatus: (status) => set({ systemStatus: status }),

  // UI State
  selectedFileId: null,
  setSelectedFileId: (fileId) => set({ selectedFileId: fileId }),

  activeAnalysisId: null,
  setActiveAnalysisId: (jobId) => set({ activeAnalysisId: jobId }),
}));
