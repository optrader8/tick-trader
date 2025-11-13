/**
 * TypeScript type definitions
 */

export interface FileMetadata {
  id: string;
  filename: string;
  original_name: string;
  file_size: number;
  mime_type?: string;
  uploaded_at: string;
}

export interface FileUploadResponse {
  file_id: string;
  filename: string;
  original_name: string;
  file_size: number;
  uploaded_at: string;
}

export type ModelType = 'lstm' | 'transformer' | 'ensemble' | 'cnn_lstm';

export interface ModelConfig {
  sequence_length?: number;
  lstm_units?: number[];
  dropout_rate?: number;
  epochs?: number;
  batch_size?: number;
  d_model?: number;
  n_heads?: number;
  num_layers?: number;
  conv_filters?: number[];
  n_estimators?: number;
}

export interface StartAnalysisRequest {
  dataset_id: string;
  model_type: ModelType;
  config: ModelConfig;
}

export type AnalysisStatus = 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';

export interface Analysis {
  id: string;
  dataset_id: string;
  model_type: ModelType;
  status: AnalysisStatus;
  progress: number;
  current_step?: string;
  estimated_time_remaining?: number;
  started_at?: string;
  completed_at?: string;
  error_message?: string;
  logs?: string[];
}

export interface AnalysisMetrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1_score: number;
  sharpe_ratio: number;
  max_drawdown: number;
  final_pnl: number;
}

export interface AnalysisResults {
  analysis_id: string;
  metrics: AnalysisMetrics;
  charts_data?: {
    pnl_curve: Array<{ timestamp: string; value: number }>;
    predictions: Array<{ timestamp: string; actual: number; predicted: number }>;
  };
  download_urls: {
    csv: string;
    json: string;
    report: string;
  };
}

export interface SystemStatus {
  cpu_usage: number;
  memory_usage: number;
  disk_usage: number;
  running_jobs: number;
}

export interface ApiResponse<T = any> {
  success: boolean;
  data?: T;
  error?: {
    code: string;
    message: string;
    details?: any;
  };
}
