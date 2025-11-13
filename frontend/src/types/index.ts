/**
 * TypeScript type definitions
 */

export interface FileMetadata {
  id: string;
  filename: string;
  original_name: string;
  file_size: number;
  file_type: string;
  mime_type?: string;
  is_extracted?: boolean;
  parent_file_id?: string;
  metadata?: {
    columns?: string[];
    column_count?: number;
    row_count?: number;
    line_count?: number;
    file_count?: number;
    files?: string[];
    preview_rows?: any[];
    preview?: string[];
    [key: string]: any;
  };
  uploaded_at: string;
}

export interface ExtractedFile {
  file_id: string;
  filename: string;
  file_size: number;
  file_type: string;
  metadata?: any;
}

export interface FileUploadResponse {
  file_id: string;
  filename: string;
  original_name: string;
  file_size: number;
  file_type: string;
  checksum: string;
  metadata: any;
  uploaded_at: string;
  extracted_files?: ExtractedFile[];
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
  precision?: number;
  recall?: number;
  f1_score?: number;
  sharpe_ratio: number;
  max_drawdown: number;
  final_pnl: number;
  win_rate?: number;
  total_trades?: number;
  avg_trade_pnl?: number;
  volatility?: number;
}

export interface PnLDataPoint {
  timestamp: string;
  cumulative_pnl: number;
}

export interface AccuracyDataPoint {
  timestamp: string;
  accuracy: number;
}

export interface FeatureImportance {
  feature: string;
  importance: number;
}

export interface AnalysisResults {
  analysis_id: string;
  model_type: ModelType;
  dataset_id: string;
  config: ModelConfig;
  metrics: AnalysisMetrics;
  pnl_curve?: PnLDataPoint[];
  prediction_accuracy?: AccuracyDataPoint[];
  feature_importance?: FeatureImportance[];
  download_urls?: {
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
