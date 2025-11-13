/**
 * API base URL configuration
 */
export const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000/api';
export const WS_BASE_URL = process.env.REACT_APP_WS_URL || 'ws://localhost:8000/api';

/**
 * File upload configuration
 */
export const FILE_UPLOAD = {
  MAX_SIZE: 500 * 1024 * 1024, // 500MB (increased for zip files)
  ALLOWED_TYPES: ['.csv', '.parquet', '.json', '.txt', '.zip', '.gz', '.tar', '.tar.gz'],
  ALLOWED_MIME_TYPES: [
    'text/csv',
    'application/json',
    'application/octet-stream',
    'text/plain',
    'application/zip',
    'application/x-zip-compressed',
    'application/gzip',
    'application/x-gzip',
    'application/x-tar',
    'application/x-compressed-tar'
  ]
};

/**
 * File type icons and labels
 */
export const FILE_TYPE_INFO = {
  csv: { icon: '📊', label: 'CSV', color: '#4CAF50' },
  json: { icon: '📝', label: 'JSON', color: '#FF9800' },
  txt: { icon: '📄', label: 'Text', color: '#2196F3' },
  parquet: { icon: '🗂️', label: 'Parquet', color: '#9C27B0' },
  zip: { icon: '📦', label: 'ZIP', color: '#F44336' },
  gz: { icon: '🗜️', label: 'GZIP', color: '#607D8B' },
  tar: { icon: '📦', label: 'TAR', color: '#795548' },
  'tar.gz': { icon: '🗜️', label: 'TAR.GZ', color: '#607D8B' },
  unknown: { icon: '❓', label: 'Unknown', color: '#9E9E9E' }
};

/**
 * Model types and their default configurations
 */
export const MODEL_TYPES = {
  lstm: {
    label: 'LSTM',
    description: 'Long Short-Term Memory network for time series',
    defaultConfig: {
      sequence_length: 100,
      lstm_units: [128, 64, 32],
      dropout_rate: 0.2,
      epochs: 50,
      batch_size: 32
    }
  },
  transformer: {
    label: 'Transformer',
    description: 'Transformer with attention mechanism',
    defaultConfig: {
      d_model: 128,
      n_heads: 8,
      num_layers: 4,
      dropout_rate: 0.1,
      epochs: 50,
      batch_size: 32
    }
  },
  ensemble: {
    label: 'Ensemble',
    description: 'LightGBM + XGBoost + RandomForest',
    defaultConfig: {
      n_estimators: 100
    }
  },
  cnn_lstm: {
    label: 'CNN-LSTM',
    description: 'Hybrid CNN-LSTM architecture',
    defaultConfig: {
      conv_filters: [64, 32],
      lstm_units: 64,
      epochs: 50,
      batch_size: 32
    }
  }
};

/**
 * Analysis status types
 */
export const ANALYSIS_STATUS = {
  PENDING: 'pending',
  RUNNING: 'running',
  COMPLETED: 'completed',
  FAILED: 'failed',
  CANCELLED: 'cancelled'
};

/**
 * Status colors
 */
export const STATUS_COLORS = {
  pending: '#FFA726',
  running: '#29B6F6',
  completed: '#66BB6A',
  failed: '#EF5350',
  cancelled: '#9E9E9E'
};
