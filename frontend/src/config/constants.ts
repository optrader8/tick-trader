/**
 * API base URL configuration
 */
export const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000/api';
export const WS_BASE_URL = process.env.REACT_APP_WS_URL || 'ws://localhost:8000/api';

/**
 * File upload configuration
 */
export const FILE_UPLOAD = {
  MAX_SIZE: 100 * 1024 * 1024, // 100MB
  ALLOWED_TYPES: ['.csv', '.parquet', '.json'],
  ALLOWED_MIME_TYPES: [
    'text/csv',
    'application/json',
    'application/octet-stream'
  ]
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
