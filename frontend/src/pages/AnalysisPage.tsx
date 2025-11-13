/**
 * Analysis page - Configure and start analysis
 */

import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { toast } from 'react-toastify';
import { apiService } from '../services/api';
import { wsService } from '../services/websocket';
import { useStore } from '../store/appStore';
import { MODEL_TYPES } from '../config/constants';
import { ModelType } from '../types';

const AnalysisPage: React.FC = () => {
  const { jobId } = useParams();
  const navigate = useNavigate();
  const { files, setAnalysis, updateAnalysis } = useStore();

  const [selectedFileId, setSelectedFileId] = useState('');
  const [modelType, setModelType] = useState<ModelType>('lstm');
  const [config, setConfig] = useState(MODEL_TYPES.lstm.defaultConfig);
  const [starting, setStarting] = useState(false);

  useEffect(() => {
    if (jobId) {
      // Load existing analysis
      loadAnalysisStatus();
    }
  }, [jobId]);

  const loadAnalysisStatus = async () => {
    if (!jobId) return;
    try {
      const analysis = await apiService.getAnalysisStatus(jobId);
      setAnalysis(jobId, analysis);

      // Connect WebSocket for real-time updates
      if (analysis.status === 'running' || analysis.status === 'pending') {
        wsService.connectToAnalysis(jobId, (updatedAnalysis) => {
          updateAnalysis(jobId, updatedAnalysis);
        });
      }
    } catch (error) {
      toast.error('Failed to load analysis status');
    }
  };

  const handleStart = async () => {
    if (!selectedFileId) {
      toast.error('Please select a file');
      return;
    }

    setStarting(true);
    try {
      const result = await apiService.startAnalysis({
        dataset_id: selectedFileId,
        model_type: modelType,
        config
      });

      toast.success('Analysis started!');
      navigate(`/analysis/${result.job_id}`);
    } catch (error: any) {
      toast.error(`Failed to start analysis: ${error.response?.data?.detail || error.message}`);
    } finally {
      setStarting(false);
    }
  };

  return (
    <div className="analysis-page">
      <h2>Configure Analysis</h2>

      <div className="card">
        <h3>Select Dataset</h3>
        <select value={selectedFileId} onChange={(e) => setSelectedFileId(e.target.value)}>
          <option value="">Choose a file...</option>
          {files.map((file) => (
            <option key={file.id} value={file.id}>
              {file.original_name} ({(file.file_size / 1024 / 1024).toFixed(2)} MB)
            </option>
          ))}
        </select>
      </div>

      <div className="card">
        <h3>Select Model</h3>
        <div className="model-grid">
          {Object.entries(MODEL_TYPES).map(([key, model]) => (
            <div
              key={key}
              className={`model-card ${modelType === key ? 'selected' : ''}`}
              onClick={() => {
                setModelType(key as ModelType);
                setConfig(model.defaultConfig);
              }}
            >
              <h4>{model.label}</h4>
              <p>{model.description}</p>
            </div>
          ))}
        </div>
      </div>

      <div className="card">
        <h3>Configuration</h3>
        <div className="config-form">
          {modelType === 'lstm' && (
            <>
              <div className="form-group">
                <label>Sequence Length</label>
                <input
                  type="number"
                  value={config.sequence_length || 100}
                  onChange={(e) => setConfig({ ...config, sequence_length: parseInt(e.target.value) })}
                />
                <span className="form-help">Number of time steps to look back</span>
              </div>
              <div className="form-group">
                <label>LSTM Units (comma-separated)</label>
                <input
                  type="text"
                  value={config.lstm_units?.join(',') || '128,64,32'}
                  onChange={(e) => setConfig({ ...config, lstm_units: e.target.value.split(',').map(Number) })}
                />
                <span className="form-help">Units per LSTM layer (e.g., 128,64,32)</span>
              </div>
              <div className="form-group">
                <label>Dropout Rate</label>
                <input
                  type="number"
                  step="0.01"
                  min="0"
                  max="1"
                  value={config.dropout_rate || 0.2}
                  onChange={(e) => setConfig({ ...config, dropout_rate: parseFloat(e.target.value) })}
                />
                <span className="form-help">Dropout for regularization (0.0 - 1.0)</span>
              </div>
            </>
          )}

          {modelType === 'transformer' && (
            <>
              <div className="form-group">
                <label>Sequence Length</label>
                <input
                  type="number"
                  value={config.sequence_length || 100}
                  onChange={(e) => setConfig({ ...config, sequence_length: parseInt(e.target.value) })}
                />
              </div>
              <div className="form-group">
                <label>Model Dimension (d_model)</label>
                <input
                  type="number"
                  value={config.d_model || 128}
                  onChange={(e) => setConfig({ ...config, d_model: parseInt(e.target.value) })}
                />
                <span className="form-help">Dimensionality of the model</span>
              </div>
              <div className="form-group">
                <label>Number of Heads</label>
                <input
                  type="number"
                  value={config.n_heads || 8}
                  onChange={(e) => setConfig({ ...config, n_heads: parseInt(e.target.value) })}
                />
                <span className="form-help">Number of attention heads</span>
              </div>
              <div className="form-group">
                <label>Number of Layers</label>
                <input
                  type="number"
                  value={config.num_layers || 4}
                  onChange={(e) => setConfig({ ...config, num_layers: parseInt(e.target.value) })}
                />
              </div>
            </>
          )}

          {modelType === 'cnn_lstm' && (
            <>
              <div className="form-group">
                <label>Sequence Length</label>
                <input
                  type="number"
                  value={config.sequence_length || 100}
                  onChange={(e) => setConfig({ ...config, sequence_length: parseInt(e.target.value) })}
                />
              </div>
              <div className="form-group">
                <label>CNN Filters (comma-separated)</label>
                <input
                  type="text"
                  value={config.conv_filters?.join(',') || '64,128,256'}
                  onChange={(e) => setConfig({ ...config, conv_filters: e.target.value.split(',').map(Number) })}
                />
                <span className="form-help">Filters per CNN layer (e.g., 64,128,256)</span>
              </div>
              <div className="form-group">
                <label>LSTM Units (comma-separated)</label>
                <input
                  type="text"
                  value={config.lstm_units?.join(',') || '128,64'}
                  onChange={(e) => setConfig({ ...config, lstm_units: e.target.value.split(',').map(Number) })}
                />
              </div>
              <div className="form-group">
                <label>Dropout Rate</label>
                <input
                  type="number"
                  step="0.01"
                  min="0"
                  max="1"
                  value={config.dropout_rate || 0.2}
                  onChange={(e) => setConfig({ ...config, dropout_rate: parseFloat(e.target.value) })}
                />
              </div>
            </>
          )}

          {modelType === 'ensemble' && (
            <>
              <div className="form-group">
                <label>Number of Estimators</label>
                <input
                  type="number"
                  value={config.n_estimators || 100}
                  onChange={(e) => setConfig({ ...config, n_estimators: parseInt(e.target.value) })}
                />
                <span className="form-help">Number of trees/estimators in ensemble</span>
              </div>
            </>
          )}

          {/* Common training parameters */}
          <div className="form-divider">Training Parameters</div>
          <div className="form-group">
            <label>Epochs</label>
            <input
              type="number"
              value={config.epochs || 50}
              onChange={(e) => setConfig({ ...config, epochs: parseInt(e.target.value) })}
            />
            <span className="form-help">Number of training epochs</span>
          </div>
          <div className="form-group">
            <label>Batch Size</label>
            <input
              type="number"
              value={config.batch_size || 32}
              onChange={(e) => setConfig({ ...config, batch_size: parseInt(e.target.value) })}
            />
            <span className="form-help">Training batch size</span>
          </div>
        </div>
      </div>

      <button
        className="btn btn-primary"
        onClick={handleStart}
        disabled={starting || !selectedFileId}
      >
        {starting ? 'Starting...' : 'Start Analysis'}
      </button>
    </div>
  );
};

export default AnalysisPage;
