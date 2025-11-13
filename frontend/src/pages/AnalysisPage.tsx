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
        <pre>{JSON.stringify(config, null, 2)}</pre>
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
