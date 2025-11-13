/**
 * Results page - Display analysis results
 */

import React, { useEffect, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { toast } from 'react-toastify';
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { apiService } from '../services/api';
import { AnalysisResults } from '../types';

const ResultsPage: React.FC = () => {
  const { jobId } = useParams();
  const navigate = useNavigate();
  const [results, setResults] = useState<AnalysisResults | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (jobId) {
      loadResults();
    }
  }, [jobId]);

  const loadResults = async () => {
    if (!jobId) return;

    setLoading(true);
    try {
      const data = await apiService.getAnalysisResults(jobId);
      setResults(data);
    } catch (error: any) {
      toast.error(`Failed to load results: ${error.message}`);
      // If analysis not complete, redirect to analysis page
      if (error.response?.status === 404 || error.response?.status === 400) {
        setTimeout(() => navigate(`/analysis/${jobId}`), 2000);
      }
    } finally {
      setLoading(false);
    }
  };

  const handleDownload = async (format: 'csv' | 'json' | 'report') => {
    if (!jobId) return;

    try {
      const blob = await apiService.downloadResults(jobId, format);
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `analysis_${jobId}_${format}.${format === 'report' ? 'pdf' : format}`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
      toast.success(`Downloaded ${format.toUpperCase()} file`);
    } catch (error) {
      toast.error(`Failed to download ${format.toUpperCase()}`);
    }
  };

  if (loading) {
    return (
      <div className="results-page">
        <h2>Analysis Results</h2>
        <div className="card">
          <div className="spinner" />
          <p>Loading results...</p>
        </div>
      </div>
    );
  }

  if (!results) {
    return (
      <div className="results-page">
        <h2>Analysis Results</h2>
        <div className="card">
          <p>No results available. Redirecting to analysis page...</p>
        </div>
      </div>
    );
  }

  const metrics = results.metrics;

  return (
    <div className="results-page">
      <div className="results-header">
        <div>
          <h2>Analysis Results</h2>
          <p className="job-id">Job ID: {jobId?.substring(0, 8)}</p>
        </div>
        <div className="download-actions">
          <button className="btn btn-secondary" onClick={() => handleDownload('csv')}>
            Download CSV
          </button>
          <button className="btn btn-secondary" onClick={() => handleDownload('json')}>
            Download JSON
          </button>
          <button className="btn btn-primary" onClick={() => handleDownload('report')}>
            Download Report
          </button>
        </div>
      </div>

      {/* Performance Metrics */}
      <div className="metrics-grid">
        <div className="metric-card">
          <h3>Accuracy</h3>
          <div className="metric-value">{(metrics.accuracy * 100).toFixed(2)}%</div>
          <div className="metric-label">Prediction Accuracy</div>
        </div>
        <div className="metric-card">
          <h3>Sharpe Ratio</h3>
          <div className="metric-value">{metrics.sharpe_ratio.toFixed(3)}</div>
          <div className="metric-label">Risk-Adjusted Return</div>
        </div>
        <div className="metric-card">
          <h3>Max Drawdown</h3>
          <div className="metric-value negative">{(metrics.max_drawdown * 100).toFixed(2)}%</div>
          <div className="metric-label">Maximum Loss</div>
        </div>
        <div className="metric-card">
          <h3>Final PnL</h3>
          <div className={`metric-value ${metrics.final_pnl >= 0 ? 'positive' : 'negative'}`}>
            ${metrics.final_pnl.toFixed(2)}
          </div>
          <div className="metric-label">Profit & Loss</div>
        </div>
      </div>

      {/* Additional Metrics */}
      {metrics.win_rate !== undefined && (
        <div className="card">
          <h3>Trading Statistics</h3>
          <div className="stats-grid">
            <div className="stat-item">
              <span className="stat-label">Win Rate:</span>
              <span className="stat-value">{(metrics.win_rate * 100).toFixed(2)}%</span>
            </div>
            <div className="stat-item">
              <span className="stat-label">Total Trades:</span>
              <span className="stat-value">{metrics.total_trades}</span>
            </div>
            <div className="stat-item">
              <span className="stat-label">Avg Trade PnL:</span>
              <span className={`stat-value ${metrics.avg_trade_pnl >= 0 ? 'positive' : 'negative'}`}>
                ${metrics.avg_trade_pnl.toFixed(2)}
              </span>
            </div>
            <div className="stat-item">
              <span className="stat-label">Volatility:</span>
              <span className="stat-value">{(metrics.volatility * 100).toFixed(2)}%</span>
            </div>
          </div>
        </div>
      )}

      {/* PnL Curve Chart */}
      {results.pnl_curve && results.pnl_curve.length > 0 && (
        <div className="card">
          <h3>Cumulative PnL Curve</h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={results.pnl_curve}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis
                dataKey="timestamp"
                tickFormatter={(value) => new Date(value).toLocaleDateString()}
              />
              <YAxis />
              <Tooltip
                labelFormatter={(value) => new Date(value).toLocaleString()}
                formatter={(value: number) => [`$${value.toFixed(2)}`, 'PnL']}
              />
              <Legend />
              <Line
                type="monotone"
                dataKey="cumulative_pnl"
                stroke="#2196f3"
                name="Cumulative PnL"
                dot={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Prediction Accuracy Chart */}
      {results.prediction_accuracy && results.prediction_accuracy.length > 0 && (
        <div className="card">
          <h3>Prediction Accuracy Over Time</h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={results.prediction_accuracy}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis
                dataKey="timestamp"
                tickFormatter={(value) => new Date(value).toLocaleDateString()}
              />
              <YAxis tickFormatter={(value) => `${(value * 100).toFixed(0)}%`} />
              <Tooltip
                labelFormatter={(value) => new Date(value).toLocaleString()}
                formatter={(value: number) => [`${(value * 100).toFixed(2)}%`, 'Accuracy']}
              />
              <Legend />
              <Line
                type="monotone"
                dataKey="accuracy"
                stroke="#4caf50"
                name="Rolling Accuracy"
                dot={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Feature Importance */}
      {results.feature_importance && results.feature_importance.length > 0 && (
        <div className="card">
          <h3>Feature Importance</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={results.feature_importance.slice(0, 10)}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="feature" angle={-45} textAnchor="end" height={100} />
              <YAxis />
              <Tooltip formatter={(value: number) => value.toFixed(4)} />
              <Legend />
              <Bar dataKey="importance" fill="#ff9800" name="Importance Score" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Model Configuration */}
      <div className="card">
        <h3>Model Configuration</h3>
        <div className="config-display">
          <div className="config-item">
            <span className="config-label">Model Type:</span>
            <span className="config-value">{results.model_type.toUpperCase()}</span>
          </div>
          <div className="config-item">
            <span className="config-label">Dataset:</span>
            <span className="config-value">{results.dataset_id}</span>
          </div>
          {results.config && Object.entries(results.config).map(([key, value]) => (
            <div key={key} className="config-item">
              <span className="config-label">{key}:</span>
              <span className="config-value">{JSON.stringify(value)}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default ResultsPage;
