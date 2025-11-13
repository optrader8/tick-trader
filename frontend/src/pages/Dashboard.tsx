/**
 * Dashboard page - overview of system and recent activities
 */

import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { apiService } from '../services/api';
import { useStore } from '../store/appStore';
import { SystemStatus, Analysis } from '../types';
import { STATUS_COLORS } from '../config/constants';
import './Dashboard.css';

const Dashboard: React.FC = () => {
  const navigate = useNavigate();
  const { files, systemStatus, setSystemStatus, analyses } = useStore();

  const [loading, setLoading] = useState(true);
  const [recentAnalyses, setRecentAnalyses] = useState<Analysis[]>([]);

  useEffect(() => {
    loadData();
    const interval = setInterval(loadSystemStatus, 5000); // Update every 5s
    return () => clearInterval(interval);
  }, []);

  const loadData = async () => {
    try {
      await Promise.all([
        loadSystemStatus(),
        // Load recent analyses would go here
      ]);
    } catch (error) {
      console.error('Failed to load dashboard data:', error);
    } finally {
      setLoading(false);
    }
  };

  const loadSystemStatus = async () => {
    try {
      const status = await apiService.getSystemStatus();
      setSystemStatus(status);
    } catch (error) {
      console.error('Failed to load system status:', error);
    }
  };

  useEffect(() => {
    // Convert analyses map to array for display
    const analysesArray = Array.from(analyses.values());
    const sorted = analysesArray.sort((a, b) =>
      new Date(b.started_at || '').getTime() - new Date(a.started_at || '').getTime()
    );
    setRecentAnalyses(sorted.slice(0, 5));
  }, [analyses]);

  if (loading) {
    return <div className="spinner" />;
  }

  return (
    <div className="dashboard">
      <div className="dashboard-header">
        <h2>Dashboard</h2>
        <p className="subtitle">Overview of your tick trading system</p>
      </div>

      {/* Quick Stats */}
      <div className="stats-grid">
        <div className="stat-card">
          <div className="stat-icon">📁</div>
          <div className="stat-content">
            <div className="stat-value">{files.length}</div>
            <div className="stat-label">Uploaded Files</div>
          </div>
        </div>

        <div className="stat-card">
          <div className="stat-icon">🤖</div>
          <div className="stat-content">
            <div className="stat-value">{analyses.size}</div>
            <div className="stat-label">Total Analyses</div>
          </div>
        </div>

        <div className="stat-card">
          <div className="stat-icon">⚡</div>
          <div className="stat-content">
            <div className="stat-value">{systemStatus?.running_jobs || 0}</div>
            <div className="stat-label">Running Jobs</div>
          </div>
        </div>

        <div className="stat-card">
          <div className="stat-icon">💻</div>
          <div className="stat-content">
            <div className="stat-value">
              {systemStatus?.cpu_usage.toFixed(1)}%
            </div>
            <div className="stat-label">CPU Usage</div>
          </div>
        </div>
      </div>

      {/* System Resources */}
      {systemStatus && (
        <div className="card">
          <h3>System Resources</h3>
          <div className="resources-grid">
            <div className="resource-item">
              <div className="resource-header">
                <span>CPU Usage</span>
                <span className="resource-value">{systemStatus.cpu_usage.toFixed(1)}%</span>
              </div>
              <div className="progress-bar">
                <div
                  className="progress-bar-fill"
                  style={{ width: `${systemStatus.cpu_usage}%` }}
                />
              </div>
            </div>

            <div className="resource-item">
              <div className="resource-header">
                <span>Memory Usage</span>
                <span className="resource-value">{systemStatus.memory_usage.toFixed(1)}%</span>
              </div>
              <div className="progress-bar">
                <div
                  className="progress-bar-fill"
                  style={{ width: `${systemStatus.memory_usage}%` }}
                />
              </div>
            </div>

            <div className="resource-item">
              <div className="resource-header">
                <span>Disk Usage</span>
                <span className="resource-value">{systemStatus.disk_usage.toFixed(1)}%</span>
              </div>
              <div className="progress-bar">
                <div
                  className="progress-bar-fill"
                  style={{ width: `${systemStatus.disk_usage}%` }}
                />
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Recent Analyses */}
      <div className="card">
        <div className="card-header">
          <h3>Recent Analyses</h3>
          <button
            className="btn btn-primary"
            onClick={() => navigate('/analysis')}
          >
            Start New Analysis
          </button>
        </div>

        {recentAnalyses.length === 0 ? (
          <div className="empty-state">
            <p>No analyses yet. Start your first analysis!</p>
            <button
              className="btn btn-primary"
              onClick={() => navigate('/analysis')}
            >
              Get Started
            </button>
          </div>
        ) : (
          <div className="analyses-list">
            {recentAnalyses.map((analysis) => (
              <div
                key={analysis.id}
                className="analysis-item"
                onClick={() => navigate(`/analysis/${analysis.id}`)}
              >
                <div className="analysis-info">
                  <div className="analysis-name">
                    Analysis #{analysis.id.substring(0, 8)}
                  </div>
                  <div className="analysis-meta">
                    Model: {analysis.model_type.toUpperCase()} •{' '}
                    {new Date(analysis.started_at || '').toLocaleString()}
                  </div>
                </div>
                <div className="analysis-status">
                  <span
                    className={`status-badge status-${analysis.status}`}
                  >
                    {analysis.status}
                  </span>
                  {analysis.status === 'running' && (
                    <div className="progress-indicator">
                      {analysis.progress}%
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Quick Actions */}
      <div className="quick-actions">
        <button
          className="action-card"
          onClick={() => navigate('/upload')}
        >
          <div className="action-icon">📁</div>
          <div className="action-title">Upload Data</div>
          <div className="action-description">
            Upload tick data files for analysis
          </div>
        </button>

        <button
          className="action-card"
          onClick={() => navigate('/analysis')}
        >
          <div className="action-icon">🤖</div>
          <div className="action-title">Start Analysis</div>
          <div className="action-description">
            Configure and run AI analysis
          </div>
        </button>

        <button
          className="action-card"
          onClick={() => navigate('/history')}
        >
          <div className="action-icon">📊</div>
          <div className="action-title">View Results</div>
          <div className="action-description">
            Browse analysis results and metrics
          </div>
        </button>
      </div>
    </div>
  );
};

export default Dashboard;
