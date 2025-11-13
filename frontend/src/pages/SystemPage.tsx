/**
 * System page - System monitoring and health
 */

import React, { useEffect, useState } from 'react';
import { apiService } from '../services/api';
import { SystemStatus } from '../types';

const SystemPage: React.FC = () => {
  const [status, setStatus] = useState<SystemStatus | null>(null);
  const [health, setHealth] = useState<any>(null);

  useEffect(() => {
    loadData();
    const interval = setInterval(loadData, 5000);
    return () => clearInterval(interval);
  }, []);

  const loadData = async () => {
    try {
      const [statusData, healthData] = await Promise.all([
        apiService.getSystemStatus(),
        apiService.healthCheck()
      ]);
      setStatus(statusData);
      setHealth(healthData);
    } catch (error) {
      console.error('Failed to load system data:', error);
    }
  };

  return (
    <div className="system-page">
      <h2>System Monitoring</h2>

      <div className="card">
        <h3>Health Status</h3>
        {health && (
          <div className="health-grid">
            <div className="health-item">
              <span>Overall:</span>
              <span className={`status-badge status-${health.status === 'healthy' ? 'completed' : 'failed'}`}>
                {health.status}
              </span>
            </div>
            {health.services && Object.entries(health.services).map(([service, status]) => (
              <div key={service} className="health-item">
                <span>{service}:</span>
                <span className={`status-badge status-${status === 'ok' ? 'completed' : 'failed'}`}>
                  {status as string}
                </span>
              </div>
            ))}
          </div>
        )}
      </div>

      {status && (
        <>
          <div className="card">
            <h3>Resource Usage</h3>
            <div className="resource-item">
              <div className="resource-header">
                <span>CPU Usage</span>
                <span>{status.cpu_usage.toFixed(1)}%</span>
              </div>
              <div className="progress-bar">
                <div
                  className="progress-bar-fill"
                  style={{ width: `${status.cpu_usage}%` }}
                />
              </div>
            </div>
            <div className="resource-item">
              <div className="resource-header">
                <span>Memory Usage</span>
                <span>{status.memory_usage.toFixed(1)}%</span>
              </div>
              <div className="progress-bar">
                <div
                  className="progress-bar-fill"
                  style={{ width: `${status.memory_usage}%` }}
                />
              </div>
            </div>
            <div className="resource-item">
              <div className="resource-header">
                <span>Disk Usage</span>
                <span>{status.disk_usage.toFixed(1)}%</span>
              </div>
              <div className="progress-bar">
                <div
                  className="progress-bar-fill"
                  style={{ width: `${status.disk_usage}%` }}
                />
              </div>
            </div>
          </div>

          <div className="card">
            <h3>Running Jobs</h3>
            <p className="stat-value">{status.running_jobs}</p>
          </div>
        </>
      )}
    </div>
  );
};

export default SystemPage;
