/**
 * History page - Browse past analyses
 */

import React, { useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useStore } from '../store/appStore';

const HistoryPage: React.FC = () => {
  const navigate = useNavigate();
  const { analyses } = useStore();

  const analysesList = Array.from(analyses.values());

  return (
    <div className="history-page">
      <h2>Analysis History</h2>

      {analysesList.length === 0 ? (
        <div className="card">
          <p>No analyses yet.</p>
        </div>
      ) : (
        <div className="card">
          <table className="history-table">
            <thead>
              <tr>
                <th>Job ID</th>
                <th>Model</th>
                <th>Status</th>
                <th>Progress</th>
                <th>Started</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {analysesList.map((analysis) => (
                <tr key={analysis.id}>
                  <td>{analysis.id.substring(0, 8)}</td>
                  <td>{analysis.model_type.toUpperCase()}</td>
                  <td>
                    <span className={`status-badge status-${analysis.status}`}>
                      {analysis.status}
                    </span>
                  </td>
                  <td>{analysis.progress}%</td>
                  <td>{new Date(analysis.started_at || '').toLocaleString()}</td>
                  <td>
                    <button
                      className="btn btn-primary"
                      onClick={() => navigate(`/results/${analysis.id}`)}
                    >
                      View
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
};

export default HistoryPage;
