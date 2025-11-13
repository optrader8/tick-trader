/**
 * Results page - Display analysis results
 */

import React from 'react';
import { useParams } from 'react-router-dom';

const ResultsPage: React.FC = () => {
  const { jobId } = useParams();

  return (
    <div className="results-page">
      <h2>Analysis Results</h2>
      <p>Job ID: {jobId}</p>
      <div className="card">
        <p>Results visualization will be implemented here.</p>
        <ul>
          <li>Performance metrics (Sharpe Ratio, Max Drawdown, etc.)</li>
          <li>PnL curve chart</li>
          <li>Prediction accuracy charts</li>
          <li>Download options (CSV, JSON, Report)</li>
        </ul>
      </div>
    </div>
  );
};

export default ResultsPage;
