/**
 * Main layout component with header and sidebar
 */

import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import './Layout.css';

interface LayoutProps {
  children: React.ReactNode;
}

const Layout: React.FC<LayoutProps> = ({ children }) => {
  const location = useLocation();

  const isActive = (path: string) => {
    return location.pathname === path || location.pathname.startsWith(path + '/');
  };

  return (
    <div className="layout">
      <header className="header">
        <div className="header-content">
          <h1 className="logo">📈 Tick Trader</h1>
          <nav className="header-nav">
            <span className="user-info">Admin</span>
          </nav>
        </div>
      </header>

      <div className="main-container">
        <aside className="sidebar">
          <nav className="sidebar-nav">
            <Link
              to="/dashboard"
              className={`nav-item ${isActive('/dashboard') ? 'active' : ''}`}
            >
              <span className="nav-icon">📊</span>
              <span>Dashboard</span>
            </Link>

            <Link
              to="/upload"
              className={`nav-item ${isActive('/upload') ? 'active' : ''}`}
            >
              <span className="nav-icon">📁</span>
              <span>Upload Data</span>
            </Link>

            <Link
              to="/analysis"
              className={`nav-item ${isActive('/analysis') ? 'active' : ''}`}
            >
              <span className="nav-icon">🤖</span>
              <span>Analysis</span>
            </Link>

            <Link
              to="/history"
              className={`nav-item ${isActive('/history') ? 'active' : ''}`}
            >
              <span className="nav-icon">📜</span>
              <span>History</span>
            </Link>

            <Link
              to="/system"
              className={`nav-item ${isActive('/system') ? 'active' : ''}`}
            >
              <span className="nav-icon">⚙️</span>
              <span>System</span>
            </Link>
          </nav>
        </aside>

        <main className="content">
          {children}
        </main>
      </div>
    </div>
  );
};

export default Layout;
