/**
 * Upload page - File upload with metadata preview
 */

import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { toast } from 'react-toastify';
import { apiService } from '../services/api';
import { useStore } from '../store/appStore';
import { FILE_UPLOAD, FILE_TYPE_INFO } from '../config/constants';
import { FileUploadResponse } from '../types';

const UploadPage: React.FC = () => {
  const navigate = useNavigate();
  const { addFile } = useStore();
  const [uploading, setUploading] = useState(false);
  const [dragActive, setDragActive] = useState(false);
  const [uploadResult, setUploadResult] = useState<FileUploadResponse | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);

  const validateFile = (file: File): string | null => {
    const fileName = file.name.toLowerCase();
    let fileExt = '.' + fileName.split('.').pop();

    // Handle .tar.gz specially
    if (fileName.endsWith('.tar.gz')) {
      fileExt = '.tar.gz';
    }

    if (!FILE_UPLOAD.ALLOWED_TYPES.includes(fileExt)) {
      return `File type not allowed. Allowed types: ${FILE_UPLOAD.ALLOWED_TYPES.join(', ')}`;
    }

    if (file.size > FILE_UPLOAD.MAX_SIZE) {
      return `File too large. Max size: ${FILE_UPLOAD.MAX_SIZE / 1024 / 1024}MB`;
    }

    return null;
  };

  const formatFileSize = (bytes: number): string => {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(2) + ' KB';
    if (bytes < 1024 * 1024 * 1024) return (bytes / (1024 * 1024)).toFixed(2) + ' MB';
    return (bytes / (1024 * 1024 * 1024)).toFixed(2) + ' GB';
  };

  const getFileTypeInfo = (fileType: string) => {
    return FILE_TYPE_INFO[fileType as keyof typeof FILE_TYPE_INFO] || FILE_TYPE_INFO.unknown;
  };

  const handleFileSelect = async (file: File) => {
    const error = validateFile(file);
    if (error) {
      toast.error(error);
      return;
    }

    setSelectedFile(file);
    setUploading(true);
    setUploadResult(null);

    try {
      const response = await apiService.uploadFile(file);
      const result = response.data;

      setUploadResult(result);

      // Add to store
      addFile({
        id: result.file_id,
        filename: result.filename,
        original_name: result.original_name,
        file_size: result.file_size,
        file_type: result.file_type,
        metadata: result.metadata,
        uploaded_at: result.uploaded_at
      });

      toast.success('File uploaded successfully!');

      // If archive with extracted files, show success message
      if (result.extracted_files && result.extracted_files.length > 0) {
        toast.info(`Extracted ${result.extracted_files.length} files from archive`);
      }

    } catch (error: any) {
      toast.error(`Upload failed: ${error.response?.data?.detail || error.message}`);
      setSelectedFile(null);
    } finally {
      setUploading(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileSelect(e.dataTransfer.files[0]);
    }
  };

  const handleReset = () => {
    setSelectedFile(null);
    setUploadResult(null);
  };

  const handleNavigateToAnalysis = () => {
    navigate('/analysis');
  };

  return (
    <div className="upload-page">
      <h2>Upload Data Files</h2>

      {!uploadResult ? (
        <>
          <div
            className={`upload-zone ${dragActive ? 'active' : ''} ${uploading ? 'uploading' : ''}`}
            onDragOver={(e) => { e.preventDefault(); setDragActive(true); }}
            onDragLeave={() => setDragActive(false)}
            onDrop={handleDrop}
          >
            <div className="upload-icon">📁</div>
            <p>Drag & drop file here or click to browse</p>
            <p className="upload-hint">
              Supported: CSV, JSON, TXT, Parquet, ZIP, TAR, GZ
            </p>
            <input
              type="file"
              accept={FILE_UPLOAD.ALLOWED_TYPES.join(',')}
              onChange={(e) => e.target.files && handleFileSelect(e.target.files[0])}
              disabled={uploading}
            />
            {uploading && (
              <div className="upload-progress">
                <div className="spinner" />
                <p>Uploading {selectedFile?.name}...</p>
              </div>
            )}
          </div>

          <div className="card">
            <h3>Supported File Formats</h3>
            <div className="file-types-grid">
              {Object.entries(FILE_TYPE_INFO).filter(([key]) => key !== 'unknown').map(([type, info]) => (
                <div key={type} className="file-type-item">
                  <span className="file-type-icon">{info.icon}</span>
                  <span className="file-type-label">{info.label}</span>
                </div>
              ))}
            </div>
          </div>
        </>
      ) : (
        <div className="upload-success">
          <div className="card">
            <div className="success-icon">✅</div>
            <h3>Upload Successful!</h3>

            <div className="file-info">
              <div className="file-info-header">
                <span className="file-type-badge" style={{ backgroundColor: getFileTypeInfo(uploadResult.file_type).color }}>
                  {getFileTypeInfo(uploadResult.file_type).icon} {getFileTypeInfo(uploadResult.file_type).label}
                </span>
                <span className="file-name">{uploadResult.original_name}</span>
              </div>

              <div className="file-details">
                <div className="detail-item">
                  <span className="detail-label">File Size:</span>
                  <span className="detail-value">{formatFileSize(uploadResult.file_size)}</span>
                </div>
                <div className="detail-item">
                  <span className="detail-label">Checksum:</span>
                  <span className="detail-value checksum">{uploadResult.checksum.substring(0, 16)}...</span>
                </div>
                <div className="detail-item">
                  <span className="detail-label">Uploaded:</span>
                  <span className="detail-value">{new Date(uploadResult.uploaded_at).toLocaleString()}</span>
                </div>
              </div>

              {/* Metadata Preview */}
              {uploadResult.metadata && Object.keys(uploadResult.metadata).length > 0 && (
                <div className="metadata-preview">
                  <h4>Metadata</h4>
                  {uploadResult.metadata.row_count !== undefined && (
                    <div className="metadata-item">
                      <strong>Rows:</strong> {uploadResult.metadata.row_count.toLocaleString()}
                    </div>
                  )}
                  {uploadResult.metadata.column_count !== undefined && (
                    <div className="metadata-item">
                      <strong>Columns:</strong> {uploadResult.metadata.column_count}
                    </div>
                  )}
                  {uploadResult.metadata.columns && uploadResult.metadata.columns.length > 0 && (
                    <div className="metadata-item">
                      <strong>Column Names:</strong>
                      <div className="column-tags">
                        {uploadResult.metadata.columns.slice(0, 10).map((col: string, idx: number) => (
                          <span key={idx} className="column-tag">{col}</span>
                        ))}
                        {uploadResult.metadata.columns.length > 10 && (
                          <span className="column-tag">+{uploadResult.metadata.columns.length - 10} more</span>
                        )}
                      </div>
                    </div>
                  )}
                  {uploadResult.metadata.line_count !== undefined && (
                    <div className="metadata-item">
                      <strong>Lines:</strong> {uploadResult.metadata.line_count.toLocaleString()}
                    </div>
                  )}
                  {uploadResult.metadata.file_count !== undefined && (
                    <div className="metadata-item">
                      <strong>Files in Archive:</strong> {uploadResult.metadata.file_count}
                    </div>
                  )}
                </div>
              )}

              {/* Extracted Files */}
              {uploadResult.extracted_files && uploadResult.extracted_files.length > 0 && (
                <div className="extracted-files">
                  <h4>Extracted Files ({uploadResult.extracted_files.length})</h4>
                  <div className="extracted-files-list">
                    {uploadResult.extracted_files.map((file, idx) => (
                      <div key={idx} className="extracted-file-item">
                        <span className="file-type-badge small" style={{ backgroundColor: getFileTypeInfo(file.file_type).color }}>
                          {getFileTypeInfo(file.file_type).icon}
                        </span>
                        <span className="extracted-file-name">{file.filename}</span>
                        <span className="extracted-file-size">{formatFileSize(file.file_size)}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>

            <div className="action-buttons">
              <button className="btn btn-secondary" onClick={handleReset}>
                Upload Another File
              </button>
              <button className="btn btn-primary" onClick={handleNavigateToAnalysis}>
                Start Analysis
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default UploadPage;
