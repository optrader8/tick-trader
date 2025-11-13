/**
 * Upload page - File upload interface
 */

import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { toast } from 'react-toastify';
import { apiService } from '../services/api';
import { useStore } from '../store/appStore';

const UploadPage: React.FC = () => {
  const navigate = useNavigate();
  const { addFile } = useStore();
  const [uploading, setUploading] = useState(false);
  const [dragActive, setDragActive] = useState(false);

  const handleFileSelect = async (file: File) => {
    setUploading(true);
    try {
      const result = await apiService.uploadFile(file);
      addFile({
        id: result.file_id,
        filename: result.filename,
        original_name: result.original_name,
        file_size: result.file_size,
        uploaded_at: result.uploaded_at
      });
      toast.success('File uploaded successfully!');
      setTimeout(() => navigate('/analysis'), 1000);
    } catch (error: any) {
      toast.error(`Upload failed: ${error.response?.data?.detail || error.message}`);
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

  return (
    <div className="upload-page">
      <h2>Upload Tick Data</h2>
      <div
        className={`upload-zone ${dragActive ? 'active' : ''}`}
        onDragOver={(e) => { e.preventDefault(); setDragActive(true); }}
        onDragLeave={() => setDragActive(false)}
        onDrop={handleDrop}
      >
        <div className="upload-icon">📁</div>
        <p>Drag & drop file here or click to browse</p>
        <input
          type="file"
          accept=".csv,.parquet,.json"
          onChange={(e) => e.target.files && handleFileSelect(e.target.files[0])}
          disabled={uploading}
        />
        {uploading && <div className="spinner" />}
      </div>
    </div>
  );
};

export default UploadPage;
