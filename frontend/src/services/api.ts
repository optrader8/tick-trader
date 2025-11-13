/**
 * API service for backend communication
 */

import axios, { AxiosInstance } from 'axios';
import { API_BASE_URL } from '../config/constants';
import {
  FileUploadResponse,
  FileMetadata,
  StartAnalysisRequest,
  Analysis,
  AnalysisResults,
  SystemStatus,
  ApiResponse
} from '../types';

class ApiService {
  private client: AxiosInstance;

  constructor() {
    this.client = axios.create({
      baseURL: API_BASE_URL,
      timeout: 30000,
      headers: {
        'Content-Type': 'application/json'
      }
    });

    // Response interceptor for error handling
    this.client.interceptors.response.use(
      (response) => response,
      (error) => {
        console.error('API Error:', error.response?.data || error.message);
        return Promise.reject(error);
      }
    );
  }

  // File APIs
  async uploadFile(file: File): Promise<FileUploadResponse> {
    const formData = new FormData();
    formData.append('file', file);

    const response = await this.client.post<FileUploadResponse>(
      '/files/upload',
      formData,
      {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      }
    );
    return response.data;
  }

  async listFiles(): Promise<FileMetadata[]> {
    const response = await this.client.get<{ files: FileMetadata[] }>('/files/list');
    return response.data.files;
  }

  async deleteFile(fileId: string): Promise<void> {
    await this.client.delete(`/files/${fileId}`);
  }

  // Analysis APIs
  async startAnalysis(request: StartAnalysisRequest): Promise<{ job_id: string }> {
    const response = await this.client.post<ApiResponse<{ job_id: string }>>(
      '/analysis/start',
      request
    );
    return response.data.data!;
  }

  async getAnalysisStatus(jobId: string): Promise<Analysis> {
    const response = await this.client.get<ApiResponse<Analysis>>(
      `/analysis/${jobId}/status`
    );
    return response.data.data!;
  }

  async cancelAnalysis(jobId: string): Promise<void> {
    await this.client.post(`/analysis/${jobId}/cancel`);
  }

  async getAnalysisResults(jobId: string): Promise<AnalysisResults> {
    const response = await this.client.get<ApiResponse<AnalysisResults>>(
      `/analysis/${jobId}/results`
    );
    return response.data.data!;
  }

  // System APIs
  async getSystemStatus(): Promise<SystemStatus> {
    const response = await this.client.get<SystemStatus>('/system/status');
    return response.data;
  }

  async healthCheck(): Promise<{ status: string; services: any }> {
    const response = await this.client.get('/system/health');
    return response.data;
  }
}

export const apiService = new ApiService();
