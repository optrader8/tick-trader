/**
 * WebSocket service for real-time updates
 */

import { io, Socket } from 'socket.io-client';
import { WS_BASE_URL } from '../config/constants';
import { Analysis } from '../types';

type AnalysisUpdateCallback = (analysis: Analysis) => void;

class WebSocketService {
  private sockets: Map<string, Socket> = new Map();

  connectToAnalysis(jobId: string, onUpdate: AnalysisUpdateCallback): () => void {
    // Check if already connected
    if (this.sockets.has(jobId)) {
      console.warn(`Already connected to job ${jobId}`);
      return () => this.disconnect(jobId);
    }

    // Create WebSocket connection
    const wsUrl = `${WS_BASE_URL}/analysis/ws/${jobId}`;
    const socket = io(wsUrl, {
      transports: ['websocket'],
      reconnection: true,
      reconnectionAttempts: 5,
      reconnectionDelay: 1000
    });

    socket.on('connect', () => {
      console.log(`WebSocket connected for job ${jobId}`);
    });

    socket.on('disconnect', () => {
      console.log(`WebSocket disconnected for job ${jobId}`);
    });

    socket.on('message', (data: Analysis) => {
      onUpdate(data);
    });

    socket.on('error', (error: any) => {
      console.error(`WebSocket error for job ${jobId}:`, error);
    });

    this.sockets.set(jobId, socket);

    // Return cleanup function
    return () => this.disconnect(jobId);
  }

  disconnect(jobId: string): void {
    const socket = this.sockets.get(jobId);
    if (socket) {
      socket.disconnect();
      this.sockets.delete(jobId);
      console.log(`Disconnected from job ${jobId}`);
    }
  }

  disconnectAll(): void {
    this.sockets.forEach((socket, jobId) => {
      this.disconnect(jobId);
    });
  }
}

export const wsService = new WebSocketService();
