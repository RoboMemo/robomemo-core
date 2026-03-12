import { useEffect, useRef, useCallback, useState } from 'react';

const WS_URL = import.meta.env.VITE_WS_BASE || 'ws://localhost:3001';

export interface WebSocketMessage {
  type: string;
  [key: string]: any;
}

export class WebSocketService {
  private ws: WebSocket | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 1000;
  private messageHandlers: Map<string, ((data: any) => void)[]> = new Map();
  private clientType: string = 'visualization';

  constructor(clientType: string = 'visualization') {
    this.clientType = clientType;
  }

  connect(): void {
    if (this.ws?.readyState === WebSocket.OPEN) return;

    this.ws = new WebSocket(WS_URL);

    this.ws.onopen = () => {
      console.log('WebSocket connected');
      this.reconnectAttempts = 0;
      
      // Register client type
      this.send({
        type: 'register',
        clientType: this.clientType
      });
    };

    this.ws.onmessage = (event) => {
      try {
        const message = JSON.parse(event.data);
        this.handleMessage(message);
      } catch (err) {
        console.error('Failed to parse WebSocket message:', err);
      }
    };

    this.ws.onclose = () => {
      console.log('WebSocket disconnected');
      this.attemptReconnect();
    };

    this.ws.onerror = (error) => {
      console.error('WebSocket error:', error);
    };
  }

  disconnect(): void {
    this.ws?.close();
    this.ws = null;
  }

  private attemptReconnect(): void {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      console.error('Max reconnection attempts reached');
      return;
    }

    this.reconnectAttempts++;
    const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1);
    
    console.log(`Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts})`);
    
    setTimeout(() => {
      this.connect();
    }, delay);
  }

  send(message: WebSocketMessage): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(message));
    } else {
      console.warn('WebSocket not connected, message queued');
    }
  }

  private handleMessage(message: WebSocketMessage): void {
    const handlers = this.messageHandlers.get(message.type) || [];
    handlers.forEach(handler => handler(message));
  }

  on(type: string, handler: (data: any) => void): () => void {
    if (!this.messageHandlers.has(type)) {
      this.messageHandlers.set(type, []);
    }
    
    this.messageHandlers.get(type)!.push(handler);
    
    // Return unsubscribe function
    return () => {
      const handlers = this.messageHandlers.get(type);
      if (handlers) {
        const index = handlers.indexOf(handler);
        if (index > -1) {
          handlers.splice(index, 1);
        }
      }
    };
  }

  // Send sensor data (for data collection)
  sendSensorData(data: any): void {
    this.send({
      type: 'sensor_data',
      timestamp: Date.now(),
      ...data
    });
  }

  // Send simulation frame
  sendSimulationFrame(frame: any): void {
    this.send({
      type: 'simulation_frame',
      timestamp: Date.now(),
      ...frame
    });
  }
}

// React hook for WebSocket
export function useWebSocket(clientType: string = 'visualization') {
  const wsRef = useRef<WebSocketService | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [lastMessage, setLastMessage] = useState<any>(null);

  useEffect(() => {
    wsRef.current = new WebSocketService(clientType);
    
    const unsubscribeConnect = wsRef.current.on('connected', () => {
      setIsConnected(true);
    });

    const unsubscribeMessage = wsRef.current.on('message', (data) => {
      setLastMessage(data);
    });

    wsRef.current.connect();

    return () => {
      unsubscribeConnect();
      unsubscribeMessage();
      wsRef.current?.disconnect();
    };
  }, [clientType]);

  const send = useCallback((message: WebSocketMessage) => {
    wsRef.current?.send(message);
  }, []);

  const on = useCallback((type: string, handler: (data: any) => void) => {
    return wsRef.current?.on(type, handler) || (() => {});
  }, []);

  return {
    isConnected,
    lastMessage,
    send,
    on,
    ws: wsRef.current
  };
}

export default WebSocketService;
