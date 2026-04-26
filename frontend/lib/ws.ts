import { useEffect, useRef, useCallback } from 'react';
import { StepResponse } from './types';

const INITIAL_RETRY_DELAY = 2000; // 2 seconds
const MAX_RETRY_DELAY = 10000; // 10 seconds

export interface UseRunStreamOptions {
  runId: string;
  onMessage: (data: StepResponse) => void;
  onConnected: () => void;
  onDisconnected: () => void;
  onError: (error: string) => void;
}

/**
 * WebSocket hook for streaming run data
 * Auto-reconnects with exponential backoff (2s → 10s)
 */
export function useRunStream({
  runId,
  onMessage,
  onConnected,
  onDisconnected,
  onError,
}: UseRunStreamOptions) {
  const wsRef = useRef<WebSocket | null>(null);
  const retryCountRef = useRef(0);
  const retryTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const isUnmountingRef = useRef(false);

  const getRetryDelay = useCallback((): number => {
    return Math.min(
      INITIAL_RETRY_DELAY * Math.pow(1.5, retryCountRef.current),
      MAX_RETRY_DELAY
    );
  }, []);

  const closeWebSocket = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
  }, []);

  const connect = useCallback(() => {
    if (isUnmountingRef.current) return;

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//localhost:5000/ws/stream`;

    try {
      const ws = new WebSocket(wsUrl);

      ws.onopen = () => {
        if (isUnmountingRef.current) {
          ws.close();
          return;
        }
        retryCountRef.current = 0;
        onConnected();
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data) as StepResponse;
          onMessage(data);
        } catch (e) {
          onError(`Failed to parse WebSocket message: ${e}`);
        }
      };

      ws.onerror = (event) => {
        onError(`WebSocket error: ${event.type}`);
      };

      ws.onclose = () => {
        if (isUnmountingRef.current) return;

        onDisconnected();

        // Schedule reconnection with exponential backoff
        if (!isUnmountingRef.current) {
          const delay = getRetryDelay();
          retryCountRef.current += 1;
          retryTimeoutRef.current = setTimeout(() => {
            if (!isUnmountingRef.current) {
              connect();
            }
          }, delay);
        }
      };

      wsRef.current = ws;
    } catch (e) {
      onError(`Failed to connect WebSocket: ${e}`);
      // Schedule retry
      const delay = getRetryDelay();
      retryCountRef.current += 1;
      retryTimeoutRef.current = setTimeout(() => {
        if (!isUnmountingRef.current) {
          connect();
        }
      }, delay);
    }
  }, [runId, onMessage, onConnected, onDisconnected, onError, getRetryDelay]);

  useEffect(() => {
    isUnmountingRef.current = false;
    retryCountRef.current = 0;

    connect();

    return () => {
      isUnmountingRef.current = true;

      if (retryTimeoutRef.current) {
        clearTimeout(retryTimeoutRef.current);
      }

      closeWebSocket();
    };
  }, [runId, connect]);

  return {
    isConnected: wsRef.current?.readyState === WebSocket.OPEN,
  };
}
