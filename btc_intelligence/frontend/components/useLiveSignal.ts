"use client";

import { useEffect, useState } from "react";

import { EMPTY_SIGNAL, SignalPayload } from "./types";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:9000";

export function useLiveSignal() {
  const [signal, setSignal] = useState<SignalPayload>(EMPTY_SIGNAL);
  const [status, setStatus] = useState("connecting");
  const [blink, setBlink] = useState(false);

  useEffect(() => {
    let ws: WebSocket | null = null;
    let reconnectTimer: ReturnType<typeof setTimeout> | null = null;
    let pollTimer: ReturnType<typeof setInterval> | null = null;

    const pull = async () => {
      try {
        const res = await fetch(`${API_BASE}/signal`, { cache: "no-store" });
        if (res.ok) {
          const data = (await res.json()) as SignalPayload;
          setSignal(data);
        }
      } catch {
        // silent fallback
      }
    };

    const connect = () => {
      setStatus("connecting");
      ws = new WebSocket(`${API_BASE.replace("http", "ws")}/ws/live`);

      ws.onopen = () => setStatus("live");
      ws.onmessage = (evt) => {
        try {
          const data = JSON.parse(evt.data) as SignalPayload;
          setSignal(data);
          setBlink(true);
          setTimeout(() => setBlink(false), 700);
        } catch {
          // ignore malformed frames
        }
      };
      ws.onclose = () => {
        setStatus("reconnecting");
        reconnectTimer = setTimeout(connect, 3000);
      };
      ws.onerror = () => {
        setStatus("error");
      };
    };

    void pull();
    pollTimer = setInterval(pull, 30000);
    connect();

    return () => {
      if (reconnectTimer) clearTimeout(reconnectTimer);
      if (pollTimer) clearInterval(pollTimer);
      if (ws) ws.close();
    };
  }, []);

  return { signal, status, blink };
}
