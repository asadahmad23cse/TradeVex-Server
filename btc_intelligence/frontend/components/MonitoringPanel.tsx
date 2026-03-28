"use client";

import { useEffect, useState } from "react";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:9000";

type MonitoringStats = {
  recent_win_rate?: number;
  current_drawdown_pct?: number;
  portfolio_heat_pct?: number;
  auto_pause?: boolean;
  auto_pause_reason?: string;
  loss_streak?: number;
  trades_count?: number;
  avg_rr?: number;
  avg_mae?: number;
  avg_mfe?: number;
  calibrated?: boolean;
  confidence_gap?: number;
};

export default function MonitoringPanel() {
  const [stats, setStats] = useState<MonitoringStats>({});
  const [edges, setEdges] = useState<Record<string, number>>({});

  useEffect(() => {
    let timer: ReturnType<typeof setInterval> | null = null;
    const load = async () => {
      const [s, e] = await Promise.all([
        fetch(`${API_BASE}/monitoring/stats`, { cache: "no-store" }),
        fetch(`${API_BASE}/monitoring/edges`, { cache: "no-store" }),
      ]);
      if (s.ok) setStats(await s.json());
      if (e.ok) {
        const payload = await e.json();
        setEdges(payload.edges ?? payload.priors ?? {});
      }
    };
    void load();
    timer = setInterval(load, 30000);
    return () => {
      if (timer) clearInterval(timer);
    };
  }, []);

  const edgeRows = Object.entries(edges).sort((a, b) => b[1] - a[1]).slice(0, 12);

  return (
    <section className="card">
      <h2>System Monitoring</h2>
      <div className="tableLike">
        <div className="row"><span>Win Rate</span><strong>{((stats.recent_win_rate ?? 0) * 100).toFixed(2)}%</strong></div>
        <div className="row"><span>Drawdown</span><strong>{(stats.current_drawdown_pct ?? 0).toFixed(2)}%</strong></div>
        <div className="row"><span>Portfolio Heat</span><strong>{(stats.portfolio_heat_pct ?? 0).toFixed(2)}%</strong></div>
        <div className="row"><span>Auto Pause</span><strong>{stats.auto_pause ? "ON" : "OFF"}</strong></div>
        <div className="row"><span>Loss Streak</span><strong>{stats.loss_streak ?? 0}</strong></div>
        <div className="row"><span>Trades Count</span><strong>{stats.trades_count ?? 0}</strong></div>
        <div className="row"><span>Avg RR</span><strong>{(stats.avg_rr ?? 0).toFixed(2)}</strong></div>
        <div className="row"><span>Avg MAE</span><strong>{(stats.avg_mae ?? 0).toFixed(2)}</strong></div>
        <div className="row"><span>Avg MFE</span><strong>{(stats.avg_mfe ?? 0).toFixed(2)}</strong></div>
        <div className="row"><span>Calibrated</span><strong>{stats.calibrated ? "YES" : "NO"}</strong></div>
        <div className="row"><span>Confidence Gap</span><strong>{(stats.confidence_gap ?? 0).toFixed(3)}</strong></div>
      </div>

      <h3>Independent Edges</h3>
      <div className="tableLike">
        {edgeRows.map(([k, v]) => (
          <div key={k} className="row">
            <span>{k}</span>
            <strong>{(v * 100).toFixed(1)}%</strong>
          </div>
        ))}
      </div>
    </section>
  );
}
