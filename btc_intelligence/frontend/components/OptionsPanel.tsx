"use client";

import { useEffect, useState } from "react";

import { SignalPayload } from "./types";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:9000";

export default function OptionsPanel() {
  const [signal, setSignal] = useState<SignalPayload | null>(null);

  useEffect(() => {
    let timer: ReturnType<typeof setInterval> | null = null;
    const load = async () => {
      const res = await fetch(`${API_BASE}/signal`, { cache: "no-store" });
      if (!res.ok) return;
      setSignal((await res.json()) as SignalPayload);
    };
    void load();
    timer = setInterval(load, 30000);
    return () => {
      if (timer) clearInterval(timer);
    };
  }, []);

  return (
    <section className="card">
      <h2>Options Flow Dashboard</h2>
      {!signal && <p>Loading...</p>}
      {signal && (
        <div className="tableLike">
          <div className="row"><span>Max Pain</span><strong>{signal.derivatives.max_pain}</strong></div>
          <div className="row"><span>Put/Call Ratio</span><strong>{signal.derivatives.put_call_ratio}</strong></div>
          <div className="row"><span>IV Skew</span><strong>{signal.derivatives.iv_skew}</strong></div>
          <div className="row"><span>Funding Rate</span><strong>{signal.derivatives.funding_rate}</strong></div>
          <div className="row"><span>Funding Momentum</span><strong>{signal.derivatives.funding_momentum}</strong></div>
          <div className="row"><span>OI Change 1H</span><strong>{signal.derivatives.oi_change_1h}</strong></div>
          <div className="row"><span>Liq Magnet</span><strong>{signal.derivatives.liq_magnet}</strong></div>
          <div className="row"><span>L/S Ratio</span><strong>{signal.derivatives.ls_ratio}</strong></div>
        </div>
      )}
    </section>
  );
}
