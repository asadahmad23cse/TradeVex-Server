"use client";

import { useEffect, useMemo, useState } from "react";

import { useLiveSignal } from "./useLiveSignal";

function colorClass(v: string): string {
  if (v === "LONG") return "cLong";
  if (v === "SHORT") return "cShort";
  return "cHold";
}

function secondsUntil(ts: string): number {
  if (!ts) return 0;
  const target = Date.parse(ts);
  if (Number.isNaN(target)) return 0;
  return Math.max(0, Math.floor((target - Date.now()) / 1000));
}

export default function SignalPanel() {
  const { signal, status, blink } = useLiveSignal();
  const [, setTick] = useState(0);

  useEffect(() => {
    const t = setInterval(() => setTick((x) => x + 1), 1000);
    return () => clearInterval(t);
  }, []);

  const confStyle = useMemo(() => {
    const width = `${Math.max(0, Math.min(100, signal.confidence))}%`;
    const gradient = signal.signal === "SHORT" ? "linear-gradient(90deg, #ff1744, #ff8a80)" : "linear-gradient(90deg, #00c853, #69f0ae)";
    return { width, background: gradient };
  }, [signal.confidence, signal.signal]);

  const stackedStyle = useMemo(() => {
    const width = `${Math.max(0, Math.min(100, signal.stacked_probability))}%`;
    return { width, background: "linear-gradient(90deg, #2979ff, #80d8ff)" };
  }, [signal.stacked_probability]);

  const staleLeft = secondsUntil(signal.stale_after_utc);

  return (
    <section className={`panel ${blink ? "blink" : ""}`}>
      <header className="panelHeader">
        <h2>Real-Time Signal (Your Algo)</h2>
        <span className="status">{status}</span>
      </header>

      <div className="gridRows">
        <div className="k">Signal</div><div className={`v ${colorClass(signal.signal)}`}>{signal.signal}</div>
        <div className="k">Validated</div><div className={`v ${colorClass(signal.validated)}`}>{signal.validated}</div>
        <div className="k">Confidence</div><div className="v">{signal.confidence}%</div>
        <div className="k">Stacked Prob</div><div className="v">{signal.stacked_probability}%</div>
        <div className="k">Alpha Score</div><div className="v">{signal.alpha_score}</div>
        <div className="k">Net Alpha</div><div className="v">{signal.net_alpha}</div>
        <div className="k">Entry</div><div className="v">{signal.entry_zone[0]} - {signal.entry_zone[1]}</div>
        <div className="k">Stop Loss</div><div className="v">{signal.stop_loss}</div>
        <div className="k">Take Profit</div>
        <div className="v stackV">
          <span>TP1: {signal.take_profit.TP1}</span>
          <span>TP2: {signal.take_profit.TP2}</span>
          <span>TP3: {signal.take_profit.TP3}</span>
        </div>
        <div className="k">Algo</div><div className="v">{signal.algo}</div>
        <div className="k">Strategy</div><div className="v">{signal.strategy}</div>
        <div className="k">Factors</div><div className="v">{signal.factors_present.join(", ") || "-"}</div>
        <div className="k">System Status</div>
        <div className="v stackV">
          <span>Win Rate: {(signal.system_status.recent_win_rate * 100).toFixed(1)}%</span>
          <span>Drawdown: {signal.system_status.current_drawdown_pct.toFixed(2)}%</span>
          <span>Heat: {signal.system_status.portfolio_heat_pct.toFixed(2)}%</span>
        </div>
        <div className="k">As Of UTC</div><div className="v">{signal.as_of_utc.replace("T", " ").replace("Z", "")}</div>
        <div className="k">Stale After</div><div className="v">{signal.stale_after_utc.replace("T", " ").replace("Z", "")} ({staleLeft}s)</div>
      </div>

      <div className="progressWrap">
        <span>Confidence</span>
        <div className="progressTrack"><div className="progressFill" style={confStyle} /></div>
      </div>
      <div className="progressWrap compact">
        <span>Stacked Probability</span>
        <div className="progressTrack"><div className="progressFill" style={stackedStyle} /></div>
      </div>

      {signal.signal === "HOLD" && <div className="holdBanner">NO TRADE CONDITIONS MET: {signal.reason}</div>}
      {signal.system_status.auto_pause && <div className="pauseBanner">SYSTEM PAUSED: performance guard active</div>}
    </section>
  );
}
