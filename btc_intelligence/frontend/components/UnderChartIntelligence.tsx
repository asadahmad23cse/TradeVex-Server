"use client";

import { useEffect, useMemo, useState } from "react";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:9000";

type FlowPayload = {
  decision_state: "FAVOR_LONG" | "FAVOR_SHORT" | "NO_TRADE";
  reason: string;
  obi: number;
  cvd_slope: number;
  absorption_side: string;
  absorption_strength: number;
  stacked_imbalance_direction: string;
};

type VolumeBin = {
  center: number;
  relative: number;
};

type VolumePayload = {
  decision_state: "FAVOR_LONG" | "FAVOR_SHORT" | "NO_TRADE";
  reason: string;
  window_minutes: number;
  trade_count: number;
  poc: number;
  hvn: number[];
  lvn: number[];
  price_to_poc_pct: number;
  bins: VolumeBin[];
};

type VolatilityPayload = {
  volatility_regime: "LOW" | "NORMAL" | "EXPANSION";
  tradeability: "NO_TRADE" | "ALLOW" | "CAUTION";
  reason: string;
  atr14: number;
  atr_multiplier_pct: number;
  raw_regime: string;
};

const EMPTY_FLOW: FlowPayload = {
  decision_state: "NO_TRADE",
  reason: "Loading",
  obi: 0.5,
  cvd_slope: 0,
  absorption_side: "none",
  absorption_strength: 0,
  stacked_imbalance_direction: "none",
};

const EMPTY_PROFILE: VolumePayload = {
  decision_state: "NO_TRADE",
  reason: "Loading",
  window_minutes: 45,
  trade_count: 0,
  poc: 0,
  hvn: [],
  lvn: [],
  price_to_poc_pct: 0,
  bins: [],
};

const EMPTY_VOL: VolatilityPayload = {
  volatility_regime: "LOW",
  tradeability: "NO_TRADE",
  reason: "Loading",
  atr14: 0,
  atr_multiplier_pct: 0,
  raw_regime: "compression",
};

function stateClass(v: string): string {
  if (v === "FAVOR_LONG" || v === "ALLOW") return "icState icLong";
  if (v === "FAVOR_SHORT") return "icState icShort";
  if (v === "CAUTION") return "icState icCaution";
  return "icState icNeutral";
}

export default function UnderChartIntelligence() {
  const [tab, setTab] = useState<"flow" | "volume" | "volatility">("flow");
  const [flow, setFlow] = useState<FlowPayload>(EMPTY_FLOW);
  const [profile, setProfile] = useState<VolumePayload>(EMPTY_PROFILE);
  const [vol, setVol] = useState<VolatilityPayload>(EMPTY_VOL);

  useEffect(() => {
    let timer: ReturnType<typeof setInterval> | null = null;
    const load = async () => {
      try {
        const [f, p, v] = await Promise.all([
          fetch(`${API_BASE}/api/orderflow`, { cache: "no-store" }),
          fetch(`${API_BASE}/api/volume-profile?window_minutes=45&bins=24`, { cache: "no-store" }),
          fetch(`${API_BASE}/api/volatility`, { cache: "no-store" }),
        ]);
        if (f.ok) setFlow((await f.json()) as FlowPayload);
        if (p.ok) setProfile((await p.json()) as VolumePayload);
        if (v.ok) setVol((await v.json()) as VolatilityPayload);
      } catch {
        // keep last good state
      }
    };
    void load();
    timer = setInterval(load, 5000);
    return () => {
      if (timer) clearInterval(timer);
    };
  }, []);

  const volumeBars = useMemo(() => {
    const top = [...profile.bins].sort((a, b) => b.relative - a.relative).slice(0, 12);
    return top.sort((a, b) => a.center - b.center);
  }, [profile.bins]);

  return (
    <section className="intelBox">
      <header className="intelHead">
        <h4>Under-Chart Intelligence</h4>
        <div className="intelTabs">
          <button className={tab === "flow" ? "intelTab active" : "intelTab"} onClick={() => setTab("flow")}>Flow</button>
          <button className={tab === "volume" ? "intelTab active" : "intelTab"} onClick={() => setTab("volume")}>Volume</button>
          <button className={tab === "volatility" ? "intelTab active" : "intelTab"} onClick={() => setTab("volatility")}>Volatility</button>
        </div>
      </header>

      {tab === "flow" && (
        <div className="intelPanel">
          <div className="intelTopRow">
            <span className="icLabel">Decision</span>
            <span className={stateClass(flow.decision_state)}>{flow.decision_state}</span>
          </div>
          <div className="intelReason">{flow.reason}</div>
          <div className="intelGrid">
            <div><span>OBI</span><strong>{flow.obi.toFixed(3)}</strong></div>
            <div><span>CVD slope</span><strong>{flow.cvd_slope.toFixed(5)}</strong></div>
            <div><span>Absorption</span><strong>{flow.absorption_side}</strong></div>
            <div><span>Strength</span><strong>{flow.absorption_strength.toFixed(2)}</strong></div>
          </div>
        </div>
      )}

      {tab === "volume" && (
        <div className="intelPanel">
          <div className="intelTopRow">
            <span className="icLabel">Decision</span>
            <span className={stateClass(profile.decision_state)}>{profile.decision_state}</span>
          </div>
          <div className="intelReason">{profile.reason}</div>
          <div className="intelGrid">
            <div><span>Window</span><strong>{profile.window_minutes}m</strong></div>
            <div><span>Trades</span><strong>{profile.trade_count}</strong></div>
            <div><span>POC</span><strong>{profile.poc.toFixed(2)}</strong></div>
            <div><span>Price vs POC</span><strong>{profile.price_to_poc_pct.toFixed(3)}%</strong></div>
          </div>
          <div className="intelListRow">
            <span>HVN: {profile.hvn.join(", ") || "-"}</span>
            <span>LVN: {profile.lvn.join(", ") || "-"}</span>
          </div>
          <div className="vpBars">
            {volumeBars.map((b) => (
              <div key={`${b.center}`} className="vpBarRow">
                <span>{b.center.toFixed(1)}</span>
                <div className="vpTrack"><div className="vpFill" style={{ width: `${Math.min(100, b.relative * 100)}%` }} /></div>
              </div>
            ))}
          </div>
        </div>
      )}

      {tab === "volatility" && (
        <div className="intelPanel">
          <div className="intelTopRow">
            <span className="icLabel">Tradeability</span>
            <span className={stateClass(vol.tradeability)}>{vol.tradeability}</span>
          </div>
          <div className="intelReason">{vol.reason}</div>
          <div className="intelGrid">
            <div><span>Regime</span><strong>{vol.volatility_regime}</strong></div>
            <div><span>Raw</span><strong>{vol.raw_regime}</strong></div>
            <div><span>ATR14</span><strong>{vol.atr14.toFixed(3)}</strong></div>
            <div><span>ATR%</span><strong>{vol.atr_multiplier_pct.toFixed(3)}%</strong></div>
          </div>
        </div>
      )}
    </section>
  );
}

