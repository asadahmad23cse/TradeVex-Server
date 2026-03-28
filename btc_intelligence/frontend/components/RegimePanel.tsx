"use client";

import { useEffect, useState } from "react";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:9000";

export default function RegimePanel() {
  const [regime, setRegime] = useState<{ market_regime: string; as_of_utc: string }>({ market_regime: "unknown", as_of_utc: "" });
  const [health, setHealth] = useState<Record<string, unknown>>({});

  useEffect(() => {
    let timer: ReturnType<typeof setInterval> | null = null;
    const load = async () => {
      const [r1, r2] = await Promise.all([
        fetch(`${API_BASE}/regime`, { cache: "no-store" }),
        fetch(`${API_BASE}/health`, { cache: "no-store" })
      ]);
      if (r1.ok) setRegime(await r1.json());
      if (r2.ok) setHealth(await r2.json());
    };
    void load();
    timer = setInterval(load, 30000);
    return () => {
      if (timer) clearInterval(timer);
    };
  }, []);

  return (
    <section className="card">
      <h2>Market Regime</h2>
      <p><strong>{regime.market_regime}</strong></p>
      <p>As Of UTC: {regime.as_of_utc}</p>
      <h3>System Health</h3>
      <pre className="jsonBlock">{JSON.stringify(health, null, 2)}</pre>
    </section>
  );
}
