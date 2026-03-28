"use client";

import { useEffect, useState } from "react";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:9000";

export default function AnalysisPanel() {
  const [data, setData] = useState<Record<string, number>>({});

  useEffect(() => {
    let timer: ReturnType<typeof setInterval> | null = null;
    const load = async () => {
      const res = await fetch(`${API_BASE}/features`, { cache: "no-store" });
      if (!res.ok) return;
      const json = await res.json();
      setData(json.feature_map ?? {});
    };
    void load();
    timer = setInterval(load, 30000);
    return () => {
      if (timer) clearInterval(timer);
    };
  }, []);

  const entries = Object.entries(data).slice(0, 20);

  return (
    <section className="card">
      <h2>Feature Breakdown</h2>
      <div className="tableLike">
        {entries.map(([k, v]) => (
          <div key={k} className="row">
            <span>{k}</span>
            <strong>{Number(v).toFixed(4)}</strong>
          </div>
        ))}
      </div>
    </section>
  );
}
