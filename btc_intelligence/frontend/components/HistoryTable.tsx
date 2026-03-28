"use client";

import { Fragment, useEffect, useState } from "react";

import { SignalPayload } from "./types";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:9000";

export default function HistoryTable() {
  const [rows, setRows] = useState<SignalPayload[]>([]);

  useEffect(() => {
    let timer: ReturnType<typeof setInterval> | null = null;
    const load = async () => {
      const res = await fetch(`${API_BASE}/signal/history`, { cache: "no-store" });
      if (!res.ok) return;
      const json = (await res.json()) as SignalPayload[];
      setRows(json);
    };
    void load();
    timer = setInterval(load, 30000);
    return () => {
      if (timer) clearInterval(timer);
    };
  }, []);

  return (
    <section className="card">
      <h2>Signal History</h2>
      <div className="historyTable">
        <div className="h">UTC</div>
        <div className="h">Signal</div>
        <div className="h">Confidence</div>
        <div className="h">Algo</div>
        <div className="h">Reason</div>
        {rows.slice(0, 100).map((r, idx) => (
          <Fragment key={`${r.as_of_utc}-${idx}`}>
            <div>{r.as_of_utc.replace("T", " ").replace("Z", "")}</div>
            <div className={r.signal === "LONG" ? "cLong" : r.signal === "SHORT" ? "cShort" : "cHold"}>{r.signal}</div>
            <div>{r.confidence}%</div>
            <div>{r.algo}</div>
            <div>{r.reason}</div>
          </Fragment>
        ))}
      </div>
    </section>
  );
}
