"use client";

import { createChart, CandlestickData, ColorType, IChartApi, ISeriesApi, Time } from "lightweight-charts";
import { useEffect, useRef, useState } from "react";

type Candle = {
  open_time: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
};

const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:9000";

function toSeries(rows: Candle[]): CandlestickData<Time>[] {
  return rows.map((r) => ({
    time: Math.floor(r.open_time / 1000) as Time,
    open: Number(r.open),
    high: Number(r.high),
    low: Number(r.low),
    close: Number(r.close)
  }));
}

export default function LiveCandleChart() {
  const rootRef = useRef<HTMLDivElement | null>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const seriesRef = useRef<ISeriesApi<"Candlestick"> | null>(null);
  const [timeframe, setTimeframe] = useState("15m");

  useEffect(() => {
    if (!rootRef.current) return;
    const chart = createChart(rootRef.current, {
      width: rootRef.current.clientWidth,
      height: 380,
      layout: { background: { type: ColorType.Solid, color: "#161b22" }, textColor: "#a9b4c2" },
      grid: { vertLines: { color: "#222a35" }, horzLines: { color: "#222a35" } },
      rightPriceScale: { borderColor: "#30363d" },
      timeScale: { borderColor: "#30363d" }
    });
    const series = chart.addCandlestickSeries({
      upColor: "#00c853",
      downColor: "#ff1744",
      wickUpColor: "#00c853",
      wickDownColor: "#ff1744",
      borderVisible: false
    });
    chartRef.current = chart;
    seriesRef.current = series;

    const onResize = () => {
      if (!rootRef.current || !chartRef.current) return;
      chartRef.current.applyOptions({ width: rootRef.current.clientWidth });
    };
    window.addEventListener("resize", onResize);
    return () => {
      window.removeEventListener("resize", onResize);
      chart.remove();
    };
  }, []);

  useEffect(() => {
    let timer: ReturnType<typeof setInterval> | null = null;

    const load = async () => {
      const path = timeframe === "1d" ? `/market/history?timeframe=1d&limit=2000` : `/market/klines?timeframe=${timeframe}&limit=500`;
      const res = await fetch(`${API_BASE}${path}`, { cache: "no-store" });
      if (!res.ok) return;
      const json = await res.json();
      const rows = (json.rows ?? []) as Candle[];
      if (seriesRef.current) {
        seriesRef.current.setData(toSeries(rows));
      }
    };

    void load();
    timer = setInterval(load, 15000);
    return () => {
      if (timer) clearInterval(timer);
    };
  }, [timeframe]);

  return (
    <section className="chartPanel">
      <header className="chartHead">
        <h3>BTCUSDT Live Candle Chart</h3>
        <select value={timeframe} onChange={(e) => setTimeframe(e.target.value)}>
          <option value="1m">1m</option>
          <option value="5m">5m</option>
          <option value="15m">15m</option>
          <option value="1h">1h</option>
          <option value="4h">4h</option>
          <option value="1d">1d All Time</option>
        </select>
      </header>
      <div ref={rootRef} className="chartRoot" />
    </section>
  );
}
