export interface AggTrade {
  price: number;
  qty: number;
  maker: boolean;
}

export interface OrderBookDepth {
  bids: Array<[number, number]>;
  asks: Array<[number, number]>;
}

export interface OrderFlowOutput {
  cvd_trend: "RISING" | "FALLING" | "FLAT";
  cvd_value: number;
  cvd_slope: number;
  obi_value: number;
  aggression: number;
  absorption_levels: number[];
  flow_score: number;
}

const clamp = (v: number, min = -1, max = 1): number => Math.max(min, Math.min(max, v));

export function computeOrderFlow(trades: AggTrade[], depth: OrderBookDepth, levels = 15): OrderFlowOutput {
  const window = trades.slice(-1200);
  let buy = 0;
  let sell = 0;
  const signed: number[] = [];

  for (const t of window) {
    const qty = Number(t.qty || 0);
    if (qty <= 0) continue;
    if (t.maker) {
      sell += qty;
      signed.push(-qty);
    } else {
      buy += qty;
      signed.push(qty);
    }
  }

  const total = buy + sell;
  const aggression = total > 0 ? (buy / total) * 100 : 50;
  const cvdSeries: number[] = [];
  let cvd = 0;
  for (const v of signed) {
    cvd += v;
    cvdSeries.push(cvd);
  }
  const tail = cvdSeries.slice(-120);
  const cvdSlope =
    tail.length > 1
      ? (tail[tail.length - 1] - tail[0]) / Math.max(tail.length - 1, 1)
      : 0;
  const cvdTrend: OrderFlowOutput["cvd_trend"] =
    cvdSlope > 0.001 ? "RISING" : cvdSlope < -0.001 ? "FALLING" : "FLAT";

  const bids = (depth?.bids || []).slice(0, levels);
  const asks = (depth?.asks || []).slice(0, levels);
  let bidWeighted = 0;
  let askWeighted = 0;
  bids.forEach(([px, qty], i) => {
    const w = 1 / (1 + i * 0.25);
    bidWeighted += Number(px || 0) * Number(qty || 0) * w;
  });
  asks.forEach(([px, qty], i) => {
    const w = 1 / (1 + i * 0.25);
    askWeighted += Number(px || 0) * Number(qty || 0) * w;
  });
  const totalDepth = bidWeighted + askWeighted;
  const obi = totalDepth > 0 ? (bidWeighted - askWeighted) / totalDepth : 0;

  const allLevels = [...bids, ...asks].map(([px, qty]) => ({ px: Number(px), qty: Number(qty) }));
  const qtys = allLevels.map((x) => x.qty).filter((x) => Number.isFinite(x) && x > 0);
  qtys.sort((a, b) => a - b);
  const qIdx = Math.max(0, Math.floor(qtys.length * 0.85) - 1);
  const threshold = qtys.length ? qtys[qIdx] : Number.POSITIVE_INFINITY;
  const absorption = allLevels
    .filter((x) => x.qty >= threshold)
    .map((x) => x.px)
    .filter((v, i, arr) => Number.isFinite(v) && arr.indexOf(v) === i)
    .slice(0, 6);

  const flowScore = clamp(0.55 * obi + 0.35 * Math.tanh(cvdSlope) + 0.1 * ((aggression - 50) / 50));

  return {
    cvd_trend: cvdTrend,
    cvd_value: Number(cvd.toFixed(6)),
    cvd_slope: Number(cvdSlope.toFixed(8)),
    obi_value: Number(obi.toFixed(6)),
    aggression: Number(aggression.toFixed(2)),
    absorption_levels: absorption.map((x) => Number(x.toFixed(2))),
    flow_score: Number(flowScore.toFixed(6)),
  };
}

