export type SlippageRisk = "LOW" | "MEDIUM" | "HIGH";

export interface ExecutionInput {
  currentPrice: number;
  atrPct: number;
  volatilityRegime: string;
  liquidityScore: number; // 0..1
  direction: "LONG" | "SHORT";
}

export interface ExecutionPlan {
  entry_zone: [number, number];
  stop_loss: number;
  take_profit: number;
  slippage_risk: SlippageRisk;
}

export function buildExecutionPlan(input: ExecutionInput): ExecutionPlan {
  const px = Math.max(0, Number(input.currentPrice || 0));
  if (px <= 0) {
    return {
      entry_zone: [0, 0],
      stop_loss: 0,
      take_profit: 0,
      slippage_risk: "HIGH",
    };
  }

  const atrPct = Math.max(0.05, Number(input.atrPct || 0.05));
  const vol = input.volatilityRegime.toUpperCase();
  const liq = Math.max(0, Math.min(1, Number(input.liquidityScore || 0)));
  const atrValue = px * (atrPct / 100);

  const baseBuffer =
    vol === "LOW"
      ? 0.03
      : vol === "NORMAL"
      ? 0.08
      : vol === "EXPANSION"
      ? 0.15
      : 0.22;
  const halfZone = px * ((baseBuffer + (1 - liq) * 0.08) / 100);
  const entryMin = px - halfZone;
  const entryMax = px + halfZone;
  const entryRef = (entryMin + entryMax) / 2;

  const stopMult =
    vol === "LOW"
      ? 1.1
      : vol === "NORMAL"
      ? 1.5
      : vol === "EXPANSION"
      ? 1.9
      : 2.2;
  const tpMult = stopMult * 1.2;

  let stopLoss = entryRef;
  let takeProfit = entryRef;
  if (input.direction === "LONG") {
    stopLoss = entryRef - stopMult * atrValue;
    takeProfit = entryRef + tpMult * atrValue;
  } else {
    stopLoss = entryRef + stopMult * atrValue;
    takeProfit = entryRef - tpMult * atrValue;
  }

  const slippageRisk: SlippageRisk =
    liq < 0.35 || vol === "HIGH_VOL"
      ? "HIGH"
      : liq < 0.6 || vol === "EXPANSION"
      ? "MEDIUM"
      : "LOW";

  return {
    entry_zone: [Number(entryMin.toFixed(2)), Number(entryMax.toFixed(2))],
    stop_loss: Number(stopLoss.toFixed(2)),
    take_profit: Number(takeProfit.toFixed(2)),
    slippage_risk: slippageRisk,
  };
}

