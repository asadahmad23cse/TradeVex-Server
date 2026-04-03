export type Decision = "LONG" | "SHORT" | "HOLD";

export interface DecisionEngineInput {
  regime: string;
  cvdSlope: number;
  obiImbalance: number;
  volatilityRegime: string;
  costScore: number;
  momentumScore: number;
  flowScore: number;
}

export interface WeightedFactorContribution {
  name: string;
  value: number;
  weight: number;
  contribution: number;
}

export interface DecisionBreakdown {
  regime_score: number;
  momentum_score: number;
  flow_score: number;
  cost_score: number;
  final_score: number;
  explanation: string;
}

export interface TradeVerdict {
  regime_alignment: "HIGH" | "LOW";
  liquidity_quality: "STRONG" | "WEAK";
  volatility_state: "TRADEABLE" | "NOT_TRADEABLE";
  final_verdict: "TRADE" | "AVOID";
}

export interface DecisionEngineOutput {
  decision: Decision;
  confidence: number;
  reason: string;
  blockers: string[];
  trade_triggers: string[];
  decision_breakdown: DecisionBreakdown;
  factors: WeightedFactorContribution[];
  trade_verdict: TradeVerdict;
}

const clamp = (v: number, min = -1, max = 1): number => Math.max(min, Math.min(max, v));

const regimeDirection = (regime: string): number => {
  const r = regime.toLowerCase();
  if (r.includes("bear") || r.includes("breakout_down") || r.includes("panic")) return -1;
  if (r.includes("bull") || r.includes("breakout_up")) return 1;
  return 0;
};

const volatilityScore = (vol: string): number => {
  const v = vol.toUpperCase();
  if (v === "NORMAL") return 0.45;
  if (v === "EXPANSION") return 0.10;
  if (v === "LOW" || v === "COMPRESSION") return -0.35;
  if (v === "HIGH_VOL" || v === "PANIC") return -0.45;
  return 0.0;
};

export function evaluateDecision(input: DecisionEngineInput): DecisionEngineOutput {
  const momentum = clamp(input.momentumScore);
  const flow = clamp(input.flowScore);
  const cost = clamp(input.costScore);
  const obi = clamp(input.obiImbalance);
  const regimeDir = regimeDirection(input.regime);

  const directionalBias = clamp(0.65 * flow + 0.35 * momentum);
  const regimeScore = regimeDir !== 0 ? clamp(directionalBias * regimeDir) : 0.0;
  const flowScore = clamp(0.75 * flow + 0.25 * obi);
  const volScore = volatilityScore(input.volatilityRegime);

  const weights = {
    regime: 0.22,
    momentum: 0.20,
    flow: 0.30,
    cost: 0.16,
    volatility: 0.12,
  } as const;

  const finalScore = clamp(
    weights.regime * regimeScore +
      weights.momentum * momentum +
      weights.flow * flowScore +
      weights.cost * cost +
      weights.volatility * volScore,
  );

  const blockers: string[] = [];
  const vol = input.volatilityRegime.toUpperCase();
  if (vol === "LOW" || vol === "COMPRESSION") blockers.push("Volatility LOW: no trade until expansion");
  if (vol === "HIGH_VOL" || vol === "PANIC") blockers.push("Extreme volatility: execution risk too high");
  if (cost < -0.05) blockers.push("Cost gate negative: expected edge after costs <= 0");
  if (regimeDir !== 0 && directionalBias !== 0 && Math.sign(regimeDir) !== Math.sign(directionalBias)) {
    blockers.push("Regime conflict with flow/momentum direction");
  }

  let decision: Decision = "HOLD";
  if (!blockers.length) {
    if (finalScore >= 0.12) decision = "LONG";
    else if (finalScore <= -0.12) decision = "SHORT";
  }

  const confidence = clamp(Math.abs(finalScore) + 0.18 - blockers.length * 0.12, 0, 1) * 100;
  const reason =
    blockers.length > 0
      ? `Blocked (${blockers.slice(0, 2).join("; ")})`
      : decision === "LONG"
      ? "Long bias confirmed by regime+flow+momentum"
      : decision === "SHORT"
      ? "Short bias confirmed by regime+flow+momentum"
      : `No directional edge: final score ${(finalScore * 100).toFixed(1)}`;

  const tradeTriggers: string[] = [];
  const targetLong = decision === "LONG" || (decision === "HOLD" && directionalBias >= 0);
  const targetShort = decision === "SHORT" || (decision === "HOLD" && directionalBias < 0);
  if (targetLong) {
    if (input.cvdSlope <= 0) tradeTriggers.push("CVD must flip positive");
    if (input.obiImbalance <= 0.3) tradeTriggers.push("OBI > 0.30");
  }
  if (targetShort) {
    if (input.cvdSlope >= 0) tradeTriggers.push("CVD must flip negative");
    if (input.obiImbalance >= -0.3) tradeTriggers.push("OBI < -0.30");
  }
  if (vol === "LOW" || vol === "COMPRESSION") tradeTriggers.push("Volatility must expand");
  if (vol === "HIGH_VOL" || vol === "PANIC") tradeTriggers.push("Volatility must normalize");
  if (cost <= 0) tradeTriggers.push("Net alpha after costs must turn positive");
  if (!tradeTriggers.length && decision !== "HOLD") tradeTriggers.push("Execution window open");

  const factors: WeightedFactorContribution[] = [
    { name: "regime", value: regimeScore, weight: weights.regime, contribution: regimeScore * weights.regime },
    { name: "momentum", value: momentum, weight: weights.momentum, contribution: momentum * weights.momentum },
    { name: "flow", value: flowScore, weight: weights.flow, contribution: flowScore * weights.flow },
    { name: "cost", value: cost, weight: weights.cost, contribution: cost * weights.cost },
    { name: "volatility", value: volScore, weight: weights.volatility, contribution: volScore * weights.volatility },
  ];

  const tradeVerdict: TradeVerdict = {
    regime_alignment: regimeDir === 0 || regimeScore >= 0 ? "HIGH" : "LOW",
    liquidity_quality: Math.abs(obi) >= 0.18 ? "STRONG" : "WEAK",
    volatility_state: vol === "NORMAL" || vol === "EXPANSION" ? "TRADEABLE" : "NOT_TRADEABLE",
    final_verdict: decision === "HOLD" || blockers.length > 0 ? "AVOID" : "TRADE",
  };

  return {
    decision,
    confidence: Number(confidence.toFixed(2)),
    reason,
    blockers,
    trade_triggers: tradeTriggers.slice(0, 6),
    decision_breakdown: {
      regime_score: Number((regimeScore * 100).toFixed(2)),
      momentum_score: Number((momentum * 100).toFixed(2)),
      flow_score: Number((flowScore * 100).toFixed(2)),
      cost_score: Number((cost * 100).toFixed(2)),
      final_score: Number((finalScore * 100).toFixed(2)),
      explanation: `${finalScore >= 0 ? "Bullish" : "Bearish"} tilt with weighted regime/flow confirmation.`,
    },
    factors,
    trade_verdict: tradeVerdict,
  };
}

