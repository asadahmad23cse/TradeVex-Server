export type RegimeBucket = "bullish" | "bearish" | "sideways";

export interface AdaptiveLearningConfig {
  learningRate: number;
  minTradesRequired: number;
  updateFrequency: number;
  maxChangePerUpdate: number;
}

export interface LearningTrade {
  regime: RegimeBucket;
  direction: "LONG" | "SHORT";
  pnlPct: number;
  rawProb: number;
  factors: Record<string, number>;
  volatilityState: string;
  flowState: string;
}

export interface AdaptiveMetaDecision {
  baseDecision: "LONG" | "SHORT" | "HOLD";
  finalDecision: "LONG" | "SHORT" | "HOLD";
  adjustedConfidence: number;
  adaptiveScore: number;
  calibratedProb: number;
  reasons: string[];
}

const FACTORS = ["regime", "momentum", "flow", "cost", "volatility"] as const;

export class AdaptiveLearningEngine {
  private cfg: AdaptiveLearningConfig;
  private weights: Record<RegimeBucket, Record<string, number>>;
  private history: Record<RegimeBucket, LearningTrade[]>;

  constructor(cfg?: Partial<AdaptiveLearningConfig>) {
    this.cfg = {
      learningRate: 0.01,
      minTradesRequired: 30,
      updateFrequency: 20,
      maxChangePerUpdate: 0.05,
      ...cfg,
    };
    const base = { regime: 0.22, momentum: 0.2, flow: 0.3, cost: 0.16, volatility: 0.12 };
    this.weights = {
      bullish: { ...base },
      bearish: { ...base },
      sideways: { ...base },
    };
    this.history = { bullish: [], bearish: [], sideways: [] };
  }

  private normalizeWeights(w: Record<string, number>): Record<string, number> {
    const out: Record<string, number> = {};
    let sum = 0;
    for (const k of FACTORS) {
      const v = Math.max(0.05, Math.min(0.6, Number(w[k] || 0.1)));
      out[k] = v;
      sum += v;
    }
    if (sum <= 0) return out;
    for (const k of FACTORS) out[k] /= sum;
    return out;
  }

  private score(regime: RegimeBucket, factors: Record<string, number>): number {
    const w = this.weights[regime];
    let s = 0;
    for (const k of FACTORS) s += (w[k] || 0) * (Number(factors[k] || 0));
    return Math.max(-1, Math.min(1, s));
  }

  private calibrate(regime: RegimeBucket, raw: number): number {
    const rows = this.history[regime];
    if (rows.length < 15) return raw;
    const p = Math.max(0, Math.min(1, raw));
    const bin = Math.min(9, Math.max(0, Math.floor(p * 10)));
    const inBin = rows.filter((r) => Math.min(9, Math.max(0, Math.floor(Math.max(0, Math.min(1, r.rawProb)) * 10))) === bin);
    if (!inBin.length) return p;
    const acc = inBin.filter((r) => r.pnlPct > 0).length / inBin.length;
    return Math.max(0, Math.min(1, 0.6 * p + 0.4 * acc));
  }

  recordTrade(row: LearningTrade): void {
    const regime = row.regime;
    this.history[regime].push(row);
    if (this.history[regime].length > 600) this.history[regime] = this.history[regime].slice(-600);
    this.maybeUpdate(regime);
  }

  private maybeUpdate(regime: RegimeBucket): void {
    const rows = this.history[regime];
    if (rows.length < this.cfg.minTradesRequired) return;
    if (rows.length % this.cfg.updateFrequency !== 0) return;

    const train = rows.slice(-80, -20);
    const test = rows.slice(-20);
    if (train.length < 20 || test.length < 10) return;

    const current = { ...this.weights[regime] };
    const proposal = { ...current };
    for (const f of FACTORS) {
      const signal =
        train.reduce((acc, r) => {
          const y = r.pnlPct > 0 ? 1 : -1;
          return acc + y * (Number(r.factors[f] || 0));
        }, 0) / train.length;
      const deltaRaw = this.cfg.learningRate * signal;
      const maxDelta = current[f] * this.cfg.maxChangePerUpdate;
      const delta = Math.max(-maxDelta, Math.min(maxDelta, deltaRaw));
      proposal[f] = current[f] + delta;
    }
    const normalized = this.normalizeWeights(proposal);

    const evalMetric = (w: Record<string, number>): number => {
      let wins = 0;
      let brier = 0;
      for (const r of test) {
        const s =
          FACTORS.reduce((acc, f) => acc + (w[f] || 0) * (Number(r.factors[f] || 0)), 0) *
          (r.direction === "LONG" ? 1 : -1);
        const p = 1 / (1 + Math.exp(-Math.max(-30, Math.min(30, s * 3))));
        const y = r.pnlPct > 0 ? 1 : 0;
        wins += y;
        brier += (p - y) * (p - y);
      }
      const winRate = wins / test.length;
      const brierScore = brier / test.length;
      return 0.7 * winRate - 0.3 * brierScore;
    };

    const baseline = evalMetric(current);
    const candidate = evalMetric(normalized);
    if (candidate > baseline + 0.01) {
      this.weights[regime] = normalized;
    }
  }

  metaDecision(params: {
    regime: RegimeBucket;
    baseDecision: "LONG" | "SHORT" | "HOLD";
    baseConfidence: number;
    factors: Record<string, number>;
    rawProb: number;
    blockers?: string[];
  }): AdaptiveMetaDecision {
    const base = params.baseDecision;
    const blockers = params.blockers || [];
    const raw = Math.max(0, Math.min(1, params.rawProb));
    const calibrated = this.calibrate(params.regime, raw);
    const sign = base === "LONG" ? 1 : base === "SHORT" ? -1 : 0;
    const score = this.score(params.regime, params.factors) * sign;

    let finalDecision: "LONG" | "SHORT" | "HOLD" = base;
    const reasons: string[] = [];
    if (blockers.length) {
      finalDecision = "HOLD";
      reasons.push(...blockers.slice(0, 2));
    }
    if (base !== "HOLD" && calibrated < 0.53) {
      finalDecision = "HOLD";
      reasons.push(`Calibrated probability too low (${(calibrated * 100).toFixed(1)}%)`);
    }
    if (base !== "HOLD" && score < 0.05) {
      finalDecision = "HOLD";
      reasons.push(`Adaptive score too weak (${score.toFixed(3)})`);
    }

    const adjustedConfidence = Math.max(
      0,
      Math.min(100, params.baseConfidence * (0.85 + 0.3 * calibrated) * (finalDecision === "HOLD" ? 0.7 : 1.0)),
    );

    return {
      baseDecision: base,
      finalDecision,
      adjustedConfidence: Number(adjustedConfidence.toFixed(2)),
      adaptiveScore: Number(score.toFixed(6)),
      calibratedProb: Number(calibrated.toFixed(6)),
      reasons,
    };
  }
}

