export interface ProbabilityInput {
  momentumScore: number;
  flowScore: number;
  volatilityRegime: string;
  regime: string;
}

export interface ProbabilityOutput {
  up_prob: number;
  down_prob: number;
  sideways_prob: number;
}

const softmax = (arr: number[]): number[] => {
  const maxVal = Math.max(...arr);
  const exps = arr.map((v) => Math.exp(v - maxVal));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((x) => (sum > 0 ? x / sum : 0));
};

export function computeProbabilities(input: ProbabilityInput): ProbabilityOutput {
  const momentum = Math.max(-1, Math.min(1, input.momentumScore));
  const flow = Math.max(-1, Math.min(1, input.flowScore));
  const vol = input.volatilityRegime.toUpperCase();
  const regime = input.regime.toLowerCase();

  const regimeUpBias = regime.includes("bull") || regime.includes("breakout_up") ? 0.35 : 0.0;
  const regimeDownBias = regime.includes("bear") || regime.includes("breakout_down") ? 0.35 : 0.0;
  const sideBias = vol === "LOW" || vol === "COMPRESSION" ? 0.55 : vol === "HIGH_VOL" ? 0.25 : vol === "EXPANSION" ? -0.1 : 0.0;

  const upLogit = 0.9 * momentum + 0.8 * flow + regimeUpBias;
  const downLogit = -0.9 * momentum - 0.8 * flow + regimeDownBias;
  const sideLogit = 0.35 + sideBias - 0.6 * Math.abs(momentum) - 0.5 * Math.abs(flow);

  const [up, down, side] = softmax([upLogit, downLogit, sideLogit]).map((p) => p * 100);
  return {
    up_prob: Number(up.toFixed(2)),
    down_prob: Number(down.toFixed(2)),
    sideways_prob: Number(side.toFixed(2)),
  };
}

