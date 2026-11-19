export function correlationDomain(values: (number | null | undefined)[]): [number, number] {
  let low = 1, high = -1;
  for (const value of values) {
    if (value == null || !Number.isFinite(value)) continue;
    low = Math.min(low, value);
    high = Math.max(high, value);
  }
  if (low > high) return [-1, 1];
  const padding = Math.max(0.075, (high - low) * 0.08);
  return [Math.max(-1, Math.floor((low - padding) * 10) / 10), Math.min(1, Math.ceil((high + padding) * 10) / 10)];
}

export type StreamPoint = {
  step: number;
  x?: number;
  y?: number;
  truth?: number;
  mff: number | null;
  vff?: number | null;
  fixedShort?: number | null;
  fixedLong?: number | null;
  weights?: number[];
  familyWeights?: number[];
  vffMemory?: number | null;
};

export type Timeline = {
  active?: boolean;
  formatStep?: (step: number) => string;
  marks?: { step: number; label: string }[];
};

export const COLOURS = {
  x: "#60b5be", y: "#a0a5d6", mff: "#ff83a0", truth: "#e8eef5", vff: "#b48cff",
  fixedShort: "#79e1cb", fixedLong: "#f7c378",
};
export const formatNumber = (value?: number | null, digits = 2) => value == null ? "—" : value.toFixed(digits);
