export function correlationDomain(values: (number | null | undefined)[]): [number, number] {
  let low = 1, high = -1;
  for (const value of values) {
    if (value == null || !Number.isFinite(value)) continue;
    low = Math.min(low, value);
    high = Math.max(high, value);
  }
  if (low > high) return [-1, 1];
  const padding = Math.max(0.025, (high - low) * 0.08);
  return [Math.max(-1, low - padding), Math.min(1, high + padding)];
}
