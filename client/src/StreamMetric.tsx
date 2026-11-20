import type { ReactNode } from "react";

export default function StreamMetric({ label, value, colour }: {
  label: string;
  value: ReactNode;
  colour?: string;
}) {
  return (
    <div>
      <dt>{label}</dt>
      <dd style={{ color: colour }}>{value}</dd>
    </div>
  );
}
