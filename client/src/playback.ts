export type Guided = { jumpAt: number; jumpTo: number; driftAt: number; driftTo: number; driftSteps: number };

export type Playback = {
  rates: number[];
  default: number;
  preroll: number;
  warmupSeconds: number;
  eta: number;
  delta: number;
  guided: Guided;
  lab: { rho: number; memories: number[]; families: string[]; guided: Guided };
};
