"""Generate C++ study constants from the shared JSON settings."""

import json
from pathlib import Path
import sys
from tempfile import NamedTemporaryFile


def main(config_path, output_path):
    config = json.loads(Path(config_path).read_text())
    memories = config["memories"]
    if len(memories) != 5:
        raise ValueError("Chapter 6 reports expect five fixed-memory arms")
    drift_from, drift_to = config["drift_knobs"]
    jump_from, jump_to = config["jump_knobs"]
    vff_slow, vff_fast = config["vff_rates"]
    memory_values = ", ".join(map(str, memories))
    memory_csv = ",".join(map(str, memories))
    header = f"""// Generated from study_config.json.
#pragma once

#include <cstddef>
#include <cstdint>

namespace thesis {{

inline constexpr std::size_t kN = {config["observations"]};
inline constexpr std::size_t kBurn = {config["burn_in"]};
inline constexpr std::size_t kChange = {config["change_point"]};
inline constexpr unsigned kFirstSeed = {config["first_seed"]};
inline constexpr int kSeedCount = {config["seed_count"]};
inline constexpr double kConstantKnob = {config["constant_knob"]};
inline constexpr double kDriftFrom = {drift_from};
inline constexpr double kDriftTo = {drift_to};
inline constexpr double kJumpFrom = {jump_from};
inline constexpr double kJumpTo = {jump_to};
inline constexpr double kMem[] = {{{memory_values}}};
inline constexpr char kMemoryCsv[] = "{memory_csv}";
inline constexpr std::size_t kK = sizeof(kMem) / sizeof(kMem[0]);
inline constexpr std::uint16_t kKll = {config["kll_capacity"]};
inline constexpr double kClip = {config["pit_clip"]};
inline constexpr double kVffSlow = {vff_slow};
inline constexpr double kVffFast = {vff_fast};

}}  // namespace thesis
"""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode="w", dir=output.parent, delete=False) as temp:
        temp.write(header)
    Path(temp.name).replace(output)


if __name__ == "__main__":
    main(*sys.argv[1:])
