# shellcheck shell=bash
# Source this file before running any flow script:
#   source scripts/env.sh

VLA_ACCEL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export VLA_ACCEL_ROOT

# OpenROAD-flow-scripts checkout (must contain flow/Makefile).
export ORFS_HOME="${ORFS_HOME:-$VLA_ACCEL_ROOT/third_party/OpenROAD-flow-scripts}"

# ASAP7 predictive PDK SRAM generator output.
#   $SRAM_KIT/LEF/*.lef, $SRAM_KIT/LIB/*.lib, $SRAM_KIT/verilog/*.v
# The kit GDS files live next to the "generated" directory.
export SRAM_KIT="${SRAM_KIT:-$VLA_ACCEL_ROOT/third_party/asap7_sram/generated}"
export SRAM_KIT_GDS="${SRAM_KIT_GDS:-$(dirname "$SRAM_KIT")/gds}"

# Generated views and outputs.
export SRAM_VIEWS="${SRAM_VIEWS:-$VLA_ACCEL_ROOT/build/sram_views}"
export SRAM_GDS="${SRAM_GDS:-$SRAM_VIEWS/gds}"
export VIEWS="${VIEWS:-$VLA_ACCEL_ROOT/build/views}"
export LUT_NETLIST="${LUT_NETLIST:-$VLA_ACCEL_ROOT/build/lut/sparse_ternary_top_mapped.v}"

mkdir -p "$SRAM_VIEWS" "$SRAM_GDS" "$VIEWS" "$(dirname "$LUT_NETLIST")"

if ! command -v openroad >/dev/null 2>&1; then
    echo "warning: openroad is not on PATH (set PATH to your ORFS tools/install)" >&2
fi
if ! command -v make >/dev/null 2>&1; then
    echo "warning: make is not on PATH" >&2
fi
