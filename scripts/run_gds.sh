#!/usr/bin/env bash
# RTL-to-GDSII for the VLA accelerator on ASAP7.
#
#   1. LUT block   (st_band20)    -> LEF/LIB abstract + merged GDS
#   2. INT8 block  (int8_engine)  -> LEF/LIB abstract + merged GDS
#   3. Chip top    (st_soc_top)   -> final GDS
#
# Prerequisites:
#   source scripts/env.sh
#   python3 scripts/setup_sram_views.py     # once per PDK install
#   dc_shell -f scripts/synth_lut_dc.tcl    # LUT gate netlist (or set LUT_NETLIST)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "$HERE/env.sh"

: "${ORFS_HOME:?ORFS_HOME is not set (see scripts/env.sh)}"
FLOW="$ORFS_HOME/flow"
[ -f "$FLOW/Makefile" ] || { echo "error: $FLOW/Makefile not found" >&2; exit 1; }
[ -f "$LUT_NETLIST" ] || {
    echo "error: LUT netlist not found: $LUT_NETLIST" >&2
    echo "       run scripts/synth_lut_dc.tcl or set LUT_NETLIST" >&2
    exit 1
}

python3 "$HERE/patch_orfs_grt.py"

mkdir -p "$FLOW/designs/asap7"
for d in st_band20 int8_engine st_soc_top; do
    dst="$FLOW/designs/asap7/$d"
    if [ -L "$dst" ]; then
        rm "$dst"
    elif [ -e "$dst" ]; then
        echo "error: $dst exists and is not a symlink; move it aside first" >&2
        exit 1
    fi
    ln -s "$HERE/flow/$d" "$dst"
done

cd "$FLOW"

run_block () {
    local cfg=$1 design=$2
    echo "=== $cfg: place & route ==="
    make DESIGN_CONFIG=./designs/asap7/"$cfg"/config.mk all

    # The parent gets a CTS-stage abstract: a route-stage abstract writes the
    # block's own M7+ wires into the LEF as obstructions (PDN-0006 later).
    echo "=== $cfg: abstract (CTS view) ==="
    make DESIGN_CONFIG=./designs/asap7/"$cfg"/config.mk \
         ABSTRACT_SOURCE=4_1_cts ABSTRACT_BLOAT_OBS=0 do-generate_abstract

    local r="results/asap7/$cfg/base"
    mkdir -p "$VIEWS/$design"
    cp "$r/$design.lef" "$VIEWS/$design/"
    cp "$r/${design}_typ.lib" "$VIEWS/$design/"
    cp "$r/6_1_merged.gds" "$VIEWS/$design/"
}

run_block st_band20   sparse_ternary_top
run_block int8_engine int8_engine

echo "=== st_soc_top: place & route ==="
make DESIGN_CONFIG=./designs/asap7/st_soc_top/config.mk all

echo
echo "GDS: $FLOW/results/asap7/st_soc_top/base/6_final.gds"
