# sparse_ternary_top (LUT engine) on ASAP7 (7nm predictive).
#
# The gate netlist is produced by Synopsys Design Compiler against the ASAP7
# 7.5T RVT/TT libraries.  Yosys cannot synthesise this design: the generic
# techmap of the arithmetic peaks above 242 GB and is OOM-killed regardless of
# SYNTH_MEMORY_MAX_BITS / ADDER_MAP_FILE / SYNTH_OPERATIONS_ARGS.  DC finishes
# in ~7 h:
#   973,444 cells (814,513 comb + 158,621 seq), 290 SRAM macros,
#   total cell area 324,809 um^2.
# Only the top-level port list matters here; the netlist instantiates the same
# srambank_* macros as the RTL and uses the same ASAP7 standard cells.
#
# Paths come from scripts/env.sh (VLA_ACCEL_ROOT / SRAM_* / VIEWS / LUT_NETLIST).
export DESIGN_NICKNAME = st_band20
export DESIGN_NAME     = sparse_ternary_top
export PLATFORM        = asap7

export VERILOG_FILES = $(LUT_NETLIST)
export SDC_FILE      = $(dir $(DESIGN_CONFIG))constraint.sdc

# ── Real ASAP7 SRAM leaf macros (srambank_<words>x4x<bits>_6t122) ──
export ADDITIONAL_LEFS = $(SRAM_VIEWS)/srambank_64x4x64_6t122.lef \
                         $(SRAM_VIEWS)/srambank_256x4x64_6t122.lef
export ADDITIONAL_LIBS = $(SRAM_KIT)/LIB/srambank_64x4x64_6t122.lib \
                         $(SRAM_KIT)/LIB/srambank_256x4x64_6t122.lib

# Real ASAP7 SRAM geometry (GDS renamed to the LEF macro name).
export ADDITIONAL_GDS = $(SRAM_GDS)/srambank_64x4x64_6t122.gds \
                        $(SRAM_GDS)/srambank_256x4x64_6t122.gds

export DIE_AREA   = 0 0 760 1140
export CORE_AREA  = 5 5 755 1135
export PLACE_DENSITY = 0.25
export CLOCK_PERIOD  = 1.0
export NUM_CORES     = 16

# Timing-driven GPL spins on these designs.
export GPL_TIMING_DRIVEN = 0
# Routability-driven GPL re-runs a full global route per iteration.
export GPL_ROUTABILITY_DRIVEN = 1
# This block cannot meet the aggressive target; bound the repair effort.
export TNS_END_PERCENT = 0

# Place the SRAM macros explicitly (place_macro cannot satisfy MPL-0005 on
# ASAP7 SRAM pins; the script shelf-packs straight through odb).
export MACRO_PLACEMENT_TCL = $(dir $(DESIGN_CONFIG))macro_placement.tcl

# Physical bring-up compromises: the resizer's buffers land on the FIRM macros
# and DPL cannot legalise them, and without the resizer the CTS-time timing
# repair spins.  See designs/asap7/int8_engine/config.mk.
export CELL_PAD_IN_SITES_DETAIL_PLACEMENT = 0
export SKIP_CTS_REPAIR_TIMING = 1
export SKIP_INCREMENTAL_REPAIR = 1

# repair_design is ENABLED: the weight SRAM dataout bits fan out to 64 loads
# each (u_wt_cache.sram_dout[*] iterms=65) and with no buffer tree the router
# has to fan out 65 terminals straight from the macro pin -- the V3 via cluster
# at the pin becomes the entire residual DRC.  Safe to enable here because all
# SRAM macros sit in one bottom band, so repair_design buffers cannot land on a
# FIRM macro (that was the original DPL-0036 reason for skipping it).

# ASAP7 tech LEF defines M1..M9; the platform default caps routing at M7 and
# leaves two layers idle.  designs/asap7/aes-block/config.mk does the same.
export MAX_ROUTING_LAYER = M9

# GRT capacity adjustment.  ORFS variables.yaml: "global routing will succeed
# with lower values and fail with higher values ... start with 0.10".
# The aes-block design in this PDK uses 0.3; the default is 0.5.
# Overridden by the local FASTROUTE_TCL (that variable is ignored when
# FASTROUTE_TCL is set).
# export ROUTING_LAYER_ADJUSTMENT = 0.10

# Take over the platform fastroute.tcl so the 0.10 adjustment actually applies.
export FASTROUTE_TCL = $(dir $(DESIGN_CONFIG))fastroute.tcl

export IO_CONSTRAINTS = $(dir $(DESIGN_CONFIG))io.tcl
