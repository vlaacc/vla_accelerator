# int8_engine (PE block) -- INT8 side: 4x 32x32 mesh + its own act/acc SRAM
# banks and the vector epilogue.  Hardened as a block so the chip top does not
# have to flatten the array or route its 9,200 + 8,192-bit buses.
#
# Paths come from scripts/env.sh (VLA_ACCEL_ROOT / SRAM_* / VIEWS).
export DESIGN_NICKNAME = int8_engine
export DESIGN_NAME     = int8_engine
export PLATFORM        = asap7

export SRC_DIR       = $(VLA_ACCEL_ROOT)/rtl
export VERILOG_FILES = $(sort $(wildcard $(SRC_DIR)/*.v)) \
                       $(sort $(wildcard $(SRC_DIR)/sram_7nm/*.v))
export VERILOG_INCLUDE_DIRS = $(SRC_DIR)
export SDC_FILE      = $(dir $(DESIGN_CONFIG))constraint.sdc
export SYNTH_HDL_FRONTEND = slang
export SYNTH_SLANG_ARGS = --unroll-limit=10000

export ADDITIONAL_LEFS = $(SRAM_VIEWS)/srambank_64x4x64_6t122.lef \
                         $(SRAM_VIEWS)/srambank_256x4x64_6t122.lef
export ADDITIONAL_LIBS = $(SRAM_KIT)/LIB/srambank_64x4x64_6t122.lib \
                         $(SRAM_KIT)/LIB/srambank_256x4x64_6t122.lib

export DIE_AREA   = 0 0 1200 1100
export CORE_AREA  = 5 5 1195 1095
export PLACE_DENSITY = 0.45
export CLOCK_PERIOD  = 2.0
export NUM_CORES     = 16

# Keep hierarchy: ABC cannot handle the flattened block (4096 PE + 100 fp32
# lanes).  Hierarchical mode synthesises per module and keeps the PE boundary
# visible in the final layout instead of flattening it away.
export SYNTH_HIERARCHICAL = 1

# Real ASAP7 SRAM geometry.  The kit ships GDS named by bit-width (srambank_32b/
# 64b/128b); scripts/setup_sram_views.py renames the top cell to the LEF macro
# name so KLayout merges them correctly.  Mapping:
#   srambank_32b.gds  ( 9.612 x 77.804) -> srambank_64x4x64_6t122   ( 9.612 x 77.76)
#   srambank_128b.gds (30.348 x 77.804) -> srambank_256x4x64_6t122 (30.348 x 77.76)
export ADDITIONAL_GDS = $(SRAM_GDS)/srambank_64x4x64_6t122.gds \
                        $(SRAM_GDS)/srambank_256x4x64_6t122.gds

# Timing-driven GPL runs an internal resizer that hangs on these designs
# (Remaining stuck, 0 log output) -- same failure seen on 45nm.  Disable it.
export GPL_TIMING_DRIVEN = 0

# This block cannot meet the aggressive target, so asking repair_timing to fix
# 100% of TNS makes it spin forever (cost frozen, RSZ-0075 repeated).  Bound the
# repair effort instead.
export TNS_END_PERCENT = 0

# RTLMP did not converge on this design; place the 80 SRAM macros explicitly.
export MACRO_PLACEMENT_TCL = $(dir $(DESIGN_CONFIG))macro_placement.tcl

# Routability-driven GPL re-runs a full global route each iteration; on this
# 3.5M-instance block congestion never fell below 1.27, so the iterations bought
# nothing and cost ~30 min each.  Leave congestion to the router.
export GPL_ROUTABILITY_DRIVEN = 1

# GPL runs with -pad_left/right 0 but DPL defaults to 1 site of padding per
# side, which inflates the effective cell area ~50% and made detailed placement
# fail on 753 resizer-inserted buffers.  Match GPL.
export CELL_PAD_IN_SITES_DETAIL_PLACEMENT = 0

# Resizer-inserted buffers land on the FIRM macros and DPL cannot legalise them
# (DPL-0036), so repair_design is skipped.  Without it the design has huge
# slew/cap violations and the CTS-time repair loop spins on 460k endpoints
# (RSZ-2021 repeated).  Physical bring-up, not timing signoff.
export SKIP_CTS_REPAIR_TIMING = 1
# The reset/clear fan-out is split in RTL (BUFx24 in systolic_array_int8,
# vector_unit_4096, int8_act_skew, int8_mesh_top), so the 16-hour GRT-stage
# repair_design scan is no longer needed; detailed route copes with the <=8k
# nets.
export SKIP_INCREMENTAL_REPAIR = 1

# Buffers from repair_design land on the FIRM macros and DPL cannot legalise
# them.
export SKIP_REPAIR_DESIGN = 1

# GRT was stuck in the congestion-iteration loop (8/30, 12 h on one iteration)
# with only ~5 hotspots left on this 1.2M-cell block.  One bounded iteration is
# enough; detailed route cleans up the rest.
export GLOBAL_ROUTE_ARGS = -congestion_iterations 1 -allow_congestion -verbose
export ALLOW_GRT_CONGESTION = 1
export SKIP_ANTENNA_REPAIR = 1
export SKIP_ANTENNA_REPAIR_POST_DRT = 1
