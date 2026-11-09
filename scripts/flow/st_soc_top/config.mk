# st_soc_top -- DSA chip top: CSR + LUT engine + INT8 engine.
#
# The two engines are hardened blocks and arrive as macro views (LEF + LIB);
# their RTL is deliberately NOT read here.  The vector unit is not a top-level
# macro -- it is already abstracted inside int8_engine.
#
# Paths come from scripts/env.sh (VLA_ACCEL_ROOT / SRAM_* / VIEWS).
export DESIGN_NICKNAME = st_soc_top
export DESIGN_NAME     = st_soc_top
export PLATFORM        = asap7

export SRC_DIR       = $(VLA_ACCEL_ROOT)/rtl
export VERILOG_FILES = $(SRC_DIR)/st_soc_top.v \
                       $(SRC_DIR)/config_regs.v \
                       $(SRC_DIR)/sram_7nm/sram_act_bank_int8.v \
                       $(SRC_DIR)/sram_7nm/sram_acc_bank_int8.v
export VERILOG_INCLUDE_DIRS = $(SRC_DIR)
export SDC_FILE      = $(dir $(DESIGN_CONFIG))constraint.sdc
export SYNTH_HDL_FRONTEND = slang

export SYNTH_BLACKBOXES = sparse_ternary_top int8_engine

export ADDITIONAL_LEFS = $(VIEWS)/sparse_ternary_top/sparse_ternary_top.lef \
                         $(VIEWS)/int8_engine/int8_engine.lef \
                         $(SRAM_VIEWS)/srambank_64x4x64_6t122.lef \
                         $(SRAM_VIEWS)/srambank_256x4x64_6t122.lef
export ADDITIONAL_LIBS = $(VIEWS)/sparse_ternary_top/sparse_ternary_top_typ.lib \
                         $(VIEWS)/int8_engine/int8_engine_typ.lib \
                         $(SRAM_KIT)/LIB/srambank_64x4x64_6t122.lib \
                         $(SRAM_KIT)/LIB/srambank_256x4x64_6t122.lib

export DIE_AREA  = 0 0 3000 1500
export CORE_AREA = 5 5 2995 1495
export PLACE_DENSITY = 0.30
export CLOCK_PERIOD  = 1.0
export NUM_CORES     = 16

# Timing-driven GPL runs an internal resizer that hangs on these designs.
export GPL_TIMING_DRIVEN = 0
export GPL_ROUTABILITY_DRIVEN = 1
export TNS_END_PERCENT = 0

# Shelf-pack the two block macros through odb (place_macro cannot satisfy
# MPL-0005 on ASAP7-abstracted pins).
export MACRO_PLACEMENT_TCL = $(dir $(DESIGN_CONFIG))macro_placement.tcl

# Physical bring-up compromises: resizer buffers land on the FIRM macros and
# DPL cannot legalise them; without the resizer the CTS-time timing repair
# spins.  See designs/asap7/int8_engine/config.mk.
export CELL_PAD_IN_SITES_DETAIL_PLACEMENT = 0
export SKIP_CTS_REPAIR_TIMING = 1
export SKIP_INCREMENTAL_REPAIR = 1

# Block macros pin their power on M6; platform macro grid only connects M4-M5.
export PDN_TCL = $(dir $(DESIGN_CONFIG))pdn.tcl

# Merge the two engines' real geometry so the macros are not empty boxes in the
# top-level GDS.  ADDITIONAL_GDS takes the LAST assignment.
export ADDITIONAL_GDS = $(VIEWS)/sparse_ternary_top/6_1_merged.gds \
                        $(VIEWS)/int8_engine/6_1_merged.gds \
                        $(SRAM_GDS)/srambank_64x4x64_6t122.gds \
                        $(SRAM_GDS)/srambank_256x4x64_6t122.gds

# repair_design inserts ~139k buffers here (90,844 -> 229,495 instances) and
# 20,641 of them cannot be legalised (DPL-0036).  This block is hardened only to
# CTS and handed to the parent as an abstract, so the buffering is not needed.
export SKIP_REPAIR_DESIGN = 1

# The block abstracts are CTS views and only obstruct M1-M6, while u_st routes
# on M7-M9 and u_int8 on M7.  macro_placement.tcl adds matching dbObstructions
# over each block, so the parent may use M7-M9 everywhere else.  M5/M6
# saturated (15k DRC each); M7-M9 are free outside the two blocks.
export MAX_ROUTING_LAYER = M9

# CUGR finishes this design in 23 min; FastRoute needed >5 h for 4 of its 30
# congestion iterations on the same netlist.
export GLOBAL_ROUTE_ARGS = -use_cugr
