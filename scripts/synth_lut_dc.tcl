# Design Compiler synthesis of the LUT engine (sparse_ternary_top).
#
# Yosys cannot map this design: the generic techmap of its arithmetic peaks
# above 242 GB.  The ORFS st_band20 stage therefore consumes a gate netlist.
#
# Usage:
#   source scripts/env.sh
#   export DC_SETUP=/path/to/asap7_dc_setup.tcl   # defines target/link_library
#   dc_shell -f scripts/synth_lut_dc.tcl
#
# DC_SETUP must load the ASAP7 7.5T RVT/TT libraries and set target_library /
# link_library / search_path.  If it is unset, the libraries already loaded in
# the session are used.

if {![info exists ::env(SPARSEDSA_ROOT)]} {
    error "source scripts/env.sh first"
}

set SRC $::env(SPARSEDSA_ROOT)/rtl
set OUT [file dirname $::env(LUT_NETLIST)]
file mkdir $OUT

if {[info exists ::env(DC_SETUP)] && [file exists $::env(DC_SETUP)]} {
    source $::env(DC_SETUP)
}

set_app_var search_path     "$SRC $SRC/sram_7nm [get_app_var search_path]"
set_app_var hdlin_include_paths $SRC
set_app_var hdlin_enable_hier_map true
set_app_var hdlin_prohibit_nontri_multiple_drivers false
set_app_var sh_enable_page_mode false

# LUT engine modules.  SRAM wrappers are elaborated around the ASAP7 kit macros.
foreach f {sparse_ternary_top sparse_ternary_core sparse_ternary_ctrl
           act_read_path act_line_buffer act_port_mux act_compressor addr_gen
           acc_rmw_unit build_unit ternary_engine lut_compute_unit_mixed csa3
           weight_gather weight_tile_cache sub_sum_fifo acc_block_reg} {
    analyze -format sverilog $SRC/$f.v
}
foreach f {sram_act_bank sram_wgt_bank sram_acc_bank} {
    analyze -format sverilog $SRC/sram_7nm/$f.v
}

elaborate sparse_ternary_top
link
current_design sparse_ternary_top
set_verification_top

create_clock -name clk -period 1.0 [get_ports clk]
set_input_delay  0.1 -clock clk [remove_from_collection [all_inputs] [get_ports clk]]
set_output_delay 0.1 -clock clk [all_outputs]

report_design -nosplit > $OUT/pre_report.rpt
puts "PRE_COMPILE_DONE"

compile_ultra

report_area   > $OUT/area.rpt
report_timing > $OUT/timing.rpt
write -format verilog -hierarchy -output $::env(LUT_NETLIST)
puts "COMPILE_DONE $::env(LUT_NETLIST)"

exit
