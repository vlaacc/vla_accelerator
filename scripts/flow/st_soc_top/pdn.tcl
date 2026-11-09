####################################
# st_soc_top PDN
#
# Same as platforms/asap7/openRoad/pdn/grid_strategy-M1-M2-M5-M6.tcl, except the
# macro grid.  The two hardened blocks (u_st = sparse_ternary_top, u_int8 =
# int8_engine) were built by the ORFS flow, whose stdcell grid pins its power out
# on M6 -- so their abstract power pins are M6 only.  The platform's default
# macro grid connects M4-M5, which left both grids with no shapes:
#   PDN-0232 The grid "CORE_macro_grid_1 - u_int8" does not contain any shapes
#   PDN-0233 Failed to generate full power grid.
# Connect them on M5-M6 instead.
####################################
add_global_connection -net {VDD} -inst_pattern {.*} -pin_pattern {^VDD$} -power
add_global_connection -net {VDD} -inst_pattern {.*} -pin_pattern {^VDDPE$}
add_global_connection -net {VDD} -inst_pattern {.*} -pin_pattern {^VDDCE$}
add_global_connection -net {VSS} -inst_pattern {.*} -pin_pattern {^VSS$} -ground
add_global_connection -net {VSS} -inst_pattern {.*} -pin_pattern {^VSSE$}
global_connect

set_voltage_domain -name {CORE} -power {VDD} -ground {VSS}

define_pdn_grid -name {top} -voltage_domains {CORE} -pins {M6}
add_pdn_stripe -grid {top} -layer {M1} -width {0.018} -pitch {0.54} -offset {0} -followpins
add_pdn_stripe -grid {top} -layer {M2} -width {0.018} -pitch {0.54} -offset {0} -followpins
add_pdn_stripe -grid {top} -layer {M5} -width {0.12} -spacing {0.072} -pitch {5.4} -offset {0.300}
add_pdn_stripe -grid {top} -layer {M6} -width {0.288} -spacing {0.096} -pitch {5.4} -offset {0.513}
add_pdn_connect -grid {top} -layers {M1 M2}
add_pdn_connect -grid {top} -layers {M2 M5}
add_pdn_connect -grid {top} -layers {M5 M6}

# Block macros pin their power on M6.
define_pdn_grid -name {blk_macro_grid} -voltage_domains {CORE} -macro \
  -instances {u_st u_int8} -orient {R0 R180 MX MY} -halo {2.0 2.0 2.0 2.0}
add_pdn_connect -grid {blk_macro_grid} -layers {M5 M6}
