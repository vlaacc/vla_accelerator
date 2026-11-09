# Design-local override of platforms/asap7/fastroute.tcl (which hardcodes 0.25
# and therefore silently ignores ROUTING_LAYER_ADJUSTMENT -- floorplan.tcl skips
# that variable whenever FASTROUTE_TCL is set).
#
# ORFS variables.yaml, ROUTING_LAYER_ADJUSTMENT: "Sometimes, global routing will
# succeed with lower values and fail with higher values ... Start with a too low
# value, such as 0.10, and bisect."
puts "CUSTOM FASTROUTE: signal layer adjustment = 0.10 (platform default was 0.25)"
set_global_routing_layer_adjustment $::env(MIN_ROUTING_LAYER)-$::env(MAX_ROUTING_LAYER) 0.10
set_routing_layers -clock $::env(MIN_CLK_ROUTING_LAYER)-$::env(MAX_ROUTING_LAYER)
set_routing_layers -signal $::env(MIN_ROUTING_LAYER)-$::env(MAX_ROUTING_LAYER)
