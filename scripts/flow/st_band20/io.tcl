# Pin direction constraints (sourced by floorplan.tcl at 2_1, honoured by
# place_pins at 3_2).  The parent DSA places the PE to the RIGHT of this LUT, so
# the 8192-bit wt_i8_block bus must land on the right edge -- otherwise the
# parent has to route thousands of nets around a macro.
proc _match_pins {regex} {
  set out {}
  foreach p [get_ports -regex .*] {
    set n [get_property $p name]
    if {[regexp $regex $n]} { lappend out $n }
  }
  return $out
}
set _blk [_match_pins {wt_i8_block}]
puts "IO_CONSTRAINTS: wt_i8_block pins = [llength $_blk] -> RIGHT edge"
set_io_pin_constraint -direction right -pin_names $_blk
