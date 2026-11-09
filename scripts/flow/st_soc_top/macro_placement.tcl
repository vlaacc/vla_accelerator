# st_soc_top (DSA) — two macros only.
#
#   u_st    (LUT)    760 x 1140  @ (20, 60)
#   u_int8  (PE)    1200 x 1100  @ (820, 60)
#
# The INT8 act/acc SRAM banks (80 hard macros) and the vector epilogue now live
# INSIDE int8_engine: keeping them at this level put ~9,200 + 8,192 wires between
# the engine and its memories on the parent's routing layers, which no floorplan
# could absorb (the parent's detailed route stalled at ~177k violations).
set block [ord::get_db_block]
set dbu   [$block getDbUnitsPerMicron]

proc place_at {block dbu name x y {orient R0}} {
    set inst [$block findInst $name]
    if {$inst eq "NULL"} { error "place_at: missing instance $name" }
    $inst setOrient $orient
    $inst setLocation [expr {int(round($x * $dbu))}] [expr {int(round($y * $dbu))}]
    $inst setPlacementStatus FIRM
    set m [$inst getMaster]
    puts [format "PLACED %-8s %4.0f x %-4.0f @ (%.0f,%.0f)" $name \
              [expr {[$m getWidth]/$dbu}] [expr {[$m getHeight]/$dbu}] $x $y]
}

place_at $block $dbu u_st    20.0  60.0
place_at $block $dbu u_int8  820.0 60.0
puts "PLACED 2 big macros"

# The block abstracts come from the CTS stage and only obstruct M1-M6, while the
# blocks keep routing on M7-M9 (LUT) / M7 (PE).  Obstruct those layers over each
# block so the parent cannot drop wires onto the blocks' own metal.
proc add_routing_obs {block tech name layers} {
    set inst [$block findInst $name]
    if {$inst eq "NULL"} { return }
    lassign [$inst getLocation] x y
    set m [$inst getMaster]
    set w [$m getWidth]
    set h [$m getHeight]
    foreach lyr $layers {
        set l [$tech findLayer $lyr]
        if {$l eq "NULL"} { error "layer $lyr not found" }
        set obs [odb::dbObstruction_create $block $l $x $y \
                      [expr {$x + $w}] [expr {$y + $h}]]
        # Power must still pass: the block's M6 macro-grid stripes can only
        # reach the parent VDD through M7+ above the block (PSM-0069 otherwise).
        odb::dbObstruction_setExceptPGNetsObstruction $obs 1
    }
    puts "OBS $name -> $layers"
}
set tech [$block getTech]
add_routing_obs $block $tech u_st   {M7 M8 M9}
add_routing_obs $block $tech u_int8 {M7}
