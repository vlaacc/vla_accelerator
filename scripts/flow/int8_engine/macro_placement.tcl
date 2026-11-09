# int8_engine macro placement — the block now owns its 80 SRAM macros.
#
#   16 x srambank_256x4x64 (30.348 x 77.76)  = act bank, one row along the bottom
#   64 x srambank_64x4x64   ( 9.612 x 77.76)  = acc bank, one row above it
#
# Both rows hug the bottom edge; the mesh + vector logic sits above y=200.
# Every placed instance is written to sram_positions.csv next to this script so the
# final merged GDS can be annotated with where each SRAM physically sits.
#
# NB: yosys emits escaped identifiers, so instance names carry literal backslashes
# ("u_acc_sram/slice\[0\].u"); match on the stripped name instead of findInst.
set block [ord::get_db_block]
set dbu   [$block getDbUnitsPerMicron]

set csv [open [file join [file dirname [info script]] sram_positions.csv] w]
puts $csv "instance,master,x_um,y_um,width_um,height_um"

proc place_and_record {block dbu csv inst x y} {
    $inst setOrient R0
    $inst setLocation [expr {int(round($x * $dbu))}] [expr {int(round($y * $dbu))}]
    $inst setPlacementStatus FIRM
    set m [$inst getMaster]
    set nm [string map {"\\" ""} [$inst getName]]
    puts $csv [format "%s,%s,%.3f,%.3f,%.3f,%.3f" $nm [$m getName] $x $y \
                    [expr {[$m getWidth]/$dbu}] [expr {[$m getHeight]/$dbu}]]
    return $nm
}

array set act {}
array set acc {}
foreach inst [$block getInsts] {
    if {[[$inst getMaster] getType] != "BLOCK"} { continue }
    set nm [string map {"\\" ""} [$inst getName]]
    if {[regexp {^u_act_sram/slice\[([0-9]+)\]\.u$} $nm -> idx]} { set act($idx) $inst }
    if {[regexp {^u_acc_sram/slice\[([0-9]+)\]\.u$} $nm -> idx]} { set acc($idx) $inst }
}

set na 0
foreach i [lsort -integer [array names act]] {
    place_and_record $block $dbu $csv $act($i) [expr {15.0 + $i * 34.4}] 15.0
    incr na
}
set nc 0
foreach i [lsort -integer [array names acc]] {
    place_and_record $block $dbu $csv $acc($i) [expr {15.0 + $i * 16.0}] 100.0
    incr nc
}
close $csv
puts "PE macros: placed $na act + $nc acc SRAMs (act y=15, acc y=100), positions in sram_positions.csv"
if {$na != 16 || $nc != 64} { error "PE macros: expected 16 act + 64 acc, got $na + $nc" }
