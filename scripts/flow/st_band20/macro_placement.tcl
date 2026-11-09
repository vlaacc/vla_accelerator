# st_band17: ALL 98 SRAM macros in ONE band at the die bottom (NBAND=1), with
# all standard cells above.  Reason: 100% of the residual DRC is the u_wt_cache
# logic routing OVER the 9.6-um-wide weight SRAM macros and dropping V3 vias into
# the M3 OBS (GRT treats macro OBS as a soft penalty).  Keeping every logic net on
# one side of the macro block removes the crossing altogether.
# NBAND horizontal bands spread over the die height instead of one band at the
# bottom.  A single bottom band forces every bit of the 16,384-bit acc bus to
# cross the whole block (baseline measured: avg HPWL 514 um, total 8.42 mm).
# Spreading the macros lets GPL put the drain logic between the bands.
set block [ord::get_db_block]
set dbu   [$block getDbUnitsPerMicron]
set NBAND 1
set die [$block getDieArea]
set x0 [$die xMin]; set x1 [$die xMax]
set y0 [$die yMin]; set y1 [$die yMax]
set margin [expr {int(round(10.008 * $dbu))}]
set gap    [expr {int(round( 4.032 * $dbu))}]
set macros {}
foreach inst [$block getInsts] {
    set m [$inst getMaster]
    if {[$m getType] == "BLOCK"} { lappend macros [list $inst [$m getWidth] [$m getHeight]] }
}
puts "BAND-MACRO: [llength $macros] macros, die [expr {($x1-$x0)/$dbu}] x [expr {($y1-$y0)/$dbu}] um, $NBAND bands"
set macros [lsort -command {apply {{a b} {expr {[lindex $b 2] - [lindex $a 2]}}}} $macros]
set ng [llength $macros]
set per [expr {int(ceil(double($ng)/$NBAND))}]
set avail [expr {$y1 - $y0 - 2*$margin}]
set band_h [expr {$avail / $NBAND}]
set bi 0
set total_rows 0
for {set i 0} {$i < $ng} {incr i $per} {
    set g [lrange $macros $i [expr {$i+$per-1}]]
    set top [expr {$y0 + $margin + ($bi+1)*$band_h}]
    set cur_x $margin
    set cur_y [expr {$y0 + $margin + $bi*$band_h}]
    set row_h 0
    set rows 1
    foreach e $g {
        lassign $e inst w h
        if {$cur_x + $w > $x1 - $margin} {
            set cur_x $margin
            set cur_y [expr {$cur_y + $row_h + $gap}]
            set row_h 0
            incr rows
        }
        if {$cur_y + $h > $top} { error "BAND-MACRO: overflow in band $bi" }
        $inst setOrient R0
        $inst setLocation $cur_x $cur_y
        $inst setPlacementStatus FIRM
        set cur_x [expr {$cur_x + $w + $gap}]
        if {$h > $row_h} { set row_h $h }
    }
    puts "BAND-MACRO: band $bi -> [llength $g] macros, $rows rows"
    incr total_rows $rows
    incr bi
}
puts "BAND-MACRO: done, $total_rows macro rows"
