# Rename the top cell of a GDS file (KLayout batch mode).
#
# The ASAP7 SRAM kit names its GDS top cells by bit width (srambank_32b,
# srambank_128b) while the LEF/LIB macros are named by geometry
# (srambank_64x4x64_6t122, srambank_256x4x64_6t122).  KLayout's merge step in
# the flow matches by cell name, so the GDS cell must carry the LEF name.
#
# usage:
#   klayout -b -r rename_gds_cell.rb \
#       -rd in=srambank_32b.gds -rd out=srambank_64x4x64_6t122.gds \
#       -rd old=srambank_32b -rd new=srambank_64x4x64_6t122

ly = RBA::Layout::new
ly.read($in)

cell = ly.cell($old)
raise "cell '#{$old}' not found in #{$in}" if cell.nil?

cell.name = $new
ly.write($out)
puts "wrote #{$out} (top cell #{$new})"
