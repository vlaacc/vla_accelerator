# Flow notes

Non-obvious constraints behind the configurations in `scripts/flow/`.

## SRAM views

The ASAP7 SRAM kit declares each `wd`/`dataout` pin as an M3 stub, an M4 pad
and a preset V3 cut at one end of the pad.  The detailed router escapes on the
M3 stub and drops its own VIA34 next to that cut; the cut-to-cut spacing rule
cannot be met, so every macro data pin contributes a `V3 Cut Short` violation
(about 6.2k of 7.1k DRCs on the LUT block).

`scripts/m4_only_pins.py` removes the M3 stub and V3 cut from those pins,
leaving the M4 pad as the only access window.  The macro GDS is unchanged: its
real internal V3 still connects the pad to the M3 datapath.  DRC dropped from
~7k to single digits with this change.

The kit also names GDS top cells by bit width while LEF/LIB macros are named by
geometry.  `scripts/rename_gds_cell.rb` renames the top cell so KLayout's merge
step can match them:

| Kit GDS | LEF / LIB macro |
|---------|-----------------|
| `srambank_32b.gds` | `srambank_64x4x64_6t122` |
| `srambank_128b.gds` | `srambank_256x4x64_6t122` |

## LUT synthesis

Yosys cannot map `sparse_ternary_top`: the generic techmap of its arithmetic
peaks above 242 GB regardless of `SYNTH_MEMORY_MAX_BITS`, `ADDER_MAP_FILE` or
`SYNTH_OPERATIONS_ARGS`.  The `st_band20` stage therefore takes a gate netlist
produced by Design Compiler (`scripts/synth_lut_dc.tcl`).  The netlist is not
checked in; set `LUT_NETLIST` or run the DC script.

## Block abstracts

The top-level designs consume the blocks as abstracts generated from the
**CTS** stage:

```bash
make DESIGN_CONFIG=./designs/asap7/st_band20/config.mk \
     ABSTRACT_SOURCE=4_1_cts ABSTRACT_BLOAT_OBS=0 do-generate_abstract
```

A route-stage abstract writes the block's own M7+ wiring into the LEF as
obstructions, which later fails the parent PDN with `PDN-0006`.

## Routing and repair

- Global-route congestion iterations can stall for hours on these blocks.
  `GLOBAL_ROUTE_ARGS = -congestion_iterations 1 -allow_congestion` bounds the
  effort; detailed route cleans up the remainder.
- ORFS's incremental global-route passes (after repair / power recovery) do not
  forward `GLOBAL_ROUTE_ARGS`.  `scripts/patch_orfs_grt.py` gates
  `-allow_congestion` on `ALLOW_GRT_CONGESTION` for those passes.
- Antenna repair is skipped: the PDK does not ship a `CORE ANTENNACELL`, so the
  repair step can run for many hours with no progress.
- Timing repair is bounded (`TNS_END_PERCENT = 0`) and CTS-time repair is
  skipped.  These are physical-bring-up settings, not a timing-closed signoff.

## Top-level floorplan

- The block macros are shelf-packed and set `FIRM`; `place_macro` cannot satisfy
  `MPL-0005` on ASAP7-abstracted pins.
- `st_soc_top/macro_placement.tcl` adds M7-M9 obstructions over the LUT block
  and M7 over the PE block so the parent cannot drop wires onto the blocks' own
  metal.  The obstructions are marked PG-exempt; otherwise the block's M6 power
  stripes cannot reach the parent VDD (`PSM-0069`).
- The block PDN grids connect on M6, matching the abstracts' power pins.

## Result-directory hygiene

`make` does not notice changes to die size, macro placement or LEFs.  After any
floorplan change, remove the affected `results/`, `logs/` and `objects/`
directories for that design before rerunning.  Delete with absolute paths and
check for unique artifacts (merged GDS, abstracts) first.
