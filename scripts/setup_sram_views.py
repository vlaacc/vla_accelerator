"""Prepare the SRAM physical views used by every flow stage.

Two transformations of the ASAP7 kit macros:

  1. LEF: keep only the M4 pad on wd/dataout (see m4_only_pins.py).
  2. GDS: rename the top cell from the bit-width name to the LEF macro name.

The LIB files are used straight from the kit.

usage: setup_sram_views.py
Reads the paths exported by scripts/env.sh (SRAM_KIT, SRAM_KIT_GDS, SRAM_VIEWS).
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

KIT_LEF = os.path.join(os.environ["SRAM_KIT"], "LEF")
KIT_GDS = os.environ["SRAM_KIT_GDS"]
OUT_LEF = os.environ["SRAM_VIEWS"]
OUT_GDS = os.environ["SRAM_GDS"]

# LEF macro name -> kit GDS top cell
MACROS = {
    "srambank_64x4x64_6t122": "srambank_32b",
    "srambank_256x4x64_6t122": "srambank_128b",
}

os.makedirs(OUT_LEF, exist_ok=True)
os.makedirs(OUT_GDS, exist_ok=True)

for macro, gds_cell in MACROS.items():
    src_lef = os.path.join(KIT_LEF, macro + ".lef")
    dst_lef = os.path.join(OUT_LEF, macro + ".lef")
    subprocess.run(
        [sys.executable, os.path.join(HERE, "m4_only_pins.py"), src_lef, dst_lef],
        check=True,
    )

    src_gds = os.path.join(KIT_GDS, gds_cell + ".gds")
    dst_gds = os.path.join(OUT_GDS, macro + ".gds")
    subprocess.run(
        [
            "klayout", "-b", "-r", os.path.join(HERE, "rename_gds_cell.rb"),
            "-rd", "in=" + src_gds,
            "-rd", "out=" + dst_gds,
            "-rd", "old=" + gds_cell,
            "-rd", "new=" + macro,
        ],
        check=True,
    )

print("SRAM views written to", OUT_LEF, "and", OUT_GDS)
