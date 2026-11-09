"""Restrict the ASAP7 SRAM data pins to their M4 pad.

The kit LEF declares each wd/dataout pin as an M3 stub + M4 pad + a preset V3
cut sitting at one end of a 0.648 x 0.024 pad.  The detailed router escapes on
the M3 stub and drops its own VIA34 in the pad right next to that preset cut,
which leaves zero clearance against a 0.034 cut-spacing rule (~6.2k of the LUT
block's ~7.1k DRCs are exactly that).

Dropping M3+V3 from those pins leaves only the M4 pad, so the router connects
on M4 (free above these macros: their OBS blocks M1-M3) and the macro's real
internal V3 still joins the pad to its M3 datapath.  The GDS is untouched; only
the LEF access window changes.

usage: m4_only_pins.py <kit_lef> <out_lef>
"""
import re
import sys

src, dst = sys.argv[1], sys.argv[2]
DATA_PIN = re.compile(r"(wd|dataout)\[\d+\]")

out = []
in_data = False
skip = False
for line in open(src):
    s = line.strip()
    if s == "OBS":
        in_data = False
    if s.startswith("PIN ") and len(s.split()) == 2:
        in_data = bool(DATA_PIN.fullmatch(s.split()[1]))
        skip = False
    if in_data:
        if s.startswith("LAYER "):
            skip = s.split()[1] in ("M3", "V3")
        if s == "END":          # closes the PORT; always keep it
            skip = False
        elif skip:
            continue
    out.append(line)

open(dst, "w").writelines(out)

txt = "".join(out)
print("wrote", dst, "bytes", len(txt))
for name in ("wd[0]", "dataout[0]", "ADDRESS[0]", "VDD"):
    i = txt.index("PIN %s\n" % name)
    body = txt[i:i + 400]
    print("%-12s layers: %s" % (name, re.findall(r"LAYER (\w+)", body)[:4]))
