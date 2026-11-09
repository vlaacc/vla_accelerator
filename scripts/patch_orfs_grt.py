"""Let a design opt out of GRT's hard congestion error on incremental passes.

ORFS's global_route.tcl calls `global_route -end_incremental` after
repair_design / repair_timing / recover_power without forwarding
GLOBAL_ROUTE_ARGS.  A design that lets GRT finish with -allow_congestion then
still dies on those passes (GRT-0116 on a handful of residual overflow tracks).
This patch gates the extra flag on ALLOW_GRT_CONGESTION.

The patch is idempotent.  Run it once after checking out ORFS:

    python3 scripts/patch_orfs_grt.py
"""
import os
import sys

orfs_home = os.environ.get("ORFS_HOME", "")
p = os.path.join(orfs_home, "flow", "scripts", "global_route.tcl")
if not orfs_home or not os.path.isfile(p):
    sys.exit("ORFS_HOME does not point at an OpenROAD-flow-scripts checkout")

s = open(p).read()

anchor = "source_env_var_if_exists PRE_GLOBAL_ROUTE_TCL"
if anchor not in s:
    sys.exit("global_route.tcl: anchor not found (already patched or new ORFS layout)")

if "grt_allow_args" not in s:
    s = s.replace(anchor, anchor + "\n\n"
                  "# Set ALLOW_GRT_CONGESTION=1 (design config) to keep the incremental\n"
                  "# passes from turning residual overflow into a hard error.\n"
                  "set ::grt_allow_args {}\n"
                  "if {[env_var_exists_and_non_empty ALLOW_GRT_CONGESTION]} {\n"
                  "  set ::grt_allow_args {-allow_congestion}\n"
                  "}", 1)

n = s.count("-end_incremental {*}$res_aware")
s = s.replace("-end_incremental {*}$res_aware",
              "-end_incremental {*}$res_aware {*}$::grt_allow_args")
open(p, "w").write(s)
print("patched end_incremental calls:", n)
