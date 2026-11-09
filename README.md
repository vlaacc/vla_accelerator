# SparseDSA

RTL and an RTL-to-GDSII flow for a sparse-ternary / INT8 accelerator built for
compressed FFN and attention matmuls in vision-language-action models.

The design combines two hard blocks under one CSR shell:

- **LUT engine** (`sparse_ternary_top`) — a sparse ternary / W4 weight engine
  with LUT-based mixed-precision compute units, per-group scale handling and a
  streamed accumulator drain.
- **INT8 engine** (`int8_engine`) — four 32×32 INT8 meshes with their own
  activation / accumulator SRAM banks and a 128-lane vector epilogue.

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│  st_soc_top                                                          │
│                                                                      │
│  ┌──────────────┐   ┌───────────────────────────┐                   │
│  │ config_regs  │   │ sparse_ternary_top  (LUT) │                   │
│  │  (CSR)       │   │  ternary / W4 datapath    │                   │
│  │              │   │  wt/act/acc SRAM banks    │                   │
│  └──────┬───────┘   └─────────────┬─────────────┘                   │
│         │                         │  256-bit weight column stream   │
│         │                         ▼                                 │
│         │           ┌───────────────────────────┐                   │
│         └──────────►│ int8_engine               │                   │
│                     │  4 × 32×32 INT8 mesh      │                   │
│                     │  128-lane vector epilogue │                   │
│                     │  act / acc SRAM banks     │                   │
│                     └───────────────────────────┘                   │
│                                                                      │
│  AXI4-Lite CSR · 512-bit weight / activation / result AXI4-Stream    │
└──────────────────────────────────────────────────────────────────────┘
```

| Block | Top module | Die (µm) | Hard macros | Notes |
|-------|-----------|---------:|------------:|-------|
| LUT engine | `sparse_ternary_top` | 760 × 1140 | ~290 SRAM | ternary (K=32) and W4 (K=16) modes |
| INT8 engine | `int8_engine` | 1200 × 1100 | 80 SRAM | 4 meshes, K=32, 4096-bit accumulator |
| Chip top | `st_soc_top` | 3000 × 1500 | 2 blocks | CSR + 512-bit streaming ports |

Process: ASAP7 predictive 7.5T RVT/TT.

## Repository structure

```
SparseDSA/
├── rtl/                              # Verilog sources
│   ├── st_soc_top.v                  # chip top: CSR + both engines
│   ├── config_regs.v                 # AXI4-Lite CSR file
│   ├── sparse_ternary_top.v          # LUT engine top
│   ├── sparse_ternary_core.v         # tile-major compute datapath
│   ├── sparse_ternary_ctrl.v         # LUT engine sequencer
│   ├── ternary_engine.v              # per-tile dot-product datapath
│   ├── lut_compute_unit_mixed.v      # LUT / borrow-save mixed compute
│   ├── acc_block_reg.v               # register-blocked K=128 accumulator
│   ├── weight_gather.v               # weight gathering / packing
│   ├── weight_tile_cache.v           # tile weight cache
│   ├── act_read_path.v, act_line_buffer.v, act_port_mux.v,
│   │   act_compressor.v, addr_gen.v, build_unit.v, csa3.v,
│   │   acc_rmw_unit.v, sub_sum_fifo.v
│   ├── int8_engine.v                 # INT8 block top
│   ├── int8_mesh_top.v, int8_act_skew.v, pe_int8.v,
│   │   systolic_array_int8.v
│   ├── vector_unit_4096.v, vector_lane.v   # 128-lane fp32 epilogue
│   ├── fp32_*.v                      # fp32 arithmetic / transcendentals
│   └── sram_7nm/                     # ASAP7 SRAM macro wrappers
│       ├── sram_act_bank.v, sram_wgt_bank.v, sram_acc_bank.v
│       └── sram_act_bank_int8.v, sram_acc_bank_int8.v
├── scripts/
│   ├── env.sh                        # exported paths for all scripts
│   ├── setup_sram_views.py           # SRAM LEF/GDS preparation
│   ├── m4_only_pins.py               # M4-only data-pin LEF rewrite
│   ├── rename_gds_cell.rb            # KLayout GDS top-cell rename
│   ├── patch_orfs_grt.py             # ALLOW_GRT_CONGESTION ORFS patch
│   ├── synth_lut_dc.tcl              # Design Compiler run for the LUT netlist
│   ├── run_gds.sh                    # full block + top GDS flow
│   └── flow/                         # ORFS design configs
│       ├── st_band20/                # LUT engine block
│       ├── int8_engine/              # INT8 engine block
│       └── st_soc_top/               # chip top
├── docs/
│   └── flow_notes.md                 # known pitfalls and design decisions
└── README.md
```

## Flow

The flow uses [OpenROAD-flow-scripts](https://github.com/The-OpenROAD-Project/OpenROAD-flow-scripts)
(26Q1) with the built-in ASAP7 platform.  Both blocks are hardened first; the
top then instantiates them as macros.

### 1. Prerequisites

- ORFS 26Q1 checkout with `openroad`, `yosys` and `klayout` on `PATH`
- ASAP7 PDK + the SRAM generator output (LEF / LIB / verilog / gds)
- Optional: Synopsys Design Compiler plus ASAP7 `.db` libraries for the LUT
  netlist (Yosys cannot map that block)

### 2. Point the scripts at your installs

```bash
source scripts/env.sh
export ORFS_HOME=/path/to/OpenROAD-flow-scripts
export SRAM_KIT=/path/to/asap7_sram/generated
export SRAM_KIT_GDS=/path/to/asap7_sram/gds
```

### 3. Prepare the SRAM views (once)

```bash
python3 scripts/setup_sram_views.py
```

This writes the M4-only-pin LEFs and the renamed GDS macros into `build/`.

### 4. Synthesize the LUT netlist

```bash
export DC_SETUP=/path/to/asap7_dc_setup.tcl   # sets target/link libraries
dc_shell -f scripts/synth_lut_dc.tcl          # -> build/lut/sparse_ternary_top_mapped.v
```

If you already have a gate netlist, set `LUT_NETLIST=/path/to/netlist.v`.

### 5. Run RTL-to-GDSII

```bash
bash scripts/run_gds.sh
```

Output: `$ORFS_HOME/flow/results/asap7/st_soc_top/base/6_final.gds`.

The script links `scripts/flow/<block>` into the ORFS `flow/designs/asap7/`
tree, runs `st_band20`, `int8_engine` and `st_soc_top`, and copies each block's
CTS abstract and merged GDS into `build/views/` for the parent.

See `docs/flow_notes.md` for the non-obvious flow constraints (SRAM pin access,
abstract source stage, congestion-repair flags, and result-directory hygiene).
