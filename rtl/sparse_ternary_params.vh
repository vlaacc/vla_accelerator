// sparse_ternary_params.vh — Sparse Ternary LUT Engine parameters
// Independent from sparsenn_params.vh (no namespace conflicts)
`ifndef SPARSE_TERNARY_PARAMS_VH
`define SPARSE_TERNARY_PARAMS_VH

// ── Core Configuration ──
`define ST_R_CORES              2
`define ST_T_COPIES_PER_CORE    16
`define ST_N_CH_PER_COPY        16
`define ST_N_CH_PER_CORE        (`ST_T_COPIES_PER_CORE * `ST_N_CH_PER_COPY)  // 256
`define ST_N_CH_TOTAL           (`ST_R_CORES * `ST_N_CH_PER_CORE)            // 4096

// ── Weight precision mode: RUNTIME (wt_mode port, 1=ternary 0=W4) ──
// The weight word per channel per tile is 64 bit in BOTH modes
// (16x4-bit W4 == 32x2-bit ternary), so the weight SRAM/cache width is unchanged.
// The datapath is sized for the ternary (max) config; the runtime wt_mode signal
// selects K-per-tile / tiles-per-group / weight-bit granularity (W4 zero-pads the
// upper activation half — see sparse_ternary_core.v).
`define ST_WT_WORD               64     // weight word per channel per tile (invariant)

// ── Tile / Group (max = ternary config; wt_mode picks runtime values) ──
`define ST_TILE_K_MAX            32     // K per tile, ternary (max); W4 uses 16
`define ST_TILE_K_W4             16     // K per tile, W4
`define ST_TILES_PER_GROUP_MAX   8      // tiles per group, W4 (max); ternary uses 4
`define ST_TILE_K                `ST_TILE_K_MAX   // legacy alias = max (runtime modules use ST_TILE_K_MAX)
`define ST_N_GROUPS_PER_TILE     8      // groups per tile (max; W4 uses 4)
`define ST_GROUP_K               128
`define ST_TILES_PER_GROUP       (`ST_GROUP_K / `ST_TILE_K)                  // 4 (ternary); W4 = 8
`define ST_GROUP_SIZE            (`ST_TILE_K / `ST_N_GROUPS_PER_TILE)        // 4

// ── Data Types ──
`define ST_ACT_WIDTH            4
`define ST_WT_BITS              2      // weight bits, ternary (W4 = 4, runtime)
`define ST_WT_PER_TILE          `ST_TILE_K   // weights per tile (32 max / 16 W4); word = ST_WT_WORD both
`define ST_WT_POS16_WIDTH       16
`define ST_SUBSUM_WIDTH         24     // 完整 dot sub_sum (W8 最坏 17b，24b 富余)
`define ST_SCALE_WIDTH          16
`define ST_ZP_WIDTH             16
`define ST_ACC_WIDTH            32     // 累加器 32b (复用 128b SRAM: 4 lane x 32b)

// ── BUILD Unit ──
`define ST_BUILD_N_OUTPUTS      (1 << `ST_GROUP_SIZE)                         // 16 subset sums per group
`define ST_BUILD_REGS_PER_GROUP (`ST_BUILD_N_OUTPUTS * 7)                    // 112b
`define ST_BUILD_TOTAL_REGS     (`ST_N_GROUPS_PER_TILE * `ST_BUILD_REGS_PER_GROUP) // 896b = 112B (8 groups, max)
`define ST_BUILD_N_ADDERS       (`ST_N_GROUPS_PER_TILE * 11)   // 88 (8 groups, max)

// ── COMPUTE Pipeline ──
`define ST_COMP_STAGES          4

// ── 1 GHz timing: pipelined bit-serial shift-accumulate compute path ──
// 固定 8 级 MSB-first Horner 流水线，每级 16-bit 加法用单拍 carry-select 加法器
// （低 8 位 ripple + 高 8 位双路并行 + MUX），关键路径 = 8-bit 加法 + MUX (~0.5ns)。
// 寄存器切点：
//   ST_PIPE_WEIGHT : weight 寄存器 (128b，供 bit-plane 转置)
//   ST_PIPE_BITS   : 8 个 bit-stage，每 stage 1 个 16-bit 加法 (SUB=1，无拆分)
// ST_RES_DELAY = presentation -> sub_sum_out valid。
// W<8 时权重符号扩展，高位符号位在 Horner 累加中自动抵消，流水线恒 8 级。
`define ST_PIPE_WEIGHT          1
`define ST_PIPE_BITS            4             // Horner depth (always 4 stages: W4=4 bits, ternary=2 bits reused as 2x2-stage)
`define ST_PIPE_SUB             1             // 每 bit-stage 的子级数（1 = 单拍 16-bit 快速加法器，无拆分）
`define ST_FINAL_ACC_STAGES     4             // 最终累加流水级：平衡树 pair + tile_dot + 累加低半 + 累加高半 (lut_compute_unit)
`define ST_FINAL_ACC_STAGES_CS  5             // carry-save 全链 (lut_compute_unit_cs)：树A(8->4) + 树B(4->2) + 冗余累加 + CPA 低半 + CPA 高半

// ── act_compress: activation sparsity compression (K=4 dual-mode) ──
// 每 tile 检测非零激活；N<=K 走压缩路径(16->4 gather)，N>K 走稠密回退。
// `ifdef ST_ACT_COMPRESS 使能；不定义则回退为原始稠密路径（用于 baseline 对比综合）。
`define ST_ACT_COMPRESS
`define ST_ACT_COMPRESS_K       4
// gather 流水级数：压缩器(idx/compress_en/act_compact) + 权重延迟寄存器各 1 级，
// 把 data-dependent gather 移出关键路径。0 = 纯组合（baseline/未流水）。
`ifdef ST_ACT_COMPRESS
  `define ST_ACT_COMPRESS_STAGES 0   // Option B：权重不经 weight_all_r 延迟，gps={sd,build_out} 抵消 +1
`else
  `define ST_ACT_COMPRESS_STAGES 0
`endif

// reuse-Horner unified: weight(1) + 4 Horner stages(4) + tree(0, combinational) +
// combine(0) + acc(3) = 10.  Both modes align; the 3-stage CSA+CPA acc_block_reg
// adds +1 latency (vs the old 2-stage), so the delay is 10.
`define ST_RES_DELAY   10

// ── SRAM Reuse (existing OpenRAM macros) ──
// Weight: 128KB total, 2 banks × 64KB (1024×512b per bank, 4 scratchpads)
`define ST_WT_SRAM_DEPTH        1024
`define ST_WT_SRAM_WIDTH        512
// Activation: 256KB total, 2 banks × 128KB (2048×512b per bank, 8 scratchpads)
`define ST_ACT_SRAM_DEPTH       2048
`define ST_ACT_SRAM_WIDTH       512
// Accumulator: 128KB total, 2 cores × 2 banks × 32KB (512×512b per bank, 4 scratchpads)
`define ST_ACC_SRAM_DEPTH       512
`define ST_ACC_SRAM_WIDTH       512

// ── Pipeline Timing ──
`define ST_BUILD_CYCLES_PER_TILE   1
`define ST_COMPUTE_LATENCY         `ST_COMP_STAGES
`define ST_SCALE_ZP_LATENCY        4   // scale_zp pipeline depth (legacy, controller FSM)
`define ST_ACCUMULATE_LATENCY      3   // accumulate pipeline depth (legacy, controller FSM)
`define ST_VECTOR_LATENCY          71  // vector_unit: 16-lane parallel, max 71 cycles (LayerNorm, 5-stage fp32_add/fp32_mul pipelines)

`endif // SPARSE_TERNARY_PARAMS_VH
