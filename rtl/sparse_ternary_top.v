// sparse_ternary_top.v — Sparse Ternary LUT Engine Top Level (v28, tile-major)
// R=2 cores, AXI4-Stream DMA, tile-major weight path (16x sdp 1R1W + weight_tile_cache).
//
// v28: tile-major tiling (tile OUTER, batch INNER).  The weight tile stays resident
//   for N_BATCH=16 batches, so the 16Kbit/tile weight read bandwidth is thinned 16x and
//   a narrow 2Kbit SRAM + 1-tile double-buffered cache replaces the old weight_reg_file.
//   Each sparse_ternary_core now streams per-tile redundant dots (tile_s/tile_c) into a
//   register-block accumulator (acc_block_reg) that owns the 8-tile K=128 accumulation and
//   emits one 24-bit sub_sum per batch per channel; the sub_sum FIFO now writes one entry
//   per sub_sum_valid pulse (N_BATCH entries per group).
//
// Alignment: the activation SRAM has a 2-cycle read latency, so the activation path
//   (act_row) is 2 cycles behind the controller's build/compute schedule.  All other core
//   inputs (build_en / compute_en / act_tile_idx / group_start) and the weight-cache
//   tile_boundary are delayed by 2 cycles to match, so the core sees a uniformly-shifted
//   presentation stream (its sub_sum_valid output is likewise shifted by 2, which the FIFO
//   decoupling absorbs).
`include "sparse_ternary_params.vh"
`include "sparsenn_params.vh"

(* keep_hierarchy *) module sparse_ternary_top #(
    parameter R_CORES = `ST_R_CORES,
    parameter T_PER    = `ST_T_COPIES_PER_CORE,
    parameter N_CH_PER = `ST_N_CH_PER_CORE,
    parameter N_CH_TOT = R_CORES * N_CH_PER
) (
    input  wire                     clk,
    input  wire                     rst_n,

    // AXI4-Stream: weight DMA (512b)
    input  wire                     s_axis_wgt_tvalid,
    output wire                     s_axis_wgt_tready,
    input  wire [511:0]             s_axis_wgt_tdata,
    input  wire                     s_axis_wgt_tlast,

    // AXI4-Stream: activation DMA (512b)
    input  wire                     s_axis_act_tvalid,
    output wire                     s_axis_act_tready,
    input  wire [511:0]             s_axis_act_tdata,
    input  wire                     s_axis_act_tlast,

    // AXI4-Stream: result DMA (512b)
    output wire                     m_axis_out_tvalid,
    input  wire                     m_axis_out_tready,
    output wire [511:0]             m_axis_out_tdata,
    output wire                     m_axis_out_tlast,

    // Control
    input  wire                     start,
    input  wire                     weight_signed,   // 1=有符号 / 0=无符号权重
    input  wire                     wt_mode,         // 1=ternary(K=32) 0=W4(K=16)
    output wire                     done,
    output wire                     busy,            // started and not yet all_done
    input  wire [15:0]              dim_I,
    input  wire [15:0]              dim_J,
    input  wire [15:0]              dim_K,

    // ── Weight-block DMA trigger (host handshake) ──
    // One 1-cycle pulse per (k_group, j_pass) weight block, i.e. exactly the number of
    // times the ctrl enters S_LOAD_WGT.  The host must send EXACTLY ONE weight block per
    // pulse and must NOT start it earlier: the weight_tile_cache refills continuously
    // from SRAM words 0..N_TILES*CHUNKS-1 during S_GROUP, so a block DMA'd mid-group
    // would overwrite the weights the current group is still streaming.  The
    // arrived/consumed handshake makes a block that lands EARLY harmless to the control
    // flow, but not to the SRAM contents — hence the trigger is the contract.
    output wire                     dma_wt_start,

    // ── Activation-block DMA trigger (host handshake) ──
    // Same contract for activations: one pulse per activation block, and the host sends
    // exactly one full activation tile set per pulse.  The first pulse coincides with
    // `start`; later ones mark a k_group boundary (the activations are re-loaded for each
    // k_group, so dim_K > 128 needs one block per k_group).
    output wire                     dma_act_start,

    // ── Configuration error (sticky until the next `start`) ──
    // Set when the requested (dim_I, dim_J) needs more acc rows than the macro array
    // has (rows_needed = ceil(dim_I/R_CORES) * ceil(dim_J/N_CH_PER_CORE) > MAX_ROWS,
    // i.e. roughly dim_I * dim_J > 131,072 at scheme C's 256 rows; it was 65,536 at the
    // old 128).  Before this existed the row index simply truncated and the j_pass
    // blocks silently aliased onto each other.
    // While it is set the engine does NOT run: `busy` stays low, ctrl is held in IDLE
    // and `done` never pulses.  Reporting and refusing beats returning wrong data.
    output wire                     cfg_error,

    // Debug
    output wire [3:0]               dbg_state,

    // ── sub_sum FIFO overflow (sticky until the next `start`) ──
    // Set if the write gate ever refused a sub_sum entry.  With FIFO_BP_LEVEL doing
    // its job this must stay 0 for the whole run; a long-run test should assert it.
    output wire                     fifo_overflow,

    // ── result drain ownership (for the time-multiplexed top-level output port) ──
    output wire                     out_drain_busy,

    // ── INT8 weight block (frozen spec: the 8 Kbit INT8 block rides the LUT weight
    // broadcast, so there is no separate INT8 weight SRAM) ──
    // Streamed as 32 x 256 b columns: wt_i8_col carries column wt_i8_col_idx,
    // column n = wt_bcast[256n +: 256].  The old 8192-bit parallel port forced the
    // parent to route 8192 wires between two macros; the load still takes 32 cycles.
    output wire [255:0]             wt_i8_col,
    output wire [4:0]               wt_i8_col_idx,
    output wire                     wt_i8_valid
);

    // ── Local params / forward declarations ──
    // Scheme C: a WHOLE row is one access (2 cores x 256 ch x 32 b = 16384 b), so the
    // `half` bit is gone from the address and the row count is the address depth.
    localparam ROW_IDX_W = 8;   // acc SRAM row (absolute batch) index width (256 rows)
    localparam MAX_ROWS  = 1 << ROW_IDX_W;   // 256 physical acc rows
    // Scheme C4: a logical row is FOUR 4096-b sub-words (see acc_rmw_unit.v).
    localparam ACC_SUB_W  = 2;                       // 4 sub-words per row
    localparam ACC_ROW_AW = ROW_IDX_W + ACC_SUB_W;   // 10-bit SRAM address {row, sel}
    localparam ACC_BUS_W  = R_CORES*N_CH_PER*`ST_ACC_WIDTH >> ACC_SUB_W;   // 4096
    localparam JP_W      = 8;   // j_pass tag width (row = j_pass*B_batches + abs_batch, so
                                // j_pass <= 128 whenever the capacity check passes)
    localparam FIRST_W   = 1;   // "first k_group" flag: overwrite vs accumulate
    localparam N_BATCH    = 16; // register-block depth (matches core/ctrl/addr_gen/acc_block_reg)

    // ── sub_sum FIFO backpressure level (bug 7) ────────────────────────────────
    // The ctrl may start a new group only while the FIFO holds at most
    // FIFO_BP_LEVEL entries.  `fifo_full` is far too late: the acc_block_reg /
    // LUT pipeline holds ST_RES_DELAY = 10 sub_sum pulses that keep arriving
    // AFTER the group boundary, and one group is a 16-entry burst, so a stop
    // issued at `full` still loses the tail.
    //
    // Numbers measured in sim/tb_fifo_rate.v (`make sim-fifo-rate`, seconds;
    // N_CH-scaled — every rate here is a cycle count, invariant in N_CH):
    //   ternary group = 64 cyc, producer 16 entries/group, consumer 5 cyc/entry
    //   -> the ternary producer is structurally FASTER than its consumer
    //      (16*5 = 80 cyc of consumer work per 64-cyc group), W4 (128 cyc) has
    //      ~2x surplus.  Stopping at the boundary therefore costs ternary ~25%
    //      throughput; that is the price of not dropping data.
    //   peak occupancy with level 16 was 27/32 (5 slots), with 12 it is 23/32.
    //   level 4 costs ~19% more (period 94.7 vs 80.1 cyc), so 12 is the knee.
    // Scheme C halved the consumer to 3 cycles/entry, which removed the producer's
    // stall at the FIFO entirely: sim/tb_fifo_rate.v now measures a peak occupancy of
    // **11 of 32**.  The depth was sized for the 5-cycle consumer's rate mismatch; what
    // sizes it now is only the in-flight tail (ST_RES_DELAY = 10 entries keep arriving
    // after the producer is stopped), so DEPTH = 16 and BP = 6 leaves peak <= 16 with
    // margin and saves ~1.28 mm^2 of flops (task #116: 12,297 b x 32).
    //   safety: BP + ST_RES_DELAY = 6 + 10 = 16 = DEPTH (zero drop), 4 entries spare
    //   against the measured peak of 11.
    // DEPTH is a plain register-FIFO depth (sub_sum_fifo is DFF-based, not a macro), so
    // it does NOT have to be a power of two -- only BP + ST_RES_DELAY <= DEPTH matters.
    // BP must stay at 12: that is the measured knee (see the note above: "level 4 costs
    // ~19% more, period 94.7 vs 80.1 cyc"), and lowering it to 6 made the producer stop
    // earlier and re-introduced stall on the single-batch last sub-pass (W4 dim_I=33/65:
    // 20 backpressure cycles against "W4 must never stall").  So the depth is what gives,
    // not the threshold: 12 + 10 = 22 <= 24, two entries of margin.
    // DEPTH need not be a power of two -- but only because sub_sum_fifo's pointers now
    // wrap at DEPTH-1 instead of at their own width.  They did NOT before, and DEPTH=24
    // then aliased writes into out-of-range slots: sim-st-top-w4 read got=x out of acc
    // rows that were never written.  That was the real cause of the "unexplained X"; the
    // 32-entry build hid it because at a power of two the two wraps coincide.
    localparam FIFO_DEPTH    = 24;
    localparam FIFO_BP_LEVEL = 12;   // the knee; = DEPTH - ST_RES_DELAY - 2

    // ── Configuration check: does the (dim_I, dim_J) tile set fit the acc row space? ──
    //
    // The acc row index carries the j_pass dimension:
    //     row = j_pass * B_batches + abs_batch,  B_batches = ceil(dim_I/R_CORES)
    // so one run needs B_batches * J_passes rows out of the MAX_ROWS the array has
    // (J_passes = ceil(dim_J/N_CH_PER_CORE)).  Anything larger used to truncate the
    // index (fifo_row = row_idx[6:0]) and alias the j_pass blocks on top of each other
    // with no warning at all.
    //
    // The counts are derived here rather than read from the ctrl/addr_gen, because they
    // are needed at `start` time (before the ctrl latches them) and because they must
    // keep working when the ctrl is held in IDLE.  The logic is its own module (below,
    // same file) so sim/tb_tiling_contract.v can exercise the SHIPPING arithmetic
    // instead of a testbench re-implementation — same reason as wgt_fill_handshake.
    wire [15:0] b_batches_w, j_passes_w, k_groups_w;
    wire [31:0] rows_needed;
    wire        cfg_overflow;

    cfg_row_check #(
        .R_CORES(R_CORES), .N_CH_PER(N_CH_PER),
        .MAX_ROWS(MAX_ROWS), .GROUP_K(`ST_GROUP_K)
    ) u_cfg_check (
        .dim_I(dim_I), .dim_J(dim_J), .dim_K(dim_K),
        .b_batches(b_batches_w), .j_passes(j_passes_w), .k_groups(k_groups_w),
        .rows_needed(rows_needed), .overflow(cfg_overflow)
    );

    reg cfg_error_r;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)     cfg_error_r <= 1'b0;
        else if (start) cfg_error_r <= cfg_overflow;
    end
    assign cfg_error = cfg_error_r;

    // An out-of-range run is refused: ctrl never leaves IDLE, so `done` never pulses and
    // the host cannot mistake truncated results for a completed run.
    wire run_start = start && !cfg_overflow;

    // ── Weight path sizing ──
    localparam WT_CH_TILE   = `ST_WT_WORD;                            // 64 (invariant, 16x4b / 32x2b)
    localparam WT_TILE_BITS = N_CH_PER * WT_CH_TILE;                  // 16384 (16 Kbit)
    // ASAP7 1RW: the weight SRAM has ONE port shared by the cache refill (read) and the
    // DMA fill (write).  Refill needs 8 words/tile over a 16-cycle tile (50%) and the
    // DMA needs 8 words/group over a 128-cycle group (50%) => 100%, zero margin.
    // Widening the read word from 1024 b to 2048 b is what buys the DMA its half.
    localparam SRAM_W       = 2048;                                    // cache refill read width (was 1024)
    localparam SRAM_PACK    = SRAM_W / 512;                           // 4 DMA beats per SRAM word (was 2)
    localparam WT_SRAM_DEPTH = WT_TILE_BITS * `ST_TILES_PER_GROUP_MAX / SRAM_W; // 64 words
    localparam CACHE_CHUNKS  = WT_TILE_BITS / SRAM_W;                  // 8 chunks/tile (was 16)
    localparam WT_ADDR_W     = $clog2(WT_SRAM_DEPTH);                  // 6
    localparam WT_FILL_CNT_W = $clog2(WT_SRAM_DEPTH * SRAM_PACK);      // 8 DMA beats
    // ── Block ping-pong (2026-09-27, plan 2026-09-27-gate-full-load.md 台阶 1) ──
    // The macro array is 256 words deep and a weight block only uses WT_SRAM_DEPTH (64)
    // of them, so the top address bit selects between TWO resident weight blocks.  The
    // DMA fills the block the cache is NOT reading, which is what lets the fill of the
    // next (j_pass, k_group) block overlap the compute of the current one.  See
    // weight_tile_cache.v for the buffer banks that go with it.
    localparam WT_ADDR_AW    = WT_ADDR_W + 1;                          // 7 (2 blocks)

    // Chip-select the weight_tile_cache computes for its macros (held low); declared
    // here because the weight macros below consume it.
    wire wt_cache_csb;

    wire fifo_full, fifo_empty, fifo_wr_en, fifo_rd_en, fifo_wr_drop;
    wire [$clog2(FIFO_DEPTH):0] fifo_level;
    // ctrl backpressure (early).  The comparison lives in its own module (below, same
    // file) so a seconds-level harness can instantiate the SHIPPING expression instead
    // of re-modelling it — same reason as wgt_fill_handshake / cfg_row_check / drain_fsm.
    // Mutation entry 7 (`fifo_bp` tied 0) was blind precisely because the only harness
    // that "checked" it had its own copy of the comparator (test-flow §2.4(c)).
    wire fifo_bp;
    fifo_bp_cmp #(.LEVEL(FIFO_BP_LEVEL), .W($clog2(FIFO_DEPTH)+1)) u_fifo_bp (
        .level(fifo_level),
        .bp(fifo_bp)
    );
    wire [FIRST_W + ROW_IDX_W + R_CORES*N_CH_PER*`ST_SUBSUM_WIDTH-1:0] fifo_wr_data, fifo_rd_data;
    wire [15:0] dummy_chain;
    wire [N_CH_PER*`ST_SUBSUM_WIDTH-1:0] core_sub_sum [0:R_CORES-1];
    wire [3:0]                           core_sub_sum_batch [0:R_CORES-1];
    wire [0:R_CORES-1]                   core_sub_sum_valid;

    // addr_gen outputs (declared here; used by the weight-path tag/tile-boundary logic above)
    wire [2:0]  wt_tile_idx;
    wire [2:0]  act_tile_idx;
    wire [15:0] addr_batch, addr_batch_prefetch, addr_sub_pass, addr_j_pass, addr_k_group;
    wire [4:0]  addr_b_sub;      // live batch count of the current sub-pass (may be short)

    // ── Controller ──
    wire [3:0]  dbg_state_internal;
    wire        ctrl_build_en, ctrl_compute_en;
    wire        ctrl_tile_start, ctrl_group_boundary, ctrl_group_done;
    wire        ctrl_done;
    wire [9:0]  ctrl_wt_addr;
    wire [10:0] ctrl_act_addr;
    wire [8:0]  ctrl_acc_addr;
    wire        ctrl_dma_wt_start, ctrl_dma_act_start;
    wire        dma_wt_done, dma_act_done;
    wire        buf_sel, fill_swap, fill_done;   // weight buffer ping-pong (vestigial in tile-major)
    wire        act_prefetch;                    // tied 0 (Task 5 activation transpose)

    sparse_ternary_ctrl #(.R(R_CORES), .T_PER(T_PER), .N_CH_PER_CORE(N_CH_PER)) u_ctrl (
        .clk(clk), .rst_n(rst_n),
        .start(run_start), .done(ctrl_done),
        .dim_I(dim_I), .dim_J(dim_J), .dim_K(dim_K),
        .fifo_bp(fifo_bp),          // NOT fifo_full — see FIFO_BP_LEVEL above
        .wt_mode(wt_mode),
        .build_en(ctrl_build_en), .compute_en(ctrl_compute_en),
        .tile_start(ctrl_tile_start), .group_boundary(ctrl_group_boundary),
        .group_done(ctrl_group_done),
        .wt_sram_addr(ctrl_wt_addr), .act_sram_addr(ctrl_act_addr),
        .acc_sram_addr(ctrl_acc_addr),
        .dma_wt_start(ctrl_dma_wt_start), .dma_act_start(ctrl_dma_act_start),
        .dma_wt_done(dma_wt_done), .dma_act_done(dma_act_done),
        .buf_sel(buf_sel), .fill_swap(fill_swap), .fill_done(fill_done),
        .act_prefetch(act_prefetch),
        .dbg_state(dbg_state_internal)
    );
    // Chain dummy signals into debug output to prevent ABC optimization
    assign dbg_state = dbg_state_internal ^ dummy_chain[3:0];

    // ── Weight/activation block DMA trigger (see the port comments) ──
    // REGISTERED, not passed straight through.  The ctrl's version is a combinational
    // 1-cycle pulse, and it changes in the same time step as the event that causes it
    // (e.g. `dma_wt_start` rises the instant `dma_act_done` is seen and falls the instant
    // `state` becomes S_LOAD_WGT).  A host — or a testbench — that samples the pin on the
    // opposite clock edge then races with that transition and MISSES the pulse (measured:
    // sim/tb_st_top_ternary.v asked for the next weight block and never got the trigger,
    // so the run hung in S_LOAD_WGT).  One register stage gives a pin that is stable for
    // a full cycle, which is what a DMA handshake output should be.
    reg dma_wt_start_r, dma_act_start_r;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            dma_wt_start_r  <= 1'b0;
            dma_act_start_r <= 1'b0;
        end else begin
            dma_wt_start_r  <= ctrl_dma_wt_start;
            dma_act_start_r <= ctrl_dma_act_start;
        end
    end
    assign dma_wt_start  = dma_wt_start_r;
    assign dma_act_start = dma_act_start_r;

    // ── sub_sum FIFO: decouples LUT engine (producer) from acc_rmw (consumer) ──
    // Tile-major: one entry per batch sub_sum (N_BATCH entries stream out per group),
    // so the depth must hold a full N_BATCH burst (16) with margin for the slow
    // 9-cycle/entry acc_rmw consumer.
    localparam SUBSUM_BUS_W = N_CH_PER * `ST_SUBSUM_WIDTH;   // 256 * 24 = 6144 bits/core
    localparam FIFO_W       = FIRST_W + ROW_IDX_W + R_CORES * SUBSUM_BUS_W;

    sub_sum_fifo #(.WIDTH(FIFO_W), .DEPTH(FIFO_DEPTH)) u_sub_sum_fifo (
        .clk(clk), .rst_n(rst_n),
        .wr_en(fifo_wr_en), .wr_data(fifo_wr_data),
        .rd_en(fifo_rd_en), .rd_data(fifo_rd_data),
        .full(fifo_full),   .empty(fifo_empty),
        .level(fifo_level)
    );

    // ── Tag alignment: sub_pass + first(k_group==0) for the completed group ──
    // acc_block_reg streams N_BATCH sub_sum_valid pulses (one per batch) for each
    // group.  Those pulses are the compute-pipeline output, so they arrive
    // ST_RES_DELAY+2 cycles after the controller's build schedule: the burst for a
    // group therefore STRADDLES its group boundary — the first ~3 batches fire
    // BEFORE the boundary and the last batch ~12 cycles AFTER.
    //
    // Capturing the tag AT the boundary (addr_gen's pre-advance sub_pass) gives the
    // right VALUE but the wrong TIME: the tag flips to the next sub-pass too late,
    // so the first few batches of every sub-pass read the previous sub-pass's tag
    // and write to the wrong accumulator rows (rows 0..3 instead of 16..19, etc.).
    //
    // Fix: shift the raw addr_gen counters by the compute-pipeline latency so the
    // tag tracks the sub_sum_valid stream exactly (the counters are constant for the
    // whole sub-pass, so any delay in [latency, sub_pass_len-3] aligns correctly).
    localparam TAG_DELAY = `ST_RES_DELAY + 3;   // 13 (compute pipeline + acc_block_reg 2-stage pipeline + done_r/sub_sum_valid)

    reg [3:0]  tag_sp_sr [0:TAG_DELAY-1];
    reg [JP_W-1:0] tag_jp_sr [0:TAG_DELAY-1];
    reg        tag_first_sr [0:TAG_DELAY-1];
    integer ti;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (ti = 0; ti < TAG_DELAY; ti = ti + 1) begin
                tag_sp_sr[ti]    <= 4'd0;
                tag_jp_sr[ti]    <= {JP_W{1'b0}};
                tag_first_sr[ti] <= 1'b0;
            end
        end else begin
            tag_sp_sr[0]    <= addr_sub_pass[3:0];
            tag_jp_sr[0]    <= addr_j_pass[JP_W-1:0];
            tag_first_sr[0] <= (addr_k_group == 16'd0);
            for (ti = 1; ti < TAG_DELAY; ti = ti + 1) begin
                tag_sp_sr[ti]    <= tag_sp_sr[ti-1];
                tag_jp_sr[ti]    <= tag_jp_sr[ti-1];
                tag_first_sr[ti] <= tag_first_sr[ti-1];
            end
        end
    end
    wire [15:0] tag_sub_pass = {12'd0, tag_sp_sr[TAG_DELAY-1]};
    wire [JP_W-1:0] tag_j_pass = tag_jp_sr[TAG_DELAY-1];
    wire        tag_first    = tag_first_sr[TAG_DELAY-1];

    // ── FIFO write: one entry per sub_sum_valid pulse ──
    //   row = absolute batch of ONE j_pass = sub_pass * N_BATCH + sub_sum_batch
    //         (2 cores = 2 rows/batch)  ... plus the j_pass base:
    //
    //   row' = j_pass * B_batches + abs_batch      (B_batches = ceil(dim_I/R_CORES))
    //
    // Without the j_pass term every pass wrote rows 0..B_batches-1 and pass j overwrote
    // pass j-1 (measured: 16/16 rows colliding for dim_J=512, 48 collisions for
    // dim_J=512+dim_K=256 — see sim/tb_tiling_contract.v).  The total row count
    // B_batches * J_passes is checked against the 128 physical rows at `start`
    // (cfg_overflow / cfg_error above), so `row_idx_w` always fits ROW_IDX_W.
    // `b_batches_w` (not the registered B_batches_r) is used here on purpose: it is the
    // same value (B_batches_r is loaded from it at run_start) and it keeps this
    // continuous assign independent of a register declared further down the file.
    // The encoding itself is `acc_row_encode` (module below, same file) so that
    // sim/tb_tiling_contract.v can instantiate the SHIPPING arithmetic.  Mutation entry
    // 2 (drop the j_pass term) was blind while the harness carried a copy of this
    // expression: the copy kept working no matter what this file said (§2.4(c)).
    wire [15:0] row_idx_w;
    wire [ROW_IDX_W-1:0] fifo_row;
    acc_row_encode #(.ROW_IDX_W(ROW_IDX_W), .N_BATCH(N_BATCH)) u_row_encode (
        .sub_pass(tag_sub_pass[3:0]),
        .batch   (core_sub_sum_batch[0]),
        .j_pass  (tag_j_pass),
        .b_batches(b_batches_w),
        .row_full(row_idx_w),
        .row     (fifo_row)
    );

    // The `&& !fifo_full` here is now a SAFETY NET, not the flow-control mechanism:
    // FIFO_BP_LEVEL stops the producer ~20 entries before this can ever be reached
    // (peak occupancy measured = 23/32).  Any write this gate ever refuses is
    // counted in fifo_overflow below, which is sticky and must stay 0 — bug 7 was
    // exactly this gate dropping 10.5% of the entries with nothing to show for it.
    assign fifo_wr_en    = core_sub_sum_valid[0] && !fifo_full;
    assign fifo_wr_drop  = core_sub_sum_valid[0] &&  fifo_full;    // = an entry LOST
    assign fifo_wr_data  = {tag_first, fifo_row, core_sub_sum[1], core_sub_sum[0]};

    // Sticky overflow flag: set if a sub_sum entry was ever refused by the write
    // gate.  Cleared only by `start`.  Exposed on the port list so a long-run test
    // can assert it (and so `all_written` below can never be satisfied by a run
    // that dropped entries).
    reg fifo_overflow_r;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)          fifo_overflow_r <= 1'b0;
        else if (start)      fifo_overflow_r <= 1'b0;
        else if (fifo_wr_drop) fifo_overflow_r <= 1'b1;
    end
    assign fifo_overflow = fifo_overflow_r;

    // LUT-done latch: set when the controller finishes all groups.
    reg lut_done_r;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)         lut_done_r <= 1'b0;
        else if (ctrl_done) lut_done_r <= 1'b1;
        else if (start)     lut_done_r <= 1'b0;
    end

    // ── DMA Readers ──
    assign s_axis_wgt_tready = 1'b1;
    assign dma_wt_done  = s_axis_wgt_tlast;

    wire dma_act_wr_en;
    wire [`ST_ACT_SRAM_WIDTH-1:0] dma_act_data;
    // AXI: a beat is consumed only when the port can take it.  `s_axis_act_tready` is
    // driven down the cycle `act_refill_rd` needs the port (see act_port_mux.v), so the
    // DMA's write address advances on ACCEPTED beats only.
    assign dma_act_wr_en = s_axis_act_tvalid && s_axis_act_tready;
    assign dma_act_data  = s_axis_act_tdata;
    // ...and the same rule for `done`: the block is finished when its LAST BEAT IS
    // ACCEPTED, not when the source happens to be asserting tlast.  Plan (b) made the act
    // port arbitrated, so a compliant source HOLDS tvalid/tdata/tlast until tready rises
    // -- and `dma_act_done` was still the raw `s_axis_act_tlast`, i.e. it stayed high for
    // the whole hold.  The ctrl survives that (it counts `act_done_rise` edges), but the
    // region bit does not: `act_wr_region <= ~act_wr_region` on `dma_act_done` toggles
    // ONCE PER CYCLE of the hold, so a refused last beat moves the next activation block
    // into the region the core is reading.  Qualifying with `tvalid && tready` makes
    // `dma_act_done` exactly one cycle wide again for every source, which is the shape
    // every consumer of it was written for (the weight side needs no such fix: its
    // tready is tied 1, so its tlast is already a one-cycle pulse).
    assign dma_act_done  = s_axis_act_tvalid && s_axis_act_tready && s_axis_act_tlast;

    // ── Weight path: DMA(512b) -> sram_wgt_bank (2048 b word) -> weight_tile_cache ──
    // ONE port shared by the refill read and the DMA fill write; the DMA wins when a
    // 2048 b word has been assembled, otherwise the cache refills.  The macros are
    // full-word write (no byte enable), so 4 x 512 b beats are staged into a 2048 b
    // register before one write.
    reg [WT_FILL_CNT_W-1:0] wgt_fill_cnt;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)                 wgt_fill_cnt <= {WT_FILL_CNT_W{1'b0}};
        else if (start)             wgt_fill_cnt <= {WT_FILL_CNT_W{1'b0}};
        else if (s_axis_wgt_tvalid) wgt_fill_cnt <= s_axis_wgt_tlast ? {WT_FILL_CNT_W{1'b0}} : wgt_fill_cnt + 1'b1;
    end
    wire [WT_ADDR_W-1:0] wgt_fill_word = wgt_fill_cnt[WT_FILL_CNT_W-1:2]; // 0..WT_SRAM_DEPTH-1
    wire [1:0]           wgt_fill_slot = wgt_fill_cnt[1:0];               // 0..3 within the 2048b word

    // DMA fill complete: 1-cycle pulse after the last weight beat (replaces wgt_fill_last).
    reg wgt_fill_done;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) wgt_fill_done <= 1'b0;
        else        wgt_fill_done <= s_axis_wgt_tvalid && s_axis_wgt_tlast;
    end

    // 2048 b staging register + write request.  The write address is registered because
    // wgt_fill_cnt has already advanced past the assembled word by the time we write.
    reg  [SRAM_W-1:0]    wgt_fill_stage;
    reg                  wgt_wr_req;
    reg  [WT_ADDR_W-1:0] wgt_wr_addr_r;

    // ── block ping-pong: which SRAM region the DMA fills ──
    // `wr_sel` advances one cycle AFTER the weight tlast (wgt_fill_done is the registered
    // tlast).  That one cycle matters: the last 2048 b word of the block is written from
    // `wgt_wr_addr_r` on the cycle wgt_fill_done is high, so wr_sel must still name the
    // block that word belongs to at that moment (sim/tb_wtc_refill.v checks the region
    // tags).  `blk` handed to the cache is the SAME register for the same reason: at the
    // cycle the cache latches `start` it must sample the region the block just landed in,
    // i.e. the value before this edge's toggle.
    // Declared BEFORE the address assign below: iverilog binds continuous assigns in
    // source order.
    reg wr_sel;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)                 wr_sel <= 1'b0;
        else if (run_start)         wr_sel <= 1'b0;
        else if (wgt_fill_done)     wr_sel <= ~wr_sel;
    end

    // ── port arbiter: DMA fill has priority, refill stalls for that cycle ──
    wire                 wt_do_write   = wgt_wr_req;
    wire                 wt_cache_rdy = ~wt_do_write;
    wire [SRAM_W-1:0]    wt_sram_dout;
    wire [WT_ADDR_AW-1:0] wt_cache_sram_addr;
    wire [WT_ADDR_AW-1:0] wt_sram_addr = wt_do_write ? {wr_sel, wgt_wr_addr_r}
                                                     : wt_cache_sram_addr;


    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wgt_fill_stage <= {SRAM_W{1'b0}};
            wgt_wr_req     <= 1'b0;
            wgt_wr_addr_r  <= {WT_ADDR_W{1'b0}};
        end else begin
            // Clear FIRST, then let a newly completed word re-assert: the AXI-S weight
            // stream is usually tvalid every cycle, so an `else if (wt_do_write)` form
            // would never clear and the write would hog the 1RW port forever (cache
            // starvation / deadlock).
            if (wt_do_write) wgt_wr_req <= 1'b0;
            if (s_axis_wgt_tvalid) begin
                wgt_fill_stage[wgt_fill_slot*512 +: 512] <= s_axis_wgt_tdata;
                if (wgt_fill_slot == 2'd3) begin
                    wgt_wr_req    <= 1'b1;
                    wgt_wr_addr_r <= wgt_fill_word;
                end
            end
        end
    end

    // ── Weight bank ──
    // Decision B: the library-specific macro assembly lives in rtl/sram_7nm/ (ASAP7)
    // and rtl/sram_45nm/ (OpenRAM), same module name and ports.  This top names no macro.
    sram_wgt_bank u_wgt (
        .clk (clk),
        .csb (wt_cache_csb),      // held low -> selected every cycle
        .web (~wt_do_write),
        .addr(wt_sram_addr),      // 128 words = 2 blocks x 64
        .din (wgt_fill_stage),
        .dout(wt_sram_dout)
    );

    // tile_boundary: the weight tile advances every N_BATCH (batch wrap).  Driven by the
    // addr_gen wt_tile_idx change edge, delayed 2 to align with the 2-cycle activation
    // read latency (the cache broadcast is thus the weight tile 2 presentations behind).
    reg [2:0] wt_tile_idx_prev;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) wt_tile_idx_prev <= 0;
        else        wt_tile_idx_prev <= wt_tile_idx;
    end
    wire wt_tile_change = (wt_tile_idx != wt_tile_idx_prev);
    // The weight broadcast must lag the activation by exactly one tile presentation.
    // The activation path has a 2-cycle delay (SRAM latency + explicit act_tile_idx
    // delay); the weight path already carries 2 inherent cycles (addr_gen tile_d = 1,
    // cache active-flip = 1), so the tile_boundary needs only a 1-cycle delay to land
    // weight tile (n-3) alongside act tile (n-2).
    reg tb_d;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) tb_d <= 1'b0;
        else        tb_d <= wt_tile_change;
    end
    wire cache_tile_boundary = tb_d;

    wire [WT_TILE_BITS-1:0] wt_bcast;
    wire cache_ready;

    // ── preload_all: the incoming block's FIRST sub-pass is short (#143) ─────────────
    // The weight cache walks the block's tiles once and then stops, so after the first
    // pass through them the refill is off the critical path.  The first pass is on it:
    // tile T must be resident by T*B_sub presentations.  In the greedy sub-pass blocking
    // the first sub-pass is a full N_BATCH=16 unless S_passes == 1, which is exactly
    // B_batches <= 16 -- and for B_batches < 8 the first (and only) sub-pass is short
    // enough that the refill cannot serve it.  Then, and only then, make `ready` wait for
    // the WHOLE block instead of the first three tiles (the cost is the block's remaining
    // tiles, 8 cycles each, and only for shapes smaller than 16 activation rows).
    // Latched at run_start so it is stable for the whole run.
    reg preload_all_r;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)          preload_all_r <= 1'b0;
        else if (run_start)  preload_all_r <= (((dim_I + R_CORES - 1) / R_CORES) < 8);
    end

    weight_tile_cache #(
        .N_CH(N_CH_PER), .W_CH_TILE(WT_CH_TILE), .TILE_W(WT_TILE_BITS),
        .SRAM_W(SRAM_W), .N_TILES(`ST_TILES_PER_GROUP_MAX), .CHUNKS(CACHE_CHUNKS)
    ) u_wt_cache (
        .clk(clk), .rst_n(rst_n), .wt_mode(wt_mode),
        .sram_dout(wt_sram_dout),
        .sram_addr(wt_cache_sram_addr),
        .sram_csb(wt_cache_csb),            // drives the weight macros' csb0 (held low)
        .sram_rdy(wt_cache_rdy),            // ASAP7 1RW: refill stalls when the DMA fill wins
        .start(wgt_fill_done),              // pulse: a block's SRAM fill completed
        .blk(wr_sel),                       // ...and it landed in this region
        .blk_switch(fill_swap),             // ctrl consumed it -> the block becomes current
        .tile_boundary(cache_tile_boundary),
        .preload_all(preload_all_r),
        .ready(cache_ready),
        .wt_out(wt_bcast)
    );

    // ── fill_done -> ctrl: the weight block the ctrl is WAITING FOR has landed ──
    //
    // The ctrl consumes the handshake by leaving S_LOAD_WGT (that is what its
    // `fill_swap` output is) and re-enters S_LOAD_WGT at every j_pass / k_group
    // boundary.  weight_tile_cache.v:30-31 states the contract explicitly:
    //   "The SRAM content is reloaded externally by the DMA at each j_pass boundary
    //    (the `start` pulse re-arms the fill)."
    // The cache always refills from SRAM words 0..N_TILES*CHUNKS-1 (sram_addr =
    // (refill_tile mod N_TILES)*CHUNKS + cnt), so a new (k_group, j_pass) block MUST be
    // DMA'd in before the cache re-reads it.
    //
    //   OLD: assign fill_done = cache_ready;   -- a constant 1 after the first fill.
    //        Every later S_LOAD_WGT then fell through after act_settle alone (3 cycles),
    //        whether or not the new block had arrived.  Measured: harness case 2/5,
    //        `dwell per visit = 170 3`.
    //
    // A completed/consumed counter pair makes fill_done mean "a block has arrived and
    // has not been consumed yet".  That is order-preserving and, unlike a latch cleared
    // by fill_swap, still correct if the DMA prefetched the block before the ctrl asked
    // for it.  `cache_ready` still gates it, so the cache init double-fill
    // (wgt_fill_done -> cache .start) remains a prerequisite.
    //
    // Single-block runs (J_passes == 1 && K_groups == 1) are bit-for-bit unchanged: the
    // first S_LOAD_WGT wait is identical, and no further S_LOAD_WGT visit exists
    // (verified cycle-for-cycle by sim/tb_tiling_contract.v case 0 vs case 1).
    //
    // The logic lives in its own module (below, same file — deliberately NOT a new file,
    // so none of the ~10 file lists that already name sparse_ternary_top.v need touching)
    // so that sim/tb_tiling_contract.v can instantiate the SHIPPING RTL rather than a
    // testbench re-implementation.
    wgt_fill_handshake u_wgt_fill_hs (
        .clk(clk), .rst_n(rst_n),
        .run_start(start),          // run-start pulse
        .blk_done(wgt_fill_done),   // weight DMA block completed (tlast)
        .blk_consume(fill_swap),    // ctrl consumed the handshake (left S_LOAD_WGT)
        .cache_ready(cache_ready),  // weight_tile_cache init double-fill complete
        .fill_done(fill_done)
    );

    // ── INT8 weight block snapshot ──
    // Contract: the INT8 block is the LOW 8192 b of the broadcast tile, 32 columns x
    // 256 b, column n = wt_bcast[256n +: 256], byte k of column n = W[k][n].  Captured
    // once per run on the first rising edge of cache_ready after `start` (start re-inits
    // the cache, so ready goes low first) and then held, so a consumer has a stable
    // 8 Kbit copy to serialise from.
    reg [8191:0] wt_i8_snap_r;
    reg [4:0]    wt_i8_col_r;
    reg          wt_i8_hold_r;
    reg          wt_i8_valid_r, cache_ready_d, started_r;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wt_i8_snap_r  <= 8192'd0;
            wt_i8_col_r   <= 5'd0;
            wt_i8_hold_r  <= 1'b0;
            wt_i8_valid_r <= 1'b0;
            cache_ready_d <= 1'b0;
            started_r     <= 1'b0;
        end else begin
            cache_ready_d <= cache_ready;
            if (start) begin
                started_r     <= 1'b1;
                wt_i8_valid_r <= 1'b0;
                wt_i8_col_r   <= 5'd0;
                wt_i8_hold_r  <= 1'b0;
            end
            if (started_r && cache_ready && !cache_ready_d) begin
                wt_i8_snap_r  <= wt_bcast[8191:0];
                wt_i8_valid_r <= 1'b1;
                wt_i8_col_r   <= 5'd0;
                // hold column 0 one extra cycle: the consumer registers wt_i8_valid
                // before it starts counting, so its column 0 is one cycle later.
                wt_i8_hold_r  <= 1'b1;
            end else if (wt_i8_valid_r) begin
                if (wt_i8_hold_r)              wt_i8_hold_r <= 1'b0;
                else if (wt_i8_col_r != 5'd31) wt_i8_col_r  <= wt_i8_col_r + 5'd1;
            end
        end
    end

    assign wt_i8_col     = wt_i8_snap_r[wt_i8_col_r*256 +: 256];
    assign wt_i8_col_idx = wt_i8_col_r;
    assign wt_i8_valid   = wt_i8_valid_r;

    // ── Address generator (mirrors ctrl tiling loop) ──
    addr_gen #(
        .R_CORES(R_CORES), .N_CH_PER_CORE(N_CH_PER), .T_PER(T_PER),
        .GROUP_K(`ST_GROUP_K), .TILES_PER_GROUP(`ST_TILES_PER_GROUP)
    ) u_addr_gen (
        .clk(clk), .rst_n(rst_n),
        .dim_I(dim_I), .dim_J(dim_J), .dim_K(dim_K),
        .ctrl_tile_start(ctrl_tile_start),
        .ctrl_group_done(ctrl_group_boundary),
        .ctrl_build_en(ctrl_build_en),
        .ctrl_compute_en(ctrl_compute_en),
        .ctrl_dma_wt_start(start),   // run-start (load dims / reset counters)
        .act_prefetch(act_prefetch),
        .wt_mode(wt_mode),
        .wt_tile_idx(wt_tile_idx),
        .act_tile_idx(act_tile_idx),
        .batch(addr_batch), .batch_prefetch(addr_batch_prefetch),
        .sub_pass(addr_sub_pass), .j_pass(addr_j_pass), .k_group(addr_k_group),
        // The last sub-pass of a block can be short (dim_I not a multiple of 32):
        // acc_block_reg needs its batch count to wrap its tile counter there.
        .b_sub(addr_b_sub)
    );

    // ── b_sub: the batch count of the group the ENGINE is presenting right now ──────
    // acc_block_reg has to know how many batches this sub-pass holds so it can wrap its
    // tile counter after B_sub (not after the hard-coded N_BATCH=16) -- the last
    // sub-pass of a block is short whenever dim_I is not a multiple of 2*N_BATCH.
    //
    // It consumes that count at the group's OWN last presentation, which lands ~10
    // cycles after the ctrl left S_GROUP (the LUT pipeline).  Reading `addr_sub_pass`
    // raw there is wrong: by then sub_pass has already been reset for the next block, so
    // the live value has wrapped back to the FIRST sub-pass's count.  Measured on
    // dim_I=33 (B_batches=17 -> B_sub = 16 then 1): the last group read b_sub=16, never
    // wrapped, and produced no sub_sum at all.
    //
    // So hold the count the engine is actually streaming: it tracks while the core is
    // being fed (ctrl_build_en) and freezes over the S_LOAD_ACT/S_LOAD_WGT gaps.  The
    // stall case is harmless -- build_en is low while the ctrl holds the last
    // presentation, and sub_pass does not change under a stall either.
    reg [4:0] b_sub_held;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)            b_sub_held <= 5'd16;
        else if (ctrl_build_en) b_sub_held <= addr_b_sub;
    end

    // ── Activation SRAM: 2 banks (bank0 = even rows, bank1 = odd rows) x 4 slices ──
    // Each bank = 512b = 128 act x 4b. Load: row r -> bank r%2 at addr r/2.
    // Read (batch b): bank0 addr b -> row 2b (core0), bank1 addr b -> row 2b+1 (core1).
    // Tile-major: the read address tracks the ABSOLUTE batch (sub_pass*N_BATCH + batch);
    // the 2-cycle SRAM read latency is compensated by the 2-cycle delay on the other
    // core inputs (see the 2-cycle alignment registers below).
    localparam ACT_BANKS = 2;
    // Each bank is one sram_act_bank instance = 8 x M256 macros = 512 b.  M256 is enough
    // because the acc row contract caps dim_I at 256, so the act address is <= 128.
    // (It used to be M1024x64 -- 8x deeper than any legal configuration could reach.)
    localparam ACT_SLICES = 8;
    wire [511:0] act_dout [0:ACT_BANKS-1];

    reg [10:0] act_load_cnt;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) act_load_cnt <= 11'd0;
        else if (ctrl_dma_act_start) act_load_cnt <= 11'd0;
        else if (dma_act_wr_en) act_load_cnt <= act_load_cnt + 1'b1;
    end

    wire [15:0] abs_batch = addr_sub_pass * N_BATCH + addr_batch;

    // ── Region split: the core reads one half of the macro depth, the DMA fills the other
    //
    // The cost of the old design was not bandwidth, it was PHASE: the DMA owned the port
    // for the whole load window, so `load_act` = K_groups*dim_I = 7,040 cycles/block ran
    // serial with the compute and stretched to 11,841 at the measured DRAM rate
    // (2026-09-28-gate-dram-recheck.md §4).  Plan (b) overlaps it, which requires the
    // block being loaded to sit at addresses the core is NOT reading.
    //
    // A macro is 256 words deep and a block needs ceil(dim_I/2) rows, so when that is
    // <= 128 the top address bit is free to name the region.  `act_wr_region` advances
    // on dma_act_done (the DMA has finished a block and moves to the next), and
    // `act_rd_region` follows k_group, so with the request issued one block ahead the
    // pair is always opposite — the write can never land on a row being read, which is
    // the corruption path the old unconditional `dma_act_wr_en` opened (the beat used
    // to be written to `abs_batch`, the compute read address, whenever it arrived
    // outside the load window).
    //
    // A geometry that does not fit two regions (ceil(dim_I/2) > 128, reachable at
    // dim_I > 256 with J_passes == 1) keeps the serial load: the ctrl derives the same
    // condition from its own B_batches and does not prefetch there, so the DMA only
    // writes while the FSM sits in S_LOAD_ACT and `act_overlap_en` = 0 drops the region
    // bit from both addresses.
    reg act_overlap_en_r;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)         act_overlap_en_r <= 1'b0;
        else if (run_start) act_overlap_en_r <= (b_batches_w <= 16'd128);
    end

    reg act_wr_region;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)              act_wr_region <= 1'b0;
        else if (run_start)      act_wr_region <= 1'b0;   // block 0 lands in region 0
        else if (dma_act_done)   act_wr_region <= ~act_wr_region;
    end

    // The core reads region = k_group parity, i.e. it follows the ctrl's own tiling
    // counter (addr_gen's), which advances on the group boundary at the same edge the
    // read address ramp wraps.  Block 0 is loaded into region 0 while k_group == 0.
    wire act_rd_region = addr_k_group[0];

    // ── When does the core need the SRAM (rather than the line buffer)? ────────────
    // The refill is a WRITE-THROUGH: the SRAM read at cycle S+j is what fills lb[j], and
    // the buffer entry is writable one cycle after that read's data appears, i.e. valid
    // from cycle S+j+2.  A later tile t presents batch j at S + t*B_sub + j and reads the
    // buffer two cycles earlier (the synchronous read + the act_row register), so the
    // buffer can serve it iff
    //     S + t*B_sub + j - 2  >=  S + j + 2      <=>      t*B_sub >= 4.
    // Tile 0 (offset 0) therefore always comes from the SRAM, and so do the first tiles
    // of a SHORT sub_pass: the last sub_pass of a block can have B_sub as low as 1
    // (ceil(dim_I/2) not a multiple of 16), where 4/B_sub tiles need the macro.  Reading
    // it from the buffer instead returns the PREVIOUS sub_pass's row — measured, and it
    // is what the audit's A5 oracle exists to catch (2 bad rows at dim_I=33 before this
    // rule was added; the failing tile is exactly the one where t*B_sub < 4).
    //
    // `act_tile_idx` is the build-side tile, i.e. the tile of the presentation whose row
    // the SRAM must be handed THIS cycle (the address reaches act_row 2 cycles later,
    // through either path).
    // (widths written out: a bare `3'd? * 5'd?` would self-determine to 5 bits and a W4
    // tile-7 offset of 7*16 = 112 would wrap.)
    wire [6:0] tile_off = {3'b000, act_tile_idx} * {2'b00, b_sub_held};
    wire       act_sram_rd = ctrl_build_en && (tile_off < 7'd4);

    // The act DMA honours the port arbitration: an SRAM read the core is waiting on is
    // load-bearing (the core cannot stall), a DMA beat is not (AXI holds it).  For a full
    // sub_pass that is one read per distinct row instead of one per presentation — that
    // ratio is the entire budget the DMA gets back.  See act_port_mux.v.
    assign s_axis_act_tready = !act_sram_rd;

    // Port ownership + the SRAM/line-buffer read path live in their own modules so
    // sim/tb_act_port_audit.v can instantiate the shipping glue instead of a copy of it.
    wire [7:0]  act_scratch_addr;
    wire        act_bank0_web, act_bank1_web, act_rd;
    wire [511:0] act_row_even, act_row_odd;
    act_read_path u_act_rd (
        .clk(clk), .rst_n(rst_n),
        .abs_batch(abs_batch),
        .act_sram_rd(act_sram_rd),
        .act_load_cnt(act_load_cnt),
        .dma_act_wr_en(dma_act_wr_en),
        .act_overlap_en(act_overlap_en_r),
        .act_rd_region(act_rd_region), .act_wr_region(act_wr_region),
        .act_din(dma_act_data),
        .act_dout({act_dout[1], act_dout[0]}),
        .act_scratch_addr(act_scratch_addr),
        .act_bank0_web(act_bank0_web), .act_bank1_web(act_bank1_web),
        .act_rd(act_rd),
        .act_row_even(act_row_even), .act_row_odd(act_row_odd)
    );

    // ── Activation banks ──
    // ACT_BANKS=2 instances of sram_act_bank (512 b each = 8 x M256 macros).  Both share
    // ONE address (the macro's free top bit is the region split), and bank select is
    // `web` only, since every access asserts csb.
    // M256, not M1024: the acc row contract caps dim_I at 256, so the act address never
    // exceeds 128 -- a 1024-deep macro would be 8x deeper than addressable.
    // Library-specific macro assembly lives in rtl/sram_7nm/ (decision B).
    sram_act_bank u_act0 (
        .clk (clk), .csb (1'b0),
        .web (act_bank0_web),
        .addr(act_scratch_addr),
        .din (dma_act_data),
        .dout(act_dout[0])
    );

    sram_act_bank u_act1 (
        .clk (clk), .csb (1'b0),
        .web (act_bank1_web),
        .addr(act_scratch_addr),
        .din (dma_act_data),
        .dout(act_dout[1])
    );

    // ── 2-cycle alignment: compensate the activation read latency ──
    // act_row (driven by act_read_path above) is 2 cycles behind the controller's
    // schedule; delay the other core inputs
    // (and the weight-cache tile_boundary, done above) by 2 so the core sees a uniformly
    // shifted presentation stream.
    //
    // The engine's compute enable is `build_en` delayed by ONE MORE cycle than the other
    // core inputs: the LUT's subset sums are built on the presentation cycle and the
    // Horner consumes them one cycle later, so the core needs exactly one enable per
    // PRESENTATION, one cycle after that presentation's build.
    //
    // It used to be `ctrl_compute_en | ctrl_compute_en_d0`, i.e. the OR of two adjacent
    // cycles of (build_en & ~pipe_fill).  That is the same waveform as `build_en` delayed
    // by one *as long as the controller's build_en is one contiguous run per S_GROUP
    // visit* — which is why it passed every regression.  It is NOT the same when the FIFO
    // backpressure stalls the last presentation of a group: `S_GROUP` then holds for K
    // cycles with build_en = 0 in the middle of the run, each such hole gets its leading
    // cycle filled in by the OR (an enable with NO presentation behind it — a duplicate
    // dot, and one extra tile_valid into acc_block_reg, which shifts its batch/tile
    // counters by one for the rest of the group).  Measured on `dim_I=256` (one stall per
    // k_group): 513 tile_valids for 512 presentations — and, together with the
    // acc_block_reg resync it lands next to, the missing 16th sub_sum of the stalled group
    // (#126).  Deriving the enable from build_en makes the count exactly one per
    // presentation for every stall pattern; with no stalls the waveform is bit-identical.
    reg       compute_en_core;   // = ctrl_build_en delayed 3 (build_en_d[1] delayed 1)
    reg [2:0] act_tile_idx_d [0:1];
    reg       build_en_d     [0:1];
    reg       group_start_d  [0:1];
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            act_tile_idx_d[0] <= 0; act_tile_idx_d[1] <= 0;
            build_en_d[0] <= 0;     build_en_d[1] <= 0;
            compute_en_core <= 0;
            group_start_d[0] <= 0;  group_start_d[1] <= 0;
        end else begin
            act_tile_idx_d[0] <= act_tile_idx;   act_tile_idx_d[1] <= act_tile_idx_d[0];
            build_en_d[0] <= ctrl_build_en;      build_en_d[1] <= build_en_d[0];
            compute_en_core <= build_en_d[1];
            group_start_d[0] <= ctrl_tile_start; group_start_d[1] <= group_start_d[0];
        end
    end

    // ── R Cores ──
    // Both cores share the same 256-ch weight (wt_bcast); each processes a different
    // activation row (core0 = even, core1 = odd).
    genvar core;
    generate
        for (core = 0; core < R_CORES; core = core + 1) begin : core_inst
            sparse_ternary_core #(.N_CH(N_CH_PER)) u_core (
                .clk(clk), .rst_n(rst_n),
                .wt_mode(wt_mode),
                .act_row(core == 0 ? act_row_even : act_row_odd),
                .weight_all(wt_bcast),
                .build_en(build_en_d[1]),
                .compute_en(compute_en_core),
                .weight_signed(weight_signed),
                .act_tile_idx(act_tile_idx_d[1]),
                .group_start(group_start_d[1]),
                .b_sub(b_sub_held),
                .sub_sum(core_sub_sum[core]),
                .sub_sum_batch(core_sub_sum_batch[core]),
                .sub_sum_valid(core_sub_sum_valid[core])
            );
        end
    endgenerate

    // ── Accumulator SRAM (scheme C): sram_acc_bank (256 x M256, 16384 b port) + acc_rmw ──
    // ONE shared address drives all 256 macros; one access moves 16384 b
    // (= 2 cores x 256 ch x 32 b), i.e. ONE WHOLE ROW.
    //   address = {row[7:0]}
    // The old 16 x sdp_1024x128 (1R1W) organisation is gone: 1RW cannot read and write
    // in the same cycle, so acc_rmw splits read/write in time (3 steps/entry).
    // Why 256: the LUT floor needs 8192 b/cycle of acc port (0.25 entry/compute-cycle x
    // 32768 b/entry), so 128 macros on one address is EXACTLY matched with zero margin --
    // the pipeline bubble alone cost 20%.  Doubling the array to 256 buys a 16384-b port
    // at 50% duty, which absorbs the bubble, and doubles the row capacity to 256.
    wire [ACC_BUS_W-1:0] acc_sp_din;
    wire [ACC_BUS_W-1:0] acc_sp_dout;
    wire        acc_sp_web;
    wire [ACC_ROW_AW-1:0] acc_sp_addr;
    wire        acc_group_done;

    // acc port owner mux: acc_rmw_unit while computing, the drain scan after `done`.
    wire [ACC_ROW_AW-1:0] acc_mem_addr;
    wire        acc_mem_web;

    // ── Accumulator bank ──
    // One sram_acc_bank = 256 x M256 macros = 16384 b (= 2 cores x 256 ch x 32 b, i.e.
    // ONE WHOLE ROW per access), one shared address, 256 rows.  Library-specific macro
    // assembly lives in rtl/sram_7nm/ (decision B); this top names no macro.
    sram_acc_bank u_acc (
        .clk (clk), .csb (1'b0),
        .web (acc_mem_web),        // low-active: acc_rmw_unit / the drain scan drive it
        .addr(acc_mem_addr),       // {row[7:0]}
        .din (acc_sp_din),
        .dout(acc_sp_dout)
    );

    // The parameters are passed EXPLICITLY (they equal the module defaults at the
    // shipping geometry, so this is a no-op there).  Without them the top declares
    // N_CH_PER as a parameter but is hard-wired to 256 channels / an 8192 b acc port:
    // iverilog pads the 12296/8192-bit ports of a smaller build with zeros and the
    // accumulator silently computes garbage.  That is what blocked sim/tb_l0_top.v
    // (the reduced-geometry real-top harness, test-flow §2.4(c) route B).
    acc_rmw_unit #(
        .N_CORE(R_CORES), .N_CH_CORE(N_CH_PER), .SUBSUM_W(`ST_SUBSUM_WIDTH),
        .ACC_W(`ST_ACC_WIDTH), .ROW_W(ROW_IDX_W), .FIRST_W(FIRST_W),
        .ACC_SUB_W(ACC_SUB_W)
    ) u_acc_rmw (
        .clk(clk), .rst_n(rst_n),
        .fifo_empty(fifo_empty),
        .fifo_rd_en(fifo_rd_en),
        .fifo_data(fifo_rd_data),
        .acc_dout(acc_sp_dout),
        .acc_din(acc_sp_din),
        .acc_web(acc_sp_web),
        .acc_addr(acc_sp_addr),
        .group_done(acc_group_done)
    );

    // Prevent ABC elimination: OR-reduce a few acc macro douts
    wire [15:0] dummy_acc_dout;
    assign dummy_acc_dout = acc_sp_dout[0*`ST_ACC_WIDTH +: 16]
                          | acc_sp_dout[1*`ST_ACC_WIDTH +: 16]
                          | acc_sp_dout[2*`ST_ACC_WIDTH +: 16]
                          | acc_sp_dout[3*`ST_ACC_WIDTH +: 16];

    // (removed) N×M FP16 Systolic PE Array — it was an unwired remnant sitting next to
    // the LUT engine with only a dummy-chain consumer. The dense-FP16 path is replaced by
    // the 4× INT8 mesh (see docs/superpowers/plans/2026-09-22-asap7-sram-1rw-macro-selection.md).

    // ── Anti-optimization dummy chain ──
    assign dummy_chain = wt_bcast[15:0]
                       ^ (act_dout[0][15:0] | act_dout[1][15:0])
                       ^ dummy_acc_dout;

    // ── Completion: LUT done + all sub_sum written + FIFO drained + acc_rmw idle ──
    // The sub_sum_valid stream for the last group arrives ~ST_RES_DELAY after the group
    // boundary (well after ctrl_done), so count received sub_sum_valid pulses against the
    // number the run will actually produce to keep done from firing before the last
    // writes land.
    //
    // That number is B_batches * J_passes * K_groups, NOT B_batches: acc_block_reg emits
    // one pulse per batch (B_sub of them per sub-pass, summing to B_batches over the
    // S_passes of one block) and the whole block repeats once per (k_group, j_pass).
    // With the old `>= B_batches` test the counter was satisfied after the FIRST block
    // finished and `all_written` was then held true for the rest of the run: measured
    // 67 cycles of premature satisfaction at J=2/K=1, 108 at K=2/J=1 and 743 at
    // J=2/K=2 (sim/tb_tiling_contract.v), leaving `done` to fire on a coincidence
    // (lut_done_r && fifo_empty && !acc_busy) rather than on a proof.
    //
    // It counts ACCEPTED WRITES (fifo_wr_en), not sub_sum_valid pulses: counting the
    // pulses is what let bug 7 hide — the counter reached its total while 10.5% of
    // the entries had been dropped by the write gate, so the run looked complete and
    // the acc content was quietly wrong.  With the write path never refusing an entry
    // the two are identical; if it ever does, this now stalls `done` (and the tb's
    // timeout fires) instead of returning wrong data.
    reg [15:0] B_batches_r, J_passes_r, K_groups_r, rows_total_r;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            B_batches_r <= 0; J_passes_r <= 0; K_groups_r <= 0; rows_total_r <= 0;
        end else if (run_start) begin
            B_batches_r  <= b_batches_w;
            J_passes_r   <= j_passes_w;
            K_groups_r   <= k_groups_w;
            rows_total_r <= b_batches_w * j_passes_w;         // <= 128 (checked above)
        end
    end

    // The counter and its comparator live in `completion_counter` (module below, same
    // file) so sim/tb_tiling_contract.v can instantiate the SHIPPING rule instead of
    // re-deriving the target.  Mutation entry 3 ("all_written threshold back to
    // B_batches") was blind while the harness carried its own `sub_sum_total` copy: the
    // copy stayed correct no matter what this file said (test-flow §2.4(c)).
    wire [15:0] sub_sum_total_r;
    wire [15:0] sub_sum_count;
    wire        all_written;

    completion_counter u_completion (
        .clk(clk), .rst_n(rst_n),
        .run_start(run_start),
        .wr_en    (fifo_wr_en),          // ACCEPTED writes, not sub_sum_valid pulses
        .b_batches(b_batches_w), .j_passes(j_passes_w), .k_groups(k_groups_w),
        .total(sub_sum_total_r),
        .count(sub_sum_count),
        .all_written(all_written)
    );

    reg acc_busy;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) acc_busy <= 1'b0;
        else if (fifo_rd_en)  acc_busy <= 1'b1;
        else if (acc_group_done) acc_busy <= 1'b0;
    end

    reg all_done_r;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) all_done_r <= 1'b0;
        else if (start) all_done_r <= 1'b0;
        else if (lut_done_r && all_written && fifo_empty && !acc_busy) all_done_r <= 1'b1;
    end

    reg all_done_prev;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) all_done_prev <= 1'b0;
        else        all_done_prev <= all_done_r;
    end
    assign done = all_done_r && !all_done_prev;

    reg st_busy_r;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)          st_busy_r <= 1'b0;
        // A refused configuration never becomes busy: the host's signature for it is
        // (cfg_error == 1, busy == 0, done == 0) — not a hung busy.
        else if (start)      st_busy_r <= !cfg_overflow;
        else if (all_done_r) st_busy_r <= 1'b0;
    end
    assign busy = st_busy_r;

    // ── DMA Writer (result output): drain the acc SRAM after `done` ──
    //
    // Acc layout (scheme C; the same one rtl/acc_rmw_unit.v writes and tb_st_top_ternary.v
    // checks): one port address returns the full R_CORES*N_CH_PER word, i.e.
    // 2 cores x 256 channels x 32 b = 16384 b = ONE WHOLE ROW, and the run fills
    //   0 .. rows_total-1        (address == row)
    // with
    //   row       = j_pass * B_batches + abs_batch,  abs_batch = sub_pass*16 + batch
    //   rows_total = B_batches * J_passes            (<= 128 rows, see cfg_overflow)
    //
    // ── Output order (deterministic, and it is the contract the host decodes) ──
    //   for j_pass (outer):  for abs_batch:  32 x 512 b beats
    // i.e. the j_pass blocks come out CONSECUTIVELY, one complete 256-channel result
    // each: block j carries the 256 output channels 256*j .. 256*j+255 (through
    // rtl's channel j-1).  ⚠️ THE BEAT PAYLOAD IS SCHEME C's, AND IT IS NOT THE OLD
    // ONE.  The 16384 b word is [core0 | core1] — the CORE is the outer axis, because
    // acc_rmw_unit builds acc_din[c*N_CH_PER*32 + l*32] and the drain slices that word
    // flat.  So within one 16384 b word:
    //   [0     +: 8192] = core0 (even activation row 2*abs_batch)
    //   [8192  +: 8192] = core1 (odd  activation row 2*abs_batch+1)
    // each core 256 lanes x 32 b where lane l is channel l, and a 512-b beat therefore
    // decodes as  core = slice/16,  channel-in-core = (slice%16)*16 + lane.
    // The OLD word was [core0 half | core1 half] with `half` in the ADDRESS; a decoder
    // written to that form (core = slice/8, ch = half*128 + ...) reads a DIFFERENT beat
    // and reports about half the stream as wrong.  That is exactly what happened to
    // tb_l0_top / tb_st_top_ternary / tb_st_top_w4 between 7bf14ed and this fix.
    //
    // So the drain walks rows_total addresses and slices each 16384 b word into
    // OUT_SLICES x 512 b beats.
    //
    // ── The address lag (fixed 2026-09-25) ────────────────────────────────────────────
    // `out_addr_r` is written at the END of the OS_ADDR cycle (NBA), so it is only stable
    // from the NEXT cycle on — and the macro samples its address input at the rising edge,
    // i.e. it sampled whatever out_addr_r held during that OS_ADDR cycle, which is the
    // PREVIOUS address.  Capturing `acc_sp_dout` one cycle later (in OS_WAIT) therefore
    // collected the previous address's data: the whole output stream was shifted by one
    // address, and the last address was never emitted.  (Measured: beats 0..15 = addr 0,
    // beats 16..31 = addr 0 again, … — sim/tb_st_top_ternary's drain-order check.)
    // Fix: one extra state, so the capture happens after the address has been stable for a
    // full cycle: OS_ADDR (address register updated) -> OS_WAIT (macro reads it) ->
    // OS_WAIT2 (capture) -> OS_BEAT (OUT_SLICES beats).  Purely a drain-side change:
    // `done` and every cycle count up to it are untouched.
    //
    // For a single j_pass with even dim_I this is bit-for-bit the old dim_I-1 bound
    // (2*B_batches-1 == dim_I-1); for J_passes > 1 it now covers every pass instead of
    // stopping after the first one's rows.
    //
    // The sequencer itself is `drain_fsm` (module at the bottom of this file).  It is a
    // module so that a seconds-level harness (sim/tb_drain_contract.v) can instantiate
    // the SHIPPING rtl rather than a re-implementation of it — otherwise a regression
    // here is only reachable from the ~20-minute full-top build (test-flow §1, §2.4).
    wire       out_active;
    wire [ACC_ROW_AW-1:0] out_addr_r;

    drain_fsm #(
        .R_CORES(R_CORES), .N_CH_PER(N_CH_PER), .ACC_SUB_W(ACC_SUB_W)
    ) u_drain (
        .clk(clk), .rst_n(rst_n),
        .done(done), .rows_total(rows_total_r),
        .acc_dout(acc_sp_dout),
        .m_axis_out_tready(m_axis_out_tready),
        .out_active(out_active), .out_addr(out_addr_r),
        .m_axis_out_tdata(m_axis_out_tdata),
        .m_axis_out_tvalid(m_axis_out_tvalid),
        .m_axis_out_tlast(m_axis_out_tlast)
    );

    // The drain owns the acc port while it runs (acc_rmw_unit has already gone idle by
    // the time `done` pulses); otherwise acc_rmw_unit drives it.
    assign acc_mem_addr   = out_active ? out_addr_r : acc_sp_addr;
    assign acc_mem_web    = out_active ? 1'b1      : acc_sp_web;
    assign out_drain_busy = out_active;

endmodule

// ──────────────────────────────────────────────────────────────────────────────
// drain_fsm — the result-drain address/beat sequencer, extracted VERBATIM from the
// top level (2026-09-25, no logic change: same states, same assignments).
//
// WHY IT IS A MODULE.  Bug 6 (`out_addr_r` written by NBA at the end of OS_ADDR, so the
// macro sampled the PREVIOUS address; the whole output stream was shifted by one address
// and the last address never came out) was only reachable from the ~20-minute full-top
// build.  That is the feedback-speed problem of test-flow §0-D: the one bug the drain
// layer hid behind was the one the drain layer could not see cheaply.  As a module it is
// pinned by sim/tb_drain_contract.v (seconds), and mutation-test entry 6 — "drain address
// ±1 cycle" — has to go red THERE rather than only at L2.
//
// A copy of the FSM in the testbench would not do: the mutation would then have to be
// injected into the copy, which proves nothing about this rtl (test-flow §2.4(b)).
//
// Contract (asserted by tb_drain_contract.v).  Scheme C: the address IS the row, so the
// walk is over rows_total addresses (it used to be 2*rows_total = {row, half}):
//   * exactly rows_total addresses, walked in ascending order, no repeat and no gap;
//   * each address is presented for OUT_SLICES = 32 consecutive 512-b beats;
//   * tlast on the very last beat only; total beats = rows_total*32;
//   * beat `n` carries slice n%OUT_SLICES of address n/OUT_SLICES — i.e. the captured
//     word must be the word of ITS OWN address (the bug-6 signature is a stream shifted
//     by one address).
// ──────────────────────────────────────────────────────────────────────────────
module drain_fsm #(
    parameter R_CORES    = `ST_R_CORES,
    parameter N_CH_PER   = `ST_N_CH_PER_CORE,
    parameter ACC_W      = `ST_ACC_WIDTH,
    parameter ACC_SUB_W  = 2,                          // 4 sub-words per row
    parameter OUT_W      = R_CORES*N_CH_PER*ACC_W/(1<<ACC_SUB_W),   // 4096
    parameter ROW_W      = 8,
    parameter SUB_SLICES = OUT_W/512                   // 8 beats per sub-word
) (
    input  wire             clk,
    input  wire             rst_n,
    input  wire             done,        // 1-cycle pulse: start a drain
    input  wire [15:0]      rows_total,  // rows to walk = B_batches * J_passes
    input  wire [OUT_W-1:0] acc_dout,    // acc sub-word at the registered address
    input  wire             m_axis_out_tready,
    output wire             out_active,
    output wire [ROW_W+ACC_SUB_W-1:0] out_addr,   // {row, sel}
    output wire [511:0]     m_axis_out_tdata,
    output wire             m_axis_out_tvalid,
    output wire             m_axis_out_tlast
);
    localparam N_SUB   = 1 << ACC_SUB_W;               // 4
    localparam OS_IDLE = 3'd0, OS_ADDR = 3'd1, OS_WAIT = 3'd2, OS_WAIT2 = 3'd3, OS_BEAT = 3'd4;

    reg [2:0]                 os;
    reg [15:0]                out_row;
    reg [ACC_SUB_W-1:0]       out_sel;
    reg [3:0]                 out_cnt;
    reg [ROW_W+ACC_SUB_W-1:0] out_addr_r;
    reg [OUT_W-1:0]           out_word_r;

    // Scheme C4: the acc address is {row, sel}; each sub-word is SUB_SLICES beats.
    // Emitting sub-word sel in order preserves the original beat order exactly:
    // beat n of the row = sub-word (n/8), beat (n%8).
    wire [15:0] out_last_addr = rows_total - 16'd1;    // B_batches*J_passes - 1
    wire        out_last      = (out_row == out_last_addr);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            os <= OS_IDLE; out_row <= 16'd0; out_sel <= 0; out_cnt <= 4'd0;
            out_addr_r <= 0; out_word_r <= {OUT_W{1'b0}};
        end else begin
            case (os)
                OS_IDLE: if (done && (rows_total != 16'd0)) begin
                             out_row <= 16'd0; out_sel <= 0; out_cnt <= 4'd0;
                             os <= OS_ADDR;
                         end
                OS_ADDR:  begin out_addr_r <= {out_row[ROW_W-1:0], out_sel}; os <= OS_WAIT; end
                OS_WAIT:  begin os <= OS_WAIT2; end          // macro latches out_addr_r here
                OS_WAIT2: begin out_word_r <= acc_dout; out_cnt <= 4'd0; os <= OS_BEAT; end
                OS_BEAT: if (m_axis_out_tready) begin
                             if (out_cnt == SUB_SLICES-1) begin
                                 if (out_sel == N_SUB-1) begin
                                     if (out_last) os <= OS_IDLE;
                                     else begin out_row <= out_row + 16'd1; out_sel <= 0; os <= OS_ADDR; end
                                 end else begin out_sel <= out_sel + 1'b1; os <= OS_ADDR; end
                             end else begin
                                 out_cnt <= out_cnt + 4'd1;
                             end
                         end
                default: os <= OS_IDLE;
            endcase
        end
    end

    assign out_active        = (os != OS_IDLE);
    assign out_addr          = out_addr_r;
    assign m_axis_out_tvalid = (os == OS_BEAT);
    assign m_axis_out_tdata  = out_word_r[out_cnt*512 +: 512];
    assign m_axis_out_tlast  = (os == OS_BEAT) && (out_cnt == SUB_SLICES-1)
                               && (out_sel == N_SUB-1) && out_last;

endmodule

// ──────────────────────────────────────────────────────────────────────────────
// acc_row_encode — which acc row does one sub_sum entry belong to?
//
//     row = j_pass * B_batches + abs_batch,   abs_batch = sub_pass*N_BATCH + batch
//
// WHY IT IS A MODULE.  Mutation entry 2 ("drop the `j_pass * B_batches` term", the
// original bug where the second j_pass overwrote the first) was BLIND: the only
// seconds-level harness that looks at rows carried a "copied verbatim" copy of this
// expression, so editing this file changed nothing for it (test-flow §2.4(c)).  The
// arithmetic is therefore shipped as a module and instantiated by BOTH the top and
// sim/tb_tiling_contract.v — a mutation here now reaches the harness by construction.
//
// `b_batches` is the LIVE value (not the run_start-latched register): they are equal
// for the whole run, and it keeps the module free of extra state.
// ──────────────────────────────────────────────────────────────────────────────
module acc_row_encode #(
    parameter ROW_IDX_W = 7,
    parameter N_BATCH   = 16
) (
    input  wire [3:0]           sub_pass,     // sub_pass of the entry
    input  wire [3:0]           batch,        // batch within the sub_pass
    input  wire [7:0]           j_pass,       // j_pass of the weight block
    input  wire [15:0]          b_batches,    // ceil(dim_I / R_CORES)
    output wire [15:0]          row_full,     // unmasked encoding
    output wire [ROW_IDX_W-1:0] row           // the acc macro row index
);
    wire [7:0] abs_batch = sub_pass * N_BATCH + {4'b0, batch};
    assign row_full = {8'd0, j_pass} * b_batches + {8'd0, abs_batch};
    assign row      = row_full[ROW_IDX_W-1:0];
endmodule

// ──────────────────────────────────────────────────────────────────────────────
// completion_counter — "every sub_sum of this run has been written to the FIFO"
//
//     total       = B_batches * J_passes * K_groups
//     all_written = (count >= total)
//
// WHY IT IS A MODULE.  Mutation entry 3 ("threshold back to B_batches") was blind for
// the same reason as acc_row_encode: sim/tb_tiling_contract.v computed its own
// `sub_sum_total` and so could never notice the top's threshold changing
// (test-flow §2.4(c)).  `wr_en` counts ACCEPTED writes (fifo_wr_en), not the raw
// sub_sum_valid pulses — counting pulses is what let bug 7 hide (entries dropped by
// the write gate still "completed" the run).
// ──────────────────────────────────────────────────────────────────────────────
module completion_counter (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        run_start,     // pulse: new run -> latch the target, clear the count
    input  wire        wr_en,         // pulse: one entry ACCEPTED by the FIFO write gate
    input  wire [15:0] b_batches,
    input  wire [15:0] j_passes,
    input  wire [15:0] k_groups,
    output reg  [15:0] total,
    output reg  [15:0] count,
    output wire        all_written
);
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            total <= 16'd0;
            count <= 16'd0;
        end else if (run_start) begin
            total <= b_batches * j_passes * k_groups;
            count <= 16'd0;
        end else if (wr_en) begin
            count <= count + 16'd1;
        end
    end

    assign all_written = (count >= total);
endmodule

// ──────────────────────────────────────────────────────────────────────────────
// fifo_bp_cmp — the sub_sum FIFO backpressure threshold
//
//     bp = (level > LEVEL)
//
// WHY IT IS A MODULE.  Mutation entry 7 (`fifo_bp` tied to 0, i.e. backpressure off)
// was blind: sim/tb_fifo_rate.v's mini-top mode modelled this comparison itself, so
// the harness's own copy kept stopping the producer no matter what the rtl did
// (test-flow §2.4(c)).  Its `sim-fifo-rate` assertions now drive this instance.
// ──────────────────────────────────────────────────────────────────────────────
module fifo_bp_cmp #(
    parameter LEVEL = 12,
    parameter W     = 6
) (
    input  wire [W-1:0] level,
    output wire         bp
);
    localparam [W-1:0] LVL = LEVEL;
    assign bp = (level > LVL);
endmodule

// ──────────────────────────────────────────────────────────────────────────────
// wgt_fill_handshake — "the weight block the ctrl is waiting for has landed"
//
// The LUT engine consumes one weight block per (k_group, j_pass) pair, and
// weight_tile_cache.v reloads its 3-tile window from SRAM words 0..N_TILES*CHUNKS-1
// for each block.  The cache therefore only holds the *right* weights once the DMA
// has re-filled the SRAM for that block (weight_tile_cache.v:30-31).
//
// A completed/consumed counter pair expresses that directly and is order-preserving:
// a block the DMA prefetched before the ctrl asked for it is still reported ready.
// `cache_ready` stays in the expression so the cache init double-fill (the top ties
// `wgt_fill_done` to the cache's `start`) is still a prerequisite.
//
// Instantiated by sparse_ternary_top and by sim/tb_tiling_contract.v (which mirrors
// the top's ctrl/addr_gen wiring and can therefore check the contract in ~2 s).
// ──────────────────────────────────────────────────────────────────────────────
module wgt_fill_handshake (
    input  wire clk,
    input  wire rst_n,
    input  wire run_start,      // pulse: new run — clear both counters
    input  wire blk_done,       // pulse: one weight DMA block completed (tlast)
    input  wire blk_consume,    // pulse: ctrl consumed the handshake (left S_LOAD_WGT)
    input  wire cache_ready,    // weight_tile_cache.ready
    output wire fill_done
);
    reg [15:0] arrived;         // blocks completed by the DMA
    reg [15:0] consumed;        // blocks consumed by the ctrl

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)          arrived <= 16'd0;
        else if (run_start)  arrived <= 16'd0;
        else if (blk_done)   arrived <= arrived + 16'd1;
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)          consumed <= 16'd0;
        else if (run_start)  consumed <= 16'd0;
        else if (blk_consume) consumed <= consumed + 16'd1;
    end

    assign fill_done = cache_ready && (arrived != consumed);

endmodule

// ──────────────────────────────────────────────────────────────────────────────
// cfg_row_check — does this (dim_I, dim_J, dim_K) fit the acc row space?
//
// The acc row index is
//     row = j_pass * B_batches + abs_batch,   B_batches = ceil(dim_I / R_CORES)
// so the run needs `B_batches * J_passes` of the 128 physical rows, with
//     J_passes = ceil(dim_J / N_CH_PER)   (one j_pass = N_CH_PER output channels)
//     K_groups = dim_K / GROUP_K          (>= 1: the loop always runs one group)
//
// The old design did not check this at all: `fifo_row = row_idx[6:0]` simply truncated,
// so a too-large tile set silently wrote pass j on top of pass j-1 (measured 16/16 rows
// colliding at dim_J=512 — sim/tb_tiling_contract.v).  sparse_ternary_top refuses to
// start when `overflow` is set and reports it on `cfg_error`.
//
// `k_groups` keeps working below dim_K == GROUP_K because the ctrl/addr_gen loop always
// executes one k_group regardless of the dim_K/GROUP_K quotient; the sub_sum pulse total
// B_batches*J_passes*K_groups has to agree with the ctrl, not with integer division.
//
// Instantiated by sparse_ternary_top and by sim/tb_tiling_contract.v.
// ──────────────────────────────────────────────────────────────────────────────
module cfg_row_check #(
    parameter R_CORES  = 2,
    parameter N_CH_PER = 256,
    parameter MAX_ROWS = 128,
    parameter GROUP_K  = 128
) (
    input  wire [15:0] dim_I,
    input  wire [15:0] dim_J,
    input  wire [15:0] dim_K,
    output wire [15:0] b_batches,
    output wire [15:0] j_passes,
    output wire [15:0] k_groups,
    output wire [31:0] rows_needed,
    output wire        overflow
);
    // A zero dimension cannot produce a result, so it is rejected with the rest.
    wire dims_zero = (dim_I == 16'd0) || (dim_J == 16'd0) || (dim_K == 16'd0);

    assign b_batches = (dim_I + R_CORES - 1) / R_CORES;
    assign j_passes  = (dim_J + N_CH_PER - 1) / N_CH_PER;
    assign k_groups  = (dim_K < GROUP_K) ? 16'd1 : (dim_K / GROUP_K);

    assign rows_needed = {16'd0, b_batches} * {16'd0, j_passes};

    assign overflow = dims_zero || (rows_needed > MAX_ROWS);

endmodule
