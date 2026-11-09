// sparse_ternary_ctrl.v — Tile-major tiling FSM for the Sparse Ternary LUT engine
//
// States: IDLE → LOAD_WGT → LOAD_ACT → GROUP → DONE
//
// Tile-major loop (reordered from batch-major): the batch is the INNER loop and
// the weight/activation tile is OUTER:
//   for k_group: for j_pass: for sub_pass: for tile(0..7): for batch(0..N_BATCH-1)
//
// The register-block accumulator (rtl/acc_block_reg.v) holds N_BATCH=16 running
// sums, so a batch range larger than N_BATCH is split into
//   S_passes = ceil(B_batches / N_BATCH)
// sub-passes; each sub-pass accumulates 8 tiles × N_BATCH batches (the last
// sub-pass is shorter).  Weight tiles are re-read between sub-passes (the weight
// tile index wraps back to 0), so weight reuse is N_BATCH presentations, not
// B_batches.
//
// One "group" (one acc_block_reg accumulation unit) = ST_TILES_PER_GROUP × N_BATCH
// presentations (128 cycles for a full sub-pass, 8×B_sub for the last).  The FSM
// still emits 1 build + 1 compute every cycle in steady state; group_boundary
// fires once per group (once per sub-pass) at the last (tile, batch) presentation.
// group_done fires ST_RES_DELAY cycles later (pipelined compute path).
//
`include "sparse_ternary_params.vh"

(* keep_hierarchy *) module sparse_ternary_ctrl #(
    parameter R      = `ST_R_CORES,
    parameter T_PER  = `ST_T_COPIES_PER_CORE,
    // Channels one j_pass covers = channels per core.  A PARAMETER (default = the
    // shipping macro) rather than the macro directly: it is what cfg_row_check and
    // addr_gen use, and while it was hard-wired here the top's own `N_CH_PER` parameter
    // was only half plumbed — a reduced-geometry build computed J_passes from 256 while
    // the rest of the design computed it from N_CH_PER, so the ctrl stopped after the
    // first j_pass and `done` never pulsed (found by sim/tb_l0_top.v, test-flow §2.4(c)).
    parameter N_CH_PER_CORE = `ST_N_CH_PER_CORE
) (
    input  wire         clk,
    input  wire         rst_n,
    input  wire         start,
    output reg          done,           // pulse: all LUT groups complete

    input  wire [15:0]  dim_I,
    input  wire [15:0]  dim_J,
    input  wire [15:0]  dim_K,

    // Backpressure from the sub_sum FIFO.  NOT the FIFO's literal `full`: the
    // producer must stop with HEADROOM to spare, because the entries already in
    // the compute pipeline keep arriving for ST_RES_DELAY cycles after the stop.
    // The top drives this from the FIFO occupancy (see FIFO_BP_LEVEL there); a
    // literal `full` here is far too late and silently overflows
    // (sim/tb_fifo_rate.v).
    input  wire         fifo_bp,
    input  wire         wt_mode,        // 1=ternary(4 tiles/group) 0=W4(8 tiles/group)

    // Core control signals (broadcast to all R cores)
    output reg          build_en,
    output reg          compute_en,
    output reg          tile_start,     // first (tile, batch) of a group (acc_block_reg resync)
    output wire         group_boundary, // pulse: group boundary (last (tile,batch) presented)
    output wire         group_done,     // pulse: group result ready (= group_boundary delayed ST_RES_DELAY)

    // SRAM address generation
    output reg [9:0]    wt_sram_addr,   // 1024 depth
    output reg [10:0]   act_sram_addr,  // 2048 depth
    output reg [8:0]    acc_sram_addr,  // 512 depth

    // DMA control (combinational load-start pulses: fire on entry to each LOAD state)
    output wire         dma_wt_start,
    output wire         dma_act_start,
    input  wire         dma_wt_done,
    input  wire         dma_act_done,

    // Weight buffer ping-pong
    output reg          buf_sel,       // active weight buffer (compute reads this)
    output wire         fill_swap,     // pulse: buffer swapped (clear fill_done)
    input  wire         fill_done,     // fill buffer holds a complete weight group
    output wire         act_prefetch,  // Task 5: activation transpose-load prefetch (tied 0 here)

    // Debug
    output wire [3:0]   dbg_state
);

    localparam S_IDLE      = 4'd0;
    localparam S_LOAD_WGT  = 4'd1;
    localparam S_LOAD_ACT  = 4'd2;
    localparam S_GROUP     = 4'd3;
    localparam S_ADVANCE   = 4'd4;
    localparam S_DONE      = 4'd5;

    // ── Tile-major blocking parameters ──
    //   N_BATCH must match sparse_ternary_core / acc_block_reg N_BATCH (= 16).
    localparam N_BATCH      = 16;                          // register-block depth
    localparam BATCH_W      = $clog2(N_BATCH);             // 4
    localparam TILE_W       = 3;  // $clog2(8) max; runtime wrap (3 ternary / 7 W4)
    localparam GROUP_CYCLES = `ST_TILES_PER_GROUP * N_BATCH; // 128 (full sub-pass)

    reg [3:0] state, next_state;
    assign dbg_state = state;

    // ── Tiling counters (tile-major) ──
    //   batch     : sub-pass-local batch (0..B_sub-1), INNER loop
    //   tile      : 0..7, OUTER within a sub-pass
    //   sub_pass  : 0..S_passes-1 (batch blocking)
    //   j_pass    : 0..J_passes-1
    //   k_group   : 0..K_groups-1
    reg [15:0] k_group, j_pass, sub_pass;
    reg [15:0] K_groups, J_passes, B_batches, S_passes;
    reg [TILE_W-1:0]  tile;        // 0..7
    reg [BATCH_W-1:0] batch;       // 0..15 (sub-pass local)
    reg        pipe_fill;          // 1 = pipeline filling (first S_GROUP cycle builds only)
    reg [1:0]  act_settle;         // activation-SRAM read settle countdown in S_LOAD_WGT

    // Live batch count for the current sub-pass (the last sub-pass may be short):
    //   B_sub = N_BATCH for every sub-pass except the last, where it is the
    //   remaining batch count B_batches - sub_pass*N_BATCH (in 1..N_BATCH).
    wire [15:0] B_sub = (sub_pass + 1 < S_passes)
                        ? 16'd16                       // N_BATCH
                        : (B_batches - (sub_pass << 4));

    // ── Group end: the last (tile, batch) presentation of a sub-pass ──
    wire last_batch = (batch == B_sub - 16'd1);
    wire last_tile  = (tile == (wt_mode ? 3'd3 : 3'd7));
    wire group_end  = last_batch && last_tile;

    // Combinational group-boundary pulse: fires at the last (tile, batch) of each
    // sub-pass (once per acc_block_reg accumulation unit), unless the FIFO is at
    // or above its backpressure level.  While stalled the FSM HOLDS this last
    // presentation (build_en/compute_en are suppressed and the counters hold), so
    // the group is re-presented rather than lost once the consumer catches up.
    // The RESULT appears ST_RES_DELAY cycles later (pipelined compute path), so the
    // FIFO write (group_done) is the boundary pulse delayed by ST_RES_DELAY.
    wire group_done_w  = (state == S_GROUP) && group_end && !fifo_bp;
    wire group_stalled = (state == S_GROUP) && group_end &&  fifo_bp;
    assign group_boundary = group_done_w;

    reg [`ST_RES_DELAY-1:0] done_sr;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) done_sr <= 0;
        else        done_sr <= {done_sr[`ST_RES_DELAY-2:0], group_done_w};
    end
    assign group_done = done_sr[`ST_RES_DELAY-1] && !fifo_bp;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) state <= S_IDLE;
        else        state <= next_state;
    end

    // ── Combinational next-state ──
    // Next tiling state after the current group.  Declared as a wire (not a
    // function) so the continuous assigns below are correctly sensitive to the
    // tiling counters (sub_pass/j_pass/k_group) — a function call inside an
    // `assign` left these counters out of the implicit sensitivity list,
    // stalling dma_act_start/dma_wt_start at the k_group boundary.
    wire [3:0] advance_tile;
    assign advance_tile = (sub_pass + 1 < S_passes)      ? S_GROUP    :  // next sub-pass (stay in GROUP)
                          (j_pass + 1 < J_passes)      ? S_LOAD_WGT :  // reload weights for next j_pass
                          (k_group + 1 < K_groups)     ? S_LOAD_ACT :  // reload activations for next k_group
                                                         S_DONE;

    // ── Activation block handshake (level, not pulse) + prefetch ───────────────
    // Plan (b): the activation block for the NEXT k_group is now requested when the
    // current one is CONSUMED (i.e. while the engine is still computing), exactly like
    // the weight path above — that is what takes the load off the critical path.  The
    // consequence is that `dma_act_done` normally arrives DURING S_GROUP, so a
    // combinational `next_state = dma_act_done` would miss it (it is a 1-cycle pulse).
    // The handshake is therefore a level, `act_arrived != act_consumed`, the same shape
    // wgt_fill_handshake uses for weights: a block the DMA delivered early still reads
    // as ready.
    //
    // `act_ovl` = "the geometry can hold two address regions" (ceil(dim_I/2) <= 128,
    // the macro depth halved).  When it cannot, the old serialised load is kept: the
    // request still fires at the k_group boundary (where the FSM waits in S_LOAD_ACT,
    // so the DMA's writes cannot land on a row the core is reading) and there is no
    // prefetch.  Deriving it here rather than taking a pin keeps every existing
    // ctrl instantiation valid, and it is the same expression the top latches.
    reg [15:0] act_arrived, act_consumed;
    wire       act_fill_done = (act_arrived != act_consumed);
    wire       act_more_kg   = (k_group + 16'd1 < K_groups);
    wire       act_ovl       = (B_batches <= 16'd128);

    // Count block COMPLETIONS, not cycles of `dma_act_done`.  A real DMA drives tlast as
    // a 1-cycle beat so the two agree, but several seconds-level harnesses model the
    // handshake as a LEVEL (`dma_act_done = (dbg_state == S_LOAD_ACT)`); counting levels
    // would inflate `arrived` by one per visit and leave `act_fill_done` stuck high, so
    // the NEXT block would be consumed before it arrived.
    reg act_done_d;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) act_done_d <= 1'b0;
        else        act_done_d <= dma_act_done;
    end
    wire act_done_rise = dma_act_done && !act_done_d;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)            act_arrived <= 16'd0;
        else if (start)        act_arrived <= 16'd0;
        else if (act_done_rise) act_arrived <= act_arrived + 16'd1;
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)     act_consumed <= 16'd0;
        else if (start) act_consumed <= 16'd0;
        else if (state == S_LOAD_ACT && act_fill_done)
            act_consumed <= act_consumed + 16'd1;
    end

    // ── DMA load-start pulses (combinational): fire on entry to each LOAD state ──
    //
    // dma_wt_start is PREFETCHED (2026-09-27, plan 2026-09-27-gate-full-load.md 台阶 1).
    // The block for the NEXT (j_pass, k_group) is requested the moment the current block
    // is consumed — i.e. while the current block is still being computed — instead of
    // waiting for the ctrl to enter S_LOAD_WGT.  That is what takes the weight DMA fill
    // (NW beats) off the critical path.  It is safe because the weight SRAM holds TWO
    // blocks (sparse_ternary_top.v's wr_sel + weight_tile_cache.v's two buffer banks), so
    // the fill of the next block cannot disturb the one being broadcast.
    //
    // `fill_swap` ("the ctrl consumed the handshake") and this pulse are deliberately the
    // SAME cycle: the consume frees the freshly filled block for the cache and
    // simultaneously makes the DMA's target region free for the next block.
    //
    // A request counter rather than "is there another block in the loop?" — the ctrl
    // passes through S_LOAD_ACT -> S_LOAD_WGT at EVERY k_group boundary, so a
    // loop-shape-shaped condition fired a second time for a block the prefetch had already
    // asked for (measured: K_groups=2 issued 3 requests for 2 blocks, sim/tb_tiling_contract
    // case 6/8).  `n_req` makes "one request per block" structural:
    //   * request #1 (the only one from S_LOAD_ACT — the first block has nothing to
    //     overlap with, and this is what keeps single-block runs cycle-for-cycle identical);
    //   * every later one from the consume of the block before it;
    //   * nothing once all J_passes*K_groups blocks have been asked for.
    // sim/tb_tiling_contract.v asserts the resulting request/consume ordering and that the
    // ctrl never presents a block the DMA has not delivered.
    reg [15:0] n_req;                    // weight blocks requested so far this run
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)             n_req <= 16'd0;
        else if (start)         n_req <= 16'd0;
        else if (dma_wt_start)  n_req <= n_req + 16'd1;
    end

    wire blk_first_req = (n_req == 16'd0);
    // Is there a block after the one the counters currently name?  At the consume the
    // counters still hold the block being consumed, so this is exactly "the next block
    // exists".  (Not `n_req < J_passes*K_groups`: that product overflows 16 bits for the
    // largest legal dim_K, and a wrapped-to-0 product would silently disable the prefetch
    // for every block after the first.)
    wire blk_more_req  = (j_pass + 16'd1 < J_passes) || (k_group + 16'd1 < K_groups);

    assign dma_act_start = (state == S_IDLE && start) ||
                           (state == S_LOAD_ACT && act_fill_done && act_more_kg && act_ovl) ||
                           (state == S_GROUP && group_done_w && advance_tile == S_LOAD_ACT
                            && !act_ovl);
    assign dma_wt_start  = (state == S_LOAD_ACT && dma_act_done && blk_first_req) ||
                           (state == S_LOAD_WGT && fill_done && act_settle == 0 && blk_more_req);

    always @(*) begin
        next_state = state;
        case (state)
            S_IDLE:     if (start)                    next_state = S_LOAD_ACT;
            S_LOAD_ACT: if (act_fill_done)            next_state = S_LOAD_WGT;
            S_LOAD_WGT: if (fill_done && act_settle == 0) next_state = S_GROUP;
            S_GROUP:    if (group_done_w)             next_state = advance_tile;
            S_DONE:                                   next_state = S_IDLE;
            default:                                  next_state = S_IDLE;
        endcase
    end

    // ── Weight buffer ping-pong ──
    // buf_sel flips on the weight stream's tlast (dma_wt_done) so the fill target
    // (fill_buf = ~buf_sel) switches one cycle BEFORE the next group's first beat,
    // avoiding an overwrite of the just-computed buffer.  fill_done latches on the
    // tlast and is cleared (fill_swap) only when LOAD_WGT consumes the buffer, so
    // a prefetched group stays "done" until the ctrl actually reads it. ──
    assign fill_swap = (state == S_LOAD_WGT && fill_done && act_settle == 0);
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) buf_sel <= 1'b0;
        else if (dma_wt_done) buf_sel <= ~buf_sel;
    end

    // ── Per-group schedule (combinational): fully-pipelined, 1 build + 1 compute
    //    every cycle in steady state.  The first cycle in S_GROUP (pipe_fill) builds
    //    only — the compute pipeline is still filling. ──
    always @(*) begin
        build_en   = 1'b0;
        compute_en = 1'b0;
        tile_start = 1'b0;
        if (state == S_GROUP && !group_stalled) begin
            build_en   = 1'b1;
            compute_en = !pipe_fill;                  // fill cycle: build only
            tile_start = (tile == 0) && (batch == 0); // first (tile, batch) of each group
        end
    end

    // ── Activation prefetch ──
    // batch-major prefetched the next batch's activations once per group (gc 5).
    // In tile-major the activation row changes EVERY cycle (batch is the inner
    // loop), so there is no per-group prefetch.  The activation read path is
    // rewritten in Task 5 (activation transpose load); tie off here and re-wire
    // there.
    assign act_prefetch = 1'b0;

    // ── Pipeline fill flag: set when entering S_GROUP (build only on the first
    //    cycle), cleared after that first cycle so subsequent cycles compute. ──
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) pipe_fill <= 1'b1;
        else if (state != S_GROUP) pipe_fill <= 1'b1;
        else pipe_fill <= 1'b0;
    end

    // ── Activation-SRAM read settle: the single-port activation SRAM has a
    //    2-cycle read latency (address→dout) plus the act_row register, so after
    //    S_LOAD_ACT finishes DMA'ing a new k_group's activations the act_row_even/
    //    odd registers need a couple of cycles to reflect the new row before the
    //    first build.  This only adds latency once per k_group, never inside the
    //    GROUP steady state. ──
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) act_settle <= 2'd2;
        else if (state != S_LOAD_WGT) act_settle <= 2'd2;
        else if (act_settle != 0)     act_settle <= act_settle - 1'b1;
    end

    // ── Sequential control / tiling counters ──
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            done <= 1'b0;
            wt_sram_addr <= 0; act_sram_addr <= 0; acc_sram_addr <= 0;
            k_group <= 0; j_pass <= 0; sub_pass <= 0; tile <= 0; batch <= 0;
            K_groups <= 0; J_passes <= 0; B_batches <= 0; S_passes <= 0;
        end else begin
            done <= 1'b0;

            case (state)
                S_IDLE: begin
                    if (start) begin
                        K_groups <= dim_K / `ST_GROUP_K;
                        // 256 ch shared across R cores (each core = a different
                        // activation row), so one j_pass covers N_CH_PER_CORE
                        // output channels.
                        J_passes <= (dim_J + N_CH_PER_CORE - 1) / N_CH_PER_CORE;
                        B_batches <= (dim_I + R - 1) / R;   // R rows per batch
                        S_passes  <= ((dim_I + R - 1) / R + N_BATCH - 1) / N_BATCH;
                        k_group <= 0; j_pass <= 0; sub_pass <= 0; tile <= 0; batch <= 0;
                    end
                end

                S_LOAD_ACT: begin
                    // wait for dma_act_done (next_state -> S_LOAD_WGT)
                end

                S_LOAD_WGT: begin
                    // wait for dma_wt_done (next_state -> S_GROUP)
                end

                S_GROUP: begin
                    // Tile-major advance: batch (inner) every cycle, tile when
                    // batch wraps, sub_pass (then j_pass/k_group) when tile wraps.
                    if (!group_stalled) begin
                        if (group_done_w) begin
                            // last (tile, batch) of the sub-pass: wrap tile/batch
                            // and advance sub_pass.
                            batch <= 0;
                            tile  <= 0;
                            if (sub_pass + 1 < S_passes) begin
                                sub_pass <= sub_pass + 1;
                            end else begin
                                sub_pass <= 0;
                                if (j_pass + 1 < J_passes) begin
                                    j_pass <= j_pass + 1;
                                end else begin
                                    j_pass <= 0;
                                    if (k_group + 1 < K_groups)
                                        k_group <= k_group + 1;
                                end
                            end
                        end else if (last_batch) begin
                            // batch wrapped within a tile: advance tile (tile 0..6;
                            // tile == 7 is the group_done_w case above).
                            batch <= 0;
                            tile  <= tile + 1'b1;
                        end else begin
                            batch <= batch + 1'b1;
                        end
                    end
                end

                S_DONE: begin
                    done <= 1'b1;   // 1-cycle pulse: all LUT groups complete
                end
            endcase
        end
    end

endmodule
