// acc_block_reg.v — Tile-major register-blocked accumulator for the LUT engine
//
// The tile-major dataflow emits one redundant per-tile dot (tile_s, tile_c) per
// cycle; this module blocks N_BATCH activation rows in a register file and
// accumulates the running sum across the 8 (W4) / 4 (ternary) tiles of a k_group,
// producing one K=128 sum per batch at the group boundary.
//
// Accumulation (3-stage pipeline — CSA + CPA, no redundant register file):
//   stage 0 (cycle T):   latch tile_s/tile_c (the redundant dot) + batch_r/first_r/last_r
//   stage 1 (cycle T+1): 3:2 CSA reduce (first_r ? 0 : reg_file[batch_r]) + tile_s + tile_c
//                        -> (sum_r, carry_r)   [one CSA3 per channel, no carry-propagate]
//   stage 2 (cycle T+2): reg_file[batch_rr] = sum_r + carry_r   [one CPA per channel]
//
//   This replaces the old 2-CPA design (stage0 dot=tile_s+tile_c, stage1
//   reg_file+=dot) with ONE CSA + ONE CPA, saving one 16-bit carry-propagate adder
//   per channel per cycle (~0.1-0.16 W/core).  The register file stays BINARY
//   (16-bit), so the 16:1 read MUX is unchanged — unlike a redundant (sum,carry)
//   accumulator which would double both the register width and the read MUX.
//
//   Cost: +1 pipeline stage (result latency +1, absorbed by the sub_sum FIFO /
//   ST_RES_DELAY).  first_r still overwrites (the first tile ignores the stale
//   reg_file[batch] by forcing the CSA's first input to 0).
//
// Streaming output: batch b's sum completes after the tile_cnt==last stage-2 write;
// done_r/sub_sum_valid fire one cycle after the stage-2 write, still N_BATCH-2
// cycles before the next group's tile 0 overwrites reg_file[b].

module acc_block_reg #(
    parameter N_CH            = 256,   // channels per core (one output element each)
    parameter N_BATCH         = 16,    // blocking depth: resident activation rows
    parameter TILE_W          = 16,    // per-tile dot width (signed); tile_s/tile_c
    parameter SUM_W           = 24,    // output K=128 sum width (sign-extended)
    parameter TILES_PER_GROUP = 8,
    localparam BATCH_W        = $clog2(N_BATCH)
) (
    input  wire                              clk,
    input  wire                              rst_n,
    input  wire                              tile_valid,    // 1-cycle pulse per (tile,batch)
    input  wire                              group_start,   // k_group first-tile pulse (resync)
    input  wire                              wt_mode,       // 1=ternary(4 tiles/group) 0=W4(8 tiles/group)
    input  wire [4:0]                        b_sub,         // LIVE batches in this sub-pass
    input  wire [N_CH*TILE_W-1:0]            tile_s,        // redundant per-tile dot sum
    input  wire [N_CH*TILE_W-1:0]            tile_c,        // redundant per-tile dot carry
    output reg  [N_CH*SUM_W-1:0]             sub_sum,       // K=128 sum, N_CH lanes x SUM_W
    output reg  [BATCH_W-1:0]                sub_sum_batch, // which batch (0..N_BATCH-1)
    output reg                               sub_sum_valid
);
    localparam TILE_CNT_W = 3;   // $clog2(8) max; runtime wrap point (3 ternary / 7 W4)

    // ── tile-major counters: batch_idx (inner), tile_cnt (outer) ──
    // The counters ride the tile_valid stream and are self-aligning: one group is
    // TILES_PER_GROUP x B_sub presentations, so after a full group they wrap back to
    // (0,0) on their own.  `group_start` (the top's ctrl_tile_start, delayed to match
    // this stream) re-syncs them at a group boundary where the pipeline has run EMPTY
    // for a whole S_LOAD_ACT/S_LOAD_WGT, e.g. after a weight/activation reload.
    //
    // ── B_sub: the last sub-pass of a block may be SHORT ──────────────────────────
    // The batch wrap used to be hard-coded to N_BATCH-1 = 15, which is only correct when
    // every sub-pass holds 16 batches.  When it does not (dim_I=55 -> B_batches=28 ->
    // S_passes=2, the second sub-pass holding 12), the tile counter never advances: the
    // counter stream is 4 x 12 = 48 presentations, so batch_idx reaches 15 only after all
    // 48, `tile_last` is never seen while it matters, `first_r` stays 1 throughout, and
    // the group emits ZERO sub_sums -- the run hangs in `done` for ever.  (Measured, real
    // top at N_CH_PER=64: 16 of 28 entries, `done never asserted`; sim/tb_top_timing.v.)
    //
    // `b_sub_r` is the batch count of the group the counter stream is currently in.  It
    // is re-latched ONLY while the counters sit at a group boundary, i.e. never while a
    // presentation of the previous group is still in flight -- the same hazard #126
    // documents for `group_start`.  The live `b_sub` changes one cycle after the ctrl's
    // group boundary, which is 8 cycles before the counter stream's own wrap, so latching
    // at (0,0) always picks up the new group's value and never the old one.  The module
    // therefore needs NO cycle count of its own: the counters stay self-aligning, exactly
    // as before, and every full-length group (b_sub == 16) is bit-for-bit unchanged.
    reg [BATCH_W-1:0]    batch_idx;
    reg [TILE_CNT_W-1:0] tile_cnt;
    reg [4:0]            b_sub_r;
    wire tile_last  = (tile_cnt == (wt_mode ? 3'd3 : 3'd7));
    wire batch_last = (batch_idx == (b_sub_r - 5'd1));
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            batch_idx <= {BATCH_W{1'b0}};
            tile_cnt  <= {TILE_CNT_W{1'b0}};
        end else if (tile_valid) begin
            if (batch_last) begin
                batch_idx <= {BATCH_W{1'b0}};
                tile_cnt  <= tile_last ? {TILE_CNT_W{1'b0}} : tile_cnt + 1'b1;
            end else begin
                batch_idx <= batch_idx + 1'b1;
            end
        end else if (group_start && !(batch_last && tile_last)) begin
            // ── #126: never re-sync with a presentation still in flight ──
            // (batch_idx, tile_cnt) == (N_BATCH-1, last) is exactly the state where the
            // group's FINAL presentation has been counted but its tile_valid has not
            // arrived yet (15 of the last tile's 16 batches are in).  Zeroing the
            // counters there discards that presentation: the in-flight tile_valid then
            // computes batch 0 / tile 0, so the group loses its last sub_sum (and the
            // whole next group accumulates into the wrong batches).
            //
            // Reachable because a FIFO-backpressure stall holds the group's last
            // presentation for K cycles, which is a K-cycle hole in the tile_valid
            // stream; when K >= 6 that hole covers the moment this resync fires
            // (group_start lands 2 cycles after the NEXT group's first presentation,
            // which is still inside the hole).  Measured on dim_I=256: one lost pulse
            // per k_group, 508 of 512 => `all_written` never satisfied (sim/tb_producer_probe.v).
            // Skipping the reset here is correct: the counters are already in phase, the
            // pending tile_valid wraps them to (0,0) as it always does at a group end.
            batch_idx <= {BATCH_W{1'b0}};
            tile_cnt  <= {TILE_CNT_W{1'b0}};
        end
    end

    // Latch the live batch count at the two moments where the counter stream is at a
    // group boundary, and ONLY there:
    //   * while the counters sit at (0,0) -- the window before a group's first
    //     tile_valid, so a group's FIRST presentation already uses its own B_sub (a
    //     whole short run is 4 presentations, so being one late loses the group);
    //   * on the group's LAST presentation (`batch_last && tile_last`) -- the same edge
    //     on which the counters wrap, so the next group's value is in place before its
    //     first tile_valid.  The live b_sub has already changed by then (the ctrl
    //     advances sub_pass one cycle after the group boundary, ~9 cycles before this
    //     stream's own wrap).
    // Measured without the second term: dim_I=33 (B_batches=17 -> B_sub = 16 then 1)
    // produced 16 of 17 entries and never reached `done`.
    wire bsub_at_end   = tile_valid && batch_last && tile_last;
    wire bsub_at_start = (batch_idx == {BATCH_W{1'b0}}) &&
                         (tile_cnt  == {TILE_CNT_W{1'b0}});
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) b_sub_r <= 5'd16;
        else if (bsub_at_end || bsub_at_start) b_sub_r <= b_sub;
    end

    // ── stage 0: latch the redundant dot (tile_s, tile_c) + control tags ──
    reg                 valid_r;
    reg [BATCH_W-1:0]   batch_r;
    reg                 first_r;
    reg                 last_r;
    reg signed [TILE_W-1:0] ts_r [0:N_CH-1];
    reg signed [TILE_W-1:0] tc_r [0:N_CH-1];
    integer c0;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_r <= 1'b0;
            batch_r <= {BATCH_W{1'b0}};
            first_r <= 1'b0;
            last_r  <= 1'b0;
            for (c0 = 0; c0 < N_CH; c0 = c0 + 1) begin
                ts_r[c0] <= {TILE_W{1'b0}};
                tc_r[c0] <= {TILE_W{1'b0}};
            end
        end else if (tile_valid) begin
            valid_r <= 1'b1;
            batch_r <= batch_idx;
            first_r <= (tile_cnt == {TILE_CNT_W{1'b0}});
            last_r  <= (tile_cnt == (wt_mode ? 3'd3 : 3'd7));
            for (c0 = 0; c0 < N_CH; c0 = c0 + 1) begin
                ts_r[c0] <= tile_s[c0*TILE_W +: TILE_W];
                tc_r[c0] <= tile_c[c0*TILE_W +: TILE_W];
            end
        end else begin
            valid_r <= 1'b0;
        end
    end

    // binary register file (declared here so the stage-1 CSA can read it)
    reg signed [TILE_W-1:0] reg_file [0:N_BATCH-1][0:N_CH-1];

    // ── stage 1: 3:2 CSA reduce (first_r ? 0 : reg_file[batch_r]) + ts_r + tc_r ──
    //   sum_r + carry_r = (first_r ? 0 : reg_file[batch_r]) + tile_s + tile_c
    //   carry is pre-shifted (csa3 convention), so the CPA in stage 2 just adds.
    reg                 valid_rr;
    reg [BATCH_W-1:0]   batch_rr;
    reg                 first_rr;
    reg                 last_rr;
    reg signed [TILE_W-1:0] sum_r   [0:N_CH-1];
    reg signed [TILE_W-1:0] carry_r [0:N_CH-1];
    genvar ch;
    wire signed [TILE_W-1:0] ns [0:N_CH-1];
    wire signed [TILE_W-1:0] nc [0:N_CH-1];
    wire signed [TILE_W-1:0] resolved [0:N_CH-1];   // stage-2 CPA result (also the bypass source)

    // ── stage 1 → stage 2 forwarding (B_sub == 1, #143) ──────────────────────────
    // The register file is read in stage 1 and written in stage 2, one cycle later, so
    // the stage-1 read of `reg_file[batch_r]` sees a value that is two presentations old
    // whenever this batch's PREVIOUS presentation is still in flight.  That is exactly
    // the case when the sub-pass holds ONE batch: `batch` is the inner loop, so B_sub==1
    // presents the SAME batch on TILES_PER_GROUP consecutive cycles (4 ternary / 8 W4)
    // and the chain loses every other tile's dot.
    //
    //     presentation p   stage-1 read at p+1     write of p lands at end of p+2
    //     -> the read is 1 cycle too early whenever the same batch repeats inside 2 cycles
    //
    // The header of sim/tb_acc_bsub.v described this hazard from the start, but its
    // driver put 5 idle cycles between presentations, so the window never opened and the
    // harness stayed green for two rounds (2026-09-28: phase 2 of that harness drives the
    // SHIPPING cadence and fails B_sub=1 with got=400/expected=600).
    //
    // Fix: forward the in-flight stage-2 value when it targets the same batch.  `resolved`
    // is combinational off the stage-1 outputs, so it is available in the very cycle the
    // stage-2 write commits — exactly the value the stage-1 read is missing.
    // `first_r` still wins: the group's first tile must restart the accumulation, even
    // if the previous group's last write to the same batch is the one in flight.
    wire fwd_same_batch = valid_rr && (batch_rr == batch_r);

    generate
        for (ch = 0; ch < N_CH; ch = ch + 1) begin : gacc
            wire signed [TILE_W-1:0] a = first_r ? {TILE_W{1'b0}}
                                             : (fwd_same_batch ? resolved[ch]
                                                               : reg_file[batch_r][ch]);
            csa3 #(.W(TILE_W)) u_csa (.a(a), .b(ts_r[ch]), .c(tc_r[ch]), .s(ns[ch]), .co(nc[ch]));
            assign resolved[ch] = sum_r[ch] + carry_r[ch];
        end
    endgenerate
    integer c1;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_rr <= 1'b0;
            batch_rr <= {BATCH_W{1'b0}};
            first_rr <= 1'b0;
            last_rr  <= 1'b0;
            for (c1 = 0; c1 < N_CH; c1 = c1 + 1) begin
                sum_r[c1]   <= {TILE_W{1'b0}};
                carry_r[c1] <= {TILE_W{1'b0}};
            end
        end else if (valid_r) begin
            valid_rr <= 1'b1;
            batch_rr <= batch_r;
            first_rr <= first_r;
            last_rr  <= last_r;
            for (c1 = 0; c1 < N_CH; c1 = c1 + 1) begin
                sum_r[c1]   <= ns[c1];
                carry_r[c1] <= nc[c1];
            end
        end else begin
            valid_rr <= 1'b0;
        end
    end

    // ── stage 2: CPA resolve + binary register-file write ──
    reg               done_r;
    reg [BATCH_W-1:0] done_batch;
    // `resolved[]` is declared and driven in the stage-1 generate above: the B_sub==1
    // forwarding path needs it one stage earlier than the write itself does.
    integer b2, c2;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (b2 = 0; b2 < N_BATCH; b2 = b2 + 1)
                for (c2 = 0; c2 < N_CH; c2 = c2 + 1)
                    reg_file[b2][c2] <= {TILE_W{1'b0}};
            done_r     <= 1'b0;
            done_batch <= {BATCH_W{1'b0}};
        end else if (valid_rr) begin
            for (c2 = 0; c2 < N_CH; c2 = c2 + 1)
                reg_file[batch_rr][c2] <= resolved[c2];
            done_r     <= last_rr;
            done_batch <= batch_rr;
        end else begin
            done_r <= 1'b0;
        end
    end

    // ── stream output: 1 cycle after the final accumulation is written ──
    integer c3;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            sub_sum       <= {N_CH*SUM_W{1'b0}};
            sub_sum_batch <= {BATCH_W{1'b0}};
            sub_sum_valid <= 1'b0;
        end else begin
            sub_sum_valid <= done_r;
            sub_sum_batch <= done_batch;
            for (c3 = 0; c3 < N_CH; c3 = c3 + 1)
                sub_sum[c3*SUM_W +: SUM_W] <=
                    {{(SUM_W-TILE_W){reg_file[done_batch][c3][TILE_W-1]}},
                     reg_file[done_batch][c3]};
        end
    end
endmodule
