// acc_block_reg_fifo.v — tile-major register-blocked accumulator, FIFO (shift-register) variant
//
// Functionally identical to acc_block_reg.v (3-stage CSA+CPA, binary accumulator,
// runtime TILES_PER_GROUP via wt_mode), but replaces the register file + 16:1 read
// MUX with a 16-deep BARREL (shift register):
//
//   reg_file[batch_r][ch]  (16:1 MUX read, ~61K MUX2 + ~64K feedback MUX2 for N_CH=256)
//     ->  barrel[0][ch]    (head is always the "current batch", no MUX)
//
// The batch schedule is FIXED (batch_idx cycles 0->1->...->15->0->...), so the
// accumulator for batch b is always at the head when batch b is being processed.
// Every cycle (valid_r) the barrel rotates one position: barrel[i] <= barrel[i+1],
// and the resolved CPA output (one cycle behind, i.e. the PREVIOUS batch's update)
// is inserted at the tail.  After N_BATCH shifts the update returns to the head,
// exactly when that batch is next read.
//
// Trade-off: eliminates the 16:1 read MUX (area + comb power) but clocks/shifts
// all N_BATCH*N_CH*TILE_W register bits every cycle (16x the register data
// movement of the reg_file, whose DFFs are 1/N_BATCH active).  Net area/power is
// measured by synthesis + report_power (see results/reuse_horner).
//
// Streaming output is also simpler: the final sum is the stage-2 CPA output
// (resolved) at the last tile, captured directly — no done_batch read MUX.

module acc_block_reg_fifo #(
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
    input  wire [N_CH*TILE_W-1:0]            tile_s,        // redundant per-tile dot sum
    input  wire [N_CH*TILE_W-1:0]            tile_c,        // redundant per-tile dot carry
    output reg  [N_CH*SUM_W-1:0]             sub_sum,       // K=128 sum, N_CH lanes x SUM_W
    output reg  [BATCH_W-1:0]                sub_sum_batch, // which batch (0..N_BATCH-1)
    output reg                               sub_sum_valid
);
    localparam TILE_CNT_W = 3;   // $clog2(8) max; runtime wrap point (3 ternary / 7 W4)

    // ── tile-major counters: batch_idx (inner), tile_cnt (outer) ──
    reg [BATCH_W-1:0]    batch_idx;
    reg [TILE_CNT_W-1:0] tile_cnt;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            batch_idx <= {BATCH_W{1'b0}};
            tile_cnt  <= {TILE_CNT_W{1'b0}};
        end else if (tile_valid) begin
            if (batch_idx == N_BATCH-1) begin
                batch_idx <= {BATCH_W{1'b0}};
                tile_cnt  <= (tile_cnt == (wt_mode ? 3'd3 : 3'd7))
                             ? {TILE_CNT_W{1'b0}} : tile_cnt + 1'b1;
            end else begin
                batch_idx <= batch_idx + 1'b1;
            end
        end else if (group_start) begin
            batch_idx <= {BATCH_W{1'b0}};
            tile_cnt  <= {TILE_CNT_W{1'b0}};
        end
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

    // ── BARREL: (N_BATCH-1)-deep shift register (head = barrel[0] = current batch) ──
    // Only N_BATCH-1 physical positions: the 16th accumulator is "in flight" in the
    // CSA+CPA pipeline (read at stage-1, resolved at stage-2, 1 cycle later).  So the
    // resolved value is written to barrel[N_BATCH-2] (not the tail of a full N_BATCH
    // barrel), and reaches the head exactly N_BATCH cycles after its read.
    // (declared before stage-1 so the CSA can read barrel[0])
    reg signed [TILE_W-1:0] barrel [0:N_BATCH-2][0:N_CH-1];

    // ── stage 1: 3:2 CSA reduce (first_r ? 0 : barrel[0]) + ts_r + tc_r ──
    //   sum_r + carry_r = (first_r ? 0 : barrel[0]) + tile_s + tile_c
    reg                 valid_rr;
    reg [BATCH_W-1:0]   batch_rr;
    reg                 first_rr;
    reg                 last_rr;
    reg signed [TILE_W-1:0] sum_r   [0:N_CH-1];
    reg signed [TILE_W-1:0] carry_r [0:N_CH-1];
    genvar ch;
    wire signed [TILE_W-1:0] ns [0:N_CH-1];
    wire signed [TILE_W-1:0] nc [0:N_CH-1];
    generate
        for (ch = 0; ch < N_CH; ch = ch + 1) begin : gacc
            wire signed [TILE_W-1:0] a = first_r ? {TILE_W{1'b0}} : barrel[0][ch];
            csa3 #(.W(TILE_W)) u_csa (.a(a), .b(ts_r[ch]), .c(tc_r[ch]), .s(ns[ch]), .co(nc[ch]));
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

    // ── stage 2: CPA resolve (combinational) + barrel shift ──
    //   resolved = sum_r + carry_r is the binary accumulator update for the batch
    //   that was read ONE cycle ago (the CSA's a-input).  On valid_r the barrel
    //   rotates one position and inserts `resolved` at the tail; N_BATCH shifts
    //   later it returns to the head, exactly when that batch is next read.
    wire signed [TILE_W-1:0] resolved [0:N_CH-1];
    genvar ch2;
    generate
        for (ch2 = 0; ch2 < N_CH; ch2 = ch2 + 1) begin : gres
            assign resolved[ch2] = sum_r[ch2] + carry_r[ch2];
        end
    endgenerate
    integer b2, c2;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (b2 = 0; b2 < N_BATCH-1; b2 = b2 + 1)
                for (c2 = 0; c2 < N_CH; c2 = c2 + 1)
                    barrel[b2][c2] <= {TILE_W{1'b0}};
        end else if (valid_r) begin
            for (b2 = 0; b2 < N_BATCH-2; b2 = b2 + 1)
                for (c2 = 0; c2 < N_CH; c2 = c2 + 1)
                    barrel[b2][c2] <= barrel[b2+1][c2];
            for (c2 = 0; c2 < N_CH; c2 = c2 + 1)
                barrel[N_BATCH-2][c2] <= resolved[c2];
        end
    end

    // ── stream output: capture the final sum (resolved at last tile) 1 cycle before valid ──
    reg                 done_r;
    reg [BATCH_W-1:0]   done_batch;
    reg signed [SUM_W-1:0] sub_sum_r [0:N_CH-1];
    integer c3;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (c3 = 0; c3 < N_CH; c3 = c3 + 1) sub_sum_r[c3] <= {SUM_W{1'b0}};
            done_r     <= 1'b0;
            done_batch <= {BATCH_W{1'b0}};
        end else if (valid_rr) begin
            for (c3 = 0; c3 < N_CH; c3 = c3 + 1)
                sub_sum_r[c3] <= {{(SUM_W-TILE_W){resolved[c3][TILE_W-1]}}, resolved[c3]};
            done_r     <= last_rr;
            done_batch <= batch_rr;
        end else begin
            done_r <= 1'b0;
        end
    end

    integer c4;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            sub_sum       <= {N_CH*SUM_W{1'b0}};
            sub_sum_batch <= {BATCH_W{1'b0}};
            sub_sum_valid <= 1'b0;
        end else begin
            sub_sum_valid <= done_r;
            sub_sum_batch <= done_batch;
            for (c4 = 0; c4 < N_CH; c4 = c4 + 1)
                sub_sum[c4*SUM_W +: SUM_W] <= sub_sum_r[c4];
        end
    end
endmodule
