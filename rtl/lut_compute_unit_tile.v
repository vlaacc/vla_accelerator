// lut_compute_unit_tile.v — Tile-major variant of the carry-save LUT compute unit.
//
// Truncated from lut_compute_unit_cs.v: keeps everything up to and including
// final tree stage B (which emits the per-tile redundant dot product as
// (tile_s, tile_c)), and drops the 8-tile group accumulate + CPA.  The
// tile-major dataflow owns the group accumulation in an external register-block
// accumulator (rtl/acc_block_reg.v), so this unit only needs to stream one
// redundant per-tile dot per cycle.
//
// Data path is identical to lut_compute_unit_cs.v up to tree B:
//   S0 weight register (weight_r) -> weight shift register wd[0..W-1]
//   W-stage MSB-first Horner carry-save (hsum, hcar)
//   tree A (8 -> 4) -> tree B (4 -> 2) => (tile_s, tile_c)
//
// Removed vs lut_compute_unit_cs.v: tile_count/first_t/last_t counters, the
// fA/lA/fB/lB first/last flags (only consumed by the removed group-accumulate
// + CPA stages), the redundant group accumulator (acc_s/acc_c + csa3 u_a0/u_a1),
// and the two-stage CPA (sub_lo/sub_carry/hi_a/hi_b/sub_hi/sub_lo_d/grp_done_r).
//
// Latency = ST_PIPE_WEIGHT + ST_PIPE_BITS + 2 (treeA + treeB) = 1 + 4 + 2 = 7.
`include "sparse_ternary_params.vh"

module lut_compute_unit_tile (
    input  wire                                 clk,
    input  wire                                 rst_n,
    input  wire                                 compute_en,   // 1 拍推 1 tile
    input  wire                                 weight_signed, // 1=有符号(-8..7)/0=无符号(0..15)，门控 Horner stage-0 取反
    input  wire [`ST_WT_PER_TILE*`ST_WT_BITS-1:0] weight,     // 16 权重 x 4b (小端序)
    input  wire [`ST_PIPE_SUB*`ST_PIPE_BITS*`ST_BUILD_TOTAL_REGS-1:0] group_partial_sums,  // W delayed subset-sums
    output reg signed [15:0]                    tile_s,      // 冗余 per-tile 点积 sum (16-bit 有符号)
    output reg signed [15:0]                    tile_c,      // 冗余 per-tile 点积 carry
    output reg                                  tile_valid   // 每 tile 一拍脉冲
);
    localparam W   = `ST_WT_BITS;            // 4
    localparam N_G = `ST_N_GROUPS_PER_TILE;  // 4

    // ── S0: weight register ──
    reg [`ST_WT_PER_TILE*W-1:0] weight_r;
    reg                         v_w;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) weight_r <= 0;
        else        weight_r <= weight;
    end
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) v_w <= 1'b0;
        else        v_w <= compute_en;
    end

    // ── Weight shift register: wd[S] = weight_r delayed S (S=0..W-1) ──
    wire [`ST_WT_PER_TILE*W-1:0] wd [0:W-1];
    assign wd[0] = weight_r;
    genvar d;
    generate
        for (d = 1; d < W; d = d + 1) begin : wdel
            reg [`ST_WT_PER_TILE*W-1:0] r;
            always @(posedge clk or negedge rst_n) begin
                if (!rst_n) r <= 0;
                else        r <= wd[d-1];
            end
            assign wd[d] = r;
        end
    endgenerate

    // ── W-stage MSB-first Horner, carry-save accumulator ──
    //   acc[g][S] = (acc[g][S-1] << 1) + slice[S], kept as (hsum, hcar):
    //   acc[g][S] = hsum[g][S] + hcar[g][S]  (hcar already shifted-left by the CSA)
    //   slice[S]  = subset_sum[mask[S]]  (mask[S] = weight bit W-1-S of 4 weights)
    // Stage 0 (MSB): acc = weight_signed ? -slice : slice  (no CSA, direct init)
    // Stage S>0:     3:2 CSA on (hsum<<1, hcar<<1, slice)
    reg signed [15:0] hsum [0:N_G-1][0:W-1];
    reg signed [15:0] hcar [0:N_G-1][0:W-1];
    genvar S, g, i;
    generate
        for (S = 0; S < W; S = S + 1) begin : bitstage
            wire [`ST_WT_PER_TILE-1:0] mask_s;
            for (i = 0; i < `ST_WT_PER_TILE; i = i + 1) begin : mi
                assign mask_s[i] = wd[S][i*W + (W-1-S)];
            end
            for (g = 0; g < N_G; g = g + 1) begin : grp
                wire [3:0]  m      = mask_s[g*4 +: 4];
                wire [6:0]  sl     = group_partial_sums[S*`ST_BUILD_TOTAL_REGS + g*`ST_BUILD_REGS_PER_GROUP + m*7 +: 7];
                wire [15:0] sl_ext = {{9{sl[6]}}, sl};
                if (S == 0) begin : s0
                    // acc[g][0] = weight_signed ? -sl_ext : sl_ext
                    //   signed:   hsum=~sl_ext, hcar=1  =>  ~sl_ext + 1 = -sl_ext
                    //   unsigned: hsum= sl_ext, hcar=0  =>   sl_ext
                    always @(posedge clk or negedge rst_n) begin
                        if (!rst_n) begin hsum[g][0] <= 16'sd0; hcar[g][0] <= 16'sd0; end
                        else begin
                            hsum[g][0] <= weight_signed ? ~sl_ext : sl_ext;
                            hcar[g][0] <= weight_signed ? 16'sd1 : 16'sd0;
                        end
                    end
                end else begin : sN
                    wire [15:0] ss   = {hsum[g][S-1][14:0], 1'b0};  // hsum << 1 (arithmetic)
                    wire [15:0] cc   = {hcar[g][S-1][14:0], 1'b0};  // hcar << 1
                    wire [15:0] nsum, ncar;
                    csa3 #(.W(16)) u_csa (.a(ss), .b(cc), .c(sl_ext), .s(nsum), .co(ncar));
                    always @(posedge clk or negedge rst_n) begin
                        if (!rst_n) begin hsum[g][S] <= 16'sd0; hcar[g][S] <= 16'sd0; end
                        else begin
                            hsum[g][S] <= nsum;
                            hcar[g][S] <= ncar;
                        end
                    end
                end
            end
        end
    endgenerate

    // ── valid chain: v_w delayed W (tracks hsum[..][W-1]) ──
    reg [W-1:0] v_sr;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) v_sr <= 0;
        else        v_sr <= {v_sr[W-2:0], v_w};
    end
    wire v_t = v_sr[W-1];

    // ── Final tree stage A: 8 numbers -> 4 (2 CSA levels, ~0.2ns) ──
    // grp_dot[g] = hsum[g][W-1] + hcar[g][W-1]; 4 groups = 8 numbers.
    wire [15:0] h0 = hsum[0][W-1], h1 = hcar[0][W-1], h2 = hsum[1][W-1], h3 = hcar[1][W-1];
    wire [15:0] h4 = hsum[2][W-1], h5 = hcar[2][W-1], h6 = hsum[3][W-1], h7 = hcar[3][W-1];
    // level 1: 8 -> 6
    wire [15:0] l1_0s, l1_0c, l1_1s, l1_1c, l1_2s, l1_2c;
    csa3 #(.W(16)) u_t0 (.a(h0), .b(h1), .c(h2),      .s(l1_0s), .co(l1_0c));
    csa3 #(.W(16)) u_t1 (.a(h3), .b(h4), .c(h5),      .s(l1_1s), .co(l1_1c));
    csa3 #(.W(16)) u_t2 (.a(h6), .b(h7), .c(16'd0),   .s(l1_2s), .co(l1_2c));
    // level 2: 6 -> 4
    wire [15:0] l2_0s, l2_0c, l2_1s, l2_1c;
    csa3 #(.W(16)) u_t3 (.a(l1_0s), .b(l1_0c), .c(l1_1s), .s(l2_0s), .co(l2_0c));
    csa3 #(.W(16)) u_t4 (.a(l1_1c), .b(l1_2s), .c(l1_2c), .s(l2_1s), .co(l2_1c));
    reg signed [15:0] ta0, ta1, ta2, ta3;
    reg               vA;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin ta0 <= 16'sd0; ta1 <= 16'sd0; ta2 <= 16'sd0; ta3 <= 16'sd0; vA <= 1'b0; end
        else begin
            ta0 <= l2_0s; ta1 <= l2_0c; ta2 <= l2_1s; ta3 <= l2_1c;
            vA <= v_t;
        end
    end

    // ── Final tree stage B: 4 numbers -> 2 (2 CSA levels, ~0.2ns) ──
    // tile_s + tile_c = grp_dot (the per-tile redundant dot product).
    // tile_valid (was vB in lut_compute_unit_cs.v) is the per-tile valid pulse.
    wire [15:0] tb0s, tb0c, tb1s, tb1c;
    csa3 #(.W(16)) u_t5 (.a(ta0), .b(ta1), .c(ta2),     .s(tb0s), .co(tb0c));
    csa3 #(.W(16)) u_t6 (.a(tb0s), .b(tb0c), .c(ta3),   .s(tb1s), .co(tb1c));
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin tile_s <= 16'sd0; tile_c <= 16'sd0; tile_valid <= 1'b0; end
        else begin
            tile_s <= tb1s; tile_c <= tb1c;
            tile_valid <= vA;
        end
    end

endmodule
