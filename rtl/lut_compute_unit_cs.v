// lut_compute_unit_cs.v — Carry-save (redundant) FULL-CHAIN variant of lut_compute_unit
//
// Same interface and semantics as lut_compute_unit.v, but replaces every
// carry-propagate adder (carry-select 16-bit in the Horner, ripple/carry-select
// in the final accumulation tree) with a 3:2 carry-save compressor (csa3).
//
// The whole dot-product path is carry-save:
//   Horner (W stages)      : acc kept as (sum, carry); each stage = 1 CSA level.
//   Final tree (2 stages)  : 4 groups x (sum,carry) = 8 numbers -> 2 (Wallace).
//   Group accumulate       : (acc_s, acc_c) += (tile_s, tile_c) in redundant form.
//   CPA (2 stages)         : the ONLY carry-propagate, once per group (8 tiles),
//                             split 12+12 like the original sub_lo/sub_hi.
//
// Critical path = 16:1 subset-sum MUX (~0.22ns) + 1 CSA level (~0.1ns) ≈ 0.32ns,
// vs 0.945ns for the carry-select Horner.  The final CPA is off the per-tile
// steady-state path (pipelined, fires at the group boundary).
//
// Latency = ST_PIPE_WEIGHT + ST_PIPE_BITS + 5 (treeA + treeB + acc + CPA lo + CPA hi)
//         = 1 + 4 + 5 = 10 (vs 9 for lut_compute_unit).
`include "sparse_ternary_params.vh"

module lut_compute_unit_cs (
    input  wire                                 clk,
    input  wire                                 rst_n,
    input  wire                                 tile_start,   // unused
    input  wire                                 compute_en,   // 1 拍推 1 tile
    input  wire                                 weight_signed, // 1=有符号(-8..7)/0=无符号(0..15)，门控 Horner stage-0 取反
    input  wire [`ST_WT_PER_TILE*`ST_WT_BITS-1:0] weight,     // 16 权重 x 4b (小端序)
    input  wire [`ST_PIPE_SUB*`ST_PIPE_BITS*`ST_BUILD_TOTAL_REGS-1:0] group_partial_sums,  // W delayed subset-sums
    output wire [`ST_SUBSUM_WIDTH-1:0]          sub_sum_out,
    output wire                                 sub_sum_valid
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

    // tile_count: free-running 0..7, increments once per tile (on v_t).
    reg [2:0] tile_count;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) tile_count <= 3'd0;
        else if (v_t) tile_count <= tile_count + 1'b1;
    end
    wire first_t = v_t && (tile_count == 3'd0);
    wire last_t  = v_t && (tile_count == `ST_TILES_PER_GROUP - 1);

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
    reg               vA, fA, lA;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin ta0 <= 16'sd0; ta1 <= 16'sd0; ta2 <= 16'sd0; ta3 <= 16'sd0; vA <= 1'b0; fA <= 1'b0; lA <= 1'b0; end
        else begin
            ta0 <= l2_0s; ta1 <= l2_0c; ta2 <= l2_1s; ta3 <= l2_1c;
            vA <= v_t; fA <= first_t; lA <= last_t;
        end
    end

    // ── Final tree stage B: 4 numbers -> 2 (2 CSA levels, ~0.2ns) ──
    wire [15:0] tb0s, tb0c, tb1s, tb1c;
    csa3 #(.W(16)) u_t5 (.a(ta0), .b(ta1), .c(ta2),     .s(tb0s), .co(tb0c));
    csa3 #(.W(16)) u_t6 (.a(tb0s), .b(tb0c), .c(ta3),   .s(tb1s), .co(tb1c));
    reg signed [15:0] tile_s, tile_c;
    reg               vB, fB, lB;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin tile_s <= 16'sd0; tile_c <= 16'sd0; vB <= 1'b0; fB <= 1'b0; lB <= 1'b0; end
        else begin
            tile_s <= tb1s; tile_c <= tb1c;
            vB <= vA; fB <= fA; lB <= lA;
        end
    end

    // ── Group accumulate: (acc_s, acc_c) += (tile_s, tile_c) in redundant 24-bit ──
    // 4 numbers -> 2 via 2 CSA levels (~0.2ns).  First tile overwrites (fresh group).
    wire signed [23:0] tile_s_e = {{8{tile_s[15]}}, tile_s};
    wire signed [23:0] tile_c_e = {{8{tile_c[15]}}, tile_c};
    reg signed [23:0] acc_s, acc_c;
    reg               vC, fC, lC;
    wire [23:0] ac0s, ac0c, ac1s, ac1c;
    csa3 #(.W(24)) u_a0 (.a(acc_s), .b(acc_c), .c(tile_s_e), .s(ac0s), .co(ac0c));
    csa3 #(.W(24)) u_a1 (.a(ac0s), .b(ac0c), .c(tile_c_e), .s(ac1s), .co(ac1c));
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin acc_s <= 24'sd0; acc_c <= 24'sd0; vC <= 1'b0; fC <= 1'b0; lC <= 1'b0; end
        else begin
            if (vB) begin
                if (fB) begin acc_s <= tile_s_e; acc_c <= tile_c_e; end   // first tile: start fresh
                else    begin acc_s <= ac1s;     acc_c <= ac1c;     end   // else: accumulate
            end
            vC <= vB; fC <= fB; lC <= lB;
        end
    end

    // ── CPA (group boundary): sub_sum = acc_s + acc_c, split 12+12 ──
    // Stage 1: low 12-bit add (unsigned) + capture high halves before overwrite.
    wire [12:0] lo = {1'b0, acc_s[11:0]} + {1'b0, acc_c[11:0]};
    reg [11:0]        sub_lo;
    reg               sub_carry;
    reg signed [11:0] hi_a, hi_b;
    reg               vD, fD, lD;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin sub_lo <= 12'd0; sub_carry <= 1'b0; hi_a <= 12'sd0; hi_b <= 12'sd0; vD <= 1'b0; fD <= 1'b0; lD <= 1'b0; end
        else begin
            if (vC) begin
                sub_lo    <= lo[11:0];
                sub_carry <= lo[12];
                hi_a      <= acc_s[23:12];
                hi_b      <= acc_c[23:12];
            end
            vD <= vC; fD <= fC; lD <= lC;
        end
    end

    // ── CPA stage 2: high 12-bit add (signed) + carry-in; result + valid pulse ──
    reg signed [11:0] sub_hi;
    reg [11:0]        sub_lo_d;
    reg               grp_done_r;
    wire [11:0] hi = hi_a + hi_b + {11'd0, sub_carry};
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin sub_hi <= 12'sd0; sub_lo_d <= 12'd0; grp_done_r <= 1'b0; end
        else begin
            if (vD) sub_hi <= hi;
            sub_lo_d <= sub_lo;
            grp_done_r <= vD && lD;
        end
    end

    assign sub_sum_out   = {sub_hi, sub_lo_d};
    assign sub_sum_valid = grp_done_r;

endmodule
