// lut_compute_unit_ternary.v — dual-mask LUT for ternary weights (K=32, 8 groups).
//
// Unlike the W4 bit-serial Horner, ternary weights {-1,0,+1} are computed
// directly: for each group of 4 activations, dot[g] = subset_sum[P[g]] -
// subset_sum[N[g]] (2 subset-sum reads + 1 subtract, no shift, no carry-save
// chain).  The 8 group dots are reduced by a carry-save tree to one redundant
// per-tile dot (tile_s, tile_c) that acc_block_reg accumulates across 4 tiles
// (K=128).
//
// Latency = 1 (dot register) + 2 (tree A/B register) = 3 cycles.
`include "sparse_ternary_params.vh"

module lut_compute_unit_ternary #(
    parameter N_G = 8,                    // groups per tile (2xK)
    parameter TILE_W = 16                 // per-tile dot width (signed)
) (
    input  wire                              clk,
    input  wire                              rst_n,
    input  wire                              compute_en,   // 1 pulse per tile
    input  wire [N_G*8-1:0]                  weight,       // 32 x 2-bit ternary
    input  wire [N_G*112-1:0]                group_partial_sums, // N_G x 16 x 7-bit
    output reg  signed [TILE_W-1:0]          tile_s,
    output reg  signed [TILE_W-1:0]          tile_c,
    output reg                               tile_valid
);
    // ── decode to P/N masks (combinational) ──
    wire [N_G*4-1:0] p_mask, n_mask;
    weight_ternary_decode #(.N_G(N_G)) u_dec (
        .weight(weight), .p_mask(p_mask), .n_mask(n_mask)
    );

    // ── per-group dot: subset_sum[P] - subset_sum[N], sign-extended to 16b ──
    wire signed [TILE_W-1:0] dot [0:N_G-1];
    genvar g;
    generate
        for (g = 0; g < N_G; g = g + 1) begin : gdot
            wire [3:0]  pm = p_mask[g*4 +: 4];
            wire [3:0]  nm = n_mask[g*4 +: 4];
            wire [6:0]  sp = group_partial_sums[g*112 + pm*7 +: 7];
            wire [6:0]  sn = group_partial_sums[g*112 + nm*7 +: 7];
            wire [7:0]  d  = {{1{sp[6]}}, sp} - {{1{sn[6]}}, sn};  // 8-bit signed
            assign dot[g] = {{(TILE_W-8){d[7]}}, d};
        end
    endgenerate

    // ── register the dots + valid (off the critical path) ──
    reg signed [TILE_W-1:0] dot_r [0:N_G-1];
    reg                     v_r;
    integer c0;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            v_r <= 1'b0;
            for (c0 = 0; c0 < N_G; c0 = c0 + 1) dot_r[c0] <= {TILE_W{1'b0}};
        end else begin
            v_r <= compute_en;
            for (c0 = 0; c0 < N_G; c0 = c0 + 1) dot_r[c0] <= dot[c0];
        end
    end

    // ── tree: 8 dots -> 2 redundant (s, c).  Same 2-level CSA structure as
    //    lut_compute_unit_tile's tree (8 numbers -> 4 -> register -> 2). ──
    wire [TILE_W-1:0] d0=dot_r[0], d1=dot_r[1], d2=dot_r[2], d3=dot_r[3];
    wire [TILE_W-1:0] d4=dot_r[4], d5=dot_r[5], d6=dot_r[6], d7=dot_r[7];
    // tree A, level 1: 8 -> 6
    wire [TILE_W-1:0] l1_0s,l1_0c,l1_1s,l1_1c,l1_2s,l1_2c;
    csa3 #(.W(TILE_W)) u_t0 (.a(d0),.b(d1),.c(d2),.s(l1_0s),.co(l1_0c));
    csa3 #(.W(TILE_W)) u_t1 (.a(d3),.b(d4),.c(d5),.s(l1_1s),.co(l1_1c));
    csa3 #(.W(TILE_W)) u_t2 (.a(d6),.b(d7),.c({TILE_W{1'b0}}),.s(l1_2s),.co(l1_2c));
    // tree A, level 2: 6 -> 4
    wire [TILE_W-1:0] l2_0s,l2_0c,l2_1s,l2_1c;
    csa3 #(.W(TILE_W)) u_t3 (.a(l1_0s),.b(l1_0c),.c(l1_1s),.s(l2_0s),.co(l2_0c));
    csa3 #(.W(TILE_W)) u_t4 (.a(l1_1c),.b(l1_2s),.c(l1_2c),.s(l2_1s),.co(l2_1c));
    reg signed [TILE_W-1:0] ta0,ta1,ta2,ta3;
    reg                     vA;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin ta0<=16'sd0; ta1<=16'sd0; ta2<=16'sd0; ta3<=16'sd0; vA<=1'b0; end
        else begin
            ta0<=l2_0s; ta1<=l2_0c; ta2<=l2_1s; ta3<=l2_1c;
            vA<=v_r;
        end
    end
    // tree B: 4 -> 2
    wire [TILE_W-1:0] tb0s,tb0c,tb1s,tb1c;
    csa3 #(.W(TILE_W)) u_t5 (.a(ta0),.b(ta1),.c(ta2),.s(tb0s),.co(tb0c));
    csa3 #(.W(TILE_W)) u_t6 (.a(tb0s),.b(tb0c),.c(ta3),.s(tb1s),.co(tb1c));
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin tile_s<=16'sd0; tile_c<=16'sd0; tile_valid<=1'b0; end
        else begin tile_s<=tb1s; tile_c<=tb1c; tile_valid<=vA; end
    end
endmodule
