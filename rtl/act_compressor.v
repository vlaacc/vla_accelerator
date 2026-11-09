// act_compressor.v — Activation sparsity compressor (act_compress feature)
// Detects non-zero INT4 activations, counts them (balanced-tree popcount),
// priority-encodes the first K non-zero indices (one-hot -> binary, balanced),
// and gathers those K activations to the front.
// compress_en = (nz_cnt <= K).  Caller compacts activations to
// {act_c3,act_c2,act_c1,act_c0, zeros} and gathers weights by idx_0..3.
//
// Sized for the MAX tile (K=32, ternary); W4 (K=16) zero-pads the upper 16
// activations upstream, so this single compressor serves both modes (the upper
// 16 valid bits are 0 in W4 mode).
`include "sparse_ternary_params.vh"

module act_compressor #(
    parameter K = `ST_ACT_COMPRESS_K
) (
    input  wire [`ST_TILE_K_MAX * `ST_ACT_WIDTH - 1:0] activations,   // 32 x 4b
    output wire [`ST_TILE_K_MAX-1:0]                   valid,
    output wire [5:0]                                  nz_cnt,          // 0..32
    output wire [4:0]                                  idx_0, idx_1, idx_2, idx_3,
    output wire [`ST_ACT_WIDTH-1:0]                    act_c0, act_c1, act_c2, act_c3,
    output wire                                        compress_en
);
    // ── valid mask ──
    genvar g;
    generate
        for (g = 0; g < `ST_TILE_K_MAX; g = g + 1)
            assign valid[g] = |activations[g*`ST_ACT_WIDTH +: `ST_ACT_WIDTH];
    endgenerate

    // ── balanced-tree popcount (5 adder levels, not a 32-deep chain) ──
    function [5:0] popcount32;
        input [31:0] x;
        reg [1:0] l1 [0:15];
        reg [2:0] l2 [0:7];
        reg [3:0] l3 [0:3];
        reg [4:0] l4 [0:1];
        integer k;
        begin
            for (k = 0; k < 16; k = k + 1) l1[k] = x[2*k] + x[2*k+1];
            for (k = 0; k < 8;  k = k + 1) l2[k] = {1'b0, l1[2*k]} + {1'b0, l1[2*k+1]};
            for (k = 0; k < 4;  k = k + 1) l3[k] = {1'b0, l2[2*k]} + {1'b0, l2[2*k+1]};
            for (k = 0; k < 2;  k = k + 1) l4[k] = {1'b0, l3[2*k]} + {1'b0, l3[2*k+1]};
            popcount32 = {1'b0, l4[0]} + {1'b0, l4[1]};
        end
    endfunction

    assign nz_cnt = popcount32(valid);

    // ── prefix[i] = popcount of valid bits below position i ──
    wire [5:0] prefix [0:`ST_TILE_K_MAX-1];
    genvar i;
    generate
        for (i = 0; i < `ST_TILE_K_MAX; i = i + 1)
            assign prefix[i] = popcount32(valid & ((32'd1 << i) - 32'd1));
    endgenerate

    // ── idx_j = position of the (j+1)-th set bit ──
    // match[j][i] = (prefix[i]==j && valid[i]) is one-hot (exactly one i for j<nz_cnt).
    // one-hot -> binary via balanced reduction-OR (not a priority chain).
    function [4:0] oh2bin32;
        input [31:0] m;
        begin
            oh2bin32[0] = |(m & 32'hAAAAAAAA);
            oh2bin32[1] = |(m & 32'hCCCCCCCC);
            oh2bin32[2] = |(m & 32'hF0F0F0F0);
            oh2bin32[3] = |(m & 32'hFF00FF00);
            oh2bin32[4] = |(m & 32'hFFFF0000);
        end
    endfunction

    wire [31:0] match [0:K-1];
    genvar j, ii;
    generate
        for (j = 0; j < K; j = j + 1) begin : mj
            for (ii = 0; ii < `ST_TILE_K_MAX; ii = ii + 1)
                assign match[j][ii] = (prefix[ii] == j[5:0]) && valid[ii];
        end
    endgenerate

    assign idx_0 = oh2bin32(match[0]);
    assign idx_1 = oh2bin32(match[1]);
    assign idx_2 = oh2bin32(match[2]);
    assign idx_3 = oh2bin32(match[3]);

    // ── gathered activations (gated: j>=nz_cnt slots are 0) ──
    assign act_c0 = (nz_cnt > 6'd0) ? activations[idx_0*`ST_ACT_WIDTH +: `ST_ACT_WIDTH] : {`ST_ACT_WIDTH{1'b0}};
    assign act_c1 = (nz_cnt > 6'd1) ? activations[idx_1*`ST_ACT_WIDTH +: `ST_ACT_WIDTH] : {`ST_ACT_WIDTH{1'b0}};
    assign act_c2 = (nz_cnt > 6'd2) ? activations[idx_2*`ST_ACT_WIDTH +: `ST_ACT_WIDTH] : {`ST_ACT_WIDTH{1'b0}};
    assign act_c3 = (nz_cnt > 6'd3) ? activations[idx_3*`ST_ACT_WIDTH +: `ST_ACT_WIDTH] : {`ST_ACT_WIDTH{1'b0}};

    assign compress_en = (nz_cnt <= K[5:0]);

endmodule
