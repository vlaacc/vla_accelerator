// build_unit.v — 1-cycle hierarchical LUT builder
// N_G groups x 4 INT4 activations → 16 subset sums per group
// Each group: 11 adders. Total: 11*N_G adders, N_G*112b register output
//   W4 (K=16): N_G=4 (44 adders, 448b)
//   ternary (K=32): N_G=8 (88 adders, 896b)
`include "sparse_ternary_params.vh"

(* keep_hierarchy *) module build_unit #(
    parameter N_G = `ST_N_GROUPS_PER_TILE   // groups per tile (4 / 8)
) (
    input  wire                              clk,
    input  wire                              rst_n,
    input  wire                              build_en,
    input  wire [N_G*4*`ST_ACT_WIDTH - 1:0]  activations,   // N_G*4 x 4-bit
    output reg  [N_G*112 - 1:0]              group_partial_sums
);

    // ── Unpack N_G*4 activations ──
    wire signed [`ST_ACT_WIDTH-1:0] act [0:N_G*4-1];
    genvar g;
    generate
        for (g = 0; g < N_G*4; g = g + 1)
            assign act[g] = activations[g*`ST_ACT_WIDTH +: `ST_ACT_WIDTH];
    endgenerate

    // ── Per-group subset sum function (combinational) ──
    // Inputs: 4 signed INT4 values {a,b,c,d}
    // Outputs: 16 subset sums (all 2^4 mask combinations), 7-bit each
    function automatic [111:0] compute_group_sums;
        input signed [3:0] a, b, c, d;
        reg signed [4:0]  s_ab, s_ac, s_ad, s_bc, s_bd, s_cd;
        reg signed [5:0]  s_abc, s_abd, s_acd, s_bcd;
        reg signed [6:0]  s_abcd;
        reg [6:0] outs [0:15];
        integer i;
        begin
            // Level 1: sums of 2 (6 adders)
            s_ab = a + b;
            s_ac = a + c;
            s_ad = a + d;
            s_bc = b + c;
            s_bd = b + d;
            s_cd = c + d;

            // Level 2: sums of 3 (4 adders)
            s_abc = s_ab + c;
            s_abd = s_ab + d;
            s_acd = s_ac + d;
            s_bcd = s_bc + d;

            // Level 3: sum of 4 (1 adder)
            s_abcd = s_ab + s_cd;

            // Map: mask bits → output, sign-extended to 7-bit
            outs[4'b0000] = 7'd0;
            outs[4'b0001] = { {3{a[3]}}, a };              // 4b → 7b
            outs[4'b0010] = { {3{b[3]}}, b };
            outs[4'b0011] = { {2{s_ab[4]}}, s_ab };        // 5b → 7b
            outs[4'b0100] = { {3{c[3]}}, c };
            outs[4'b0101] = { {2{s_ac[4]}}, s_ac };
            outs[4'b0110] = { {2{s_bc[4]}}, s_bc };
            outs[4'b0111] = { {1{s_abc[5]}}, s_abc };      // 6b → 7b
            outs[4'b1000] = { {3{d[3]}}, d };
            outs[4'b1001] = { {2{s_ad[4]}}, s_ad };
            outs[4'b1010] = { {2{s_bd[4]}}, s_bd };
            outs[4'b1011] = { {1{s_abd[5]}}, s_abd };
            outs[4'b1100] = { {2{s_cd[4]}}, s_cd };
            outs[4'b1101] = { {1{s_acd[5]}}, s_acd };
            outs[4'b1110] = { {1{s_bcd[5]}}, s_bcd };
            outs[4'b1111] = s_abcd;                        // 7b already

            compute_group_sums = {outs[15], outs[14], outs[13], outs[12],
                                  outs[11], outs[10], outs[9],  outs[8],
                                  outs[7],  outs[6],  outs[5],  outs[4],
                                  outs[3],  outs[2],  outs[1],  outs[0]};
        end
    endfunction

    // ── Compute all N_G groups in parallel (combinational) ──
    wire [N_G*112-1:0] gs_all;
    generate
        for (g = 0; g < N_G; g = g + 1) begin : ggrp
            assign gs_all[g*112 +: 112] = compute_group_sums(act[g*4+0], act[g*4+1],
                                                             act[g*4+2], act[g*4+3]);
        end
    endgenerate

    // ── Register output ──
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            group_partial_sums <= {N_G*112{1'b0}};
        else if (build_en)
            group_partial_sums <= gs_all;
    end

`ifdef FORMAL
    // Assertion: total output width matches parameter
    initial assert ($bits(group_partial_sums) == N_G * 112);
`endif

endmodule
