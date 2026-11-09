// weight_ternary_decode.v — decode 32 x 2-bit ternary weights into 8 groups'
// positive (P) and negative (N) 4-bit subset-sum selection masks.
//
// Ternary weight w = {sign, mag}: +1={0,1}, -1={1,1}, 0={0,0} (and the unused
// {1,0} decodes to neither).  For each group g of 4 weights:
//   P[g][i] = mag & ~sign   (weight i is +1)
//   N[g][i] = mag &  sign   (weight i is -1)
// `P` selects the subset-sum of the +1 activations, `N` the -1 activations;
// dot[g] = subset_sum[g][P[g]] - subset_sum[g][N[g]].
module weight_ternary_decode #(
    parameter N_G = 8            // groups per tile (2xK = 32 = 8 x 4)
) (
    input  wire [N_G*8-1:0] weight,       // 32 x 2-bit, little-endian per weight
    output wire [N_G*4-1:0]  p_mask,      // N_G x 4-bit positive masks
    output wire [N_G*4-1:0]  n_mask       // N_G x 4-bit negative masks
);
    genvar g;
    generate
        for (g = 0; g < N_G; g = g + 1) begin : gdec
            wire [7:0] wg = weight[g*8 +: 8];       // 4 weights x 2-bit
            wire [3:0] mag = {wg[6], wg[4], wg[2], wg[0]};
            wire [3:0] sgn = {wg[7], wg[5], wg[3], wg[1]};
            assign p_mask[g*4 +: 4] = mag & ~sgn;
            assign n_mask[g*4 +: 4] = mag &  sgn;
        end
    endgenerate
endmodule
