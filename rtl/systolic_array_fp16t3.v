// systolic_array_fp16t3.v — N×M systolic array of FP16T3 PEs
// 1-cycle full-array weight preload: 512b bus = 256 × 2b ternary weights.
// Activation flow left→right (FP16), partial sum flow top→bottom (FP32).

`include "sparsenn_params.vh"

module systolic_array_fp16t3 #(
    parameter N = `N_PE_ROWS,
    parameter M = `M_PE_COLS
) (
    input  wire                             clk,
    input  wire                             rst_n,

    input  wire                             weight_load,
    input  wire                             accum_clear,
    input  wire                             stall,

    // Activation input (left edge, N rows × FP16)
    input  wire [N*`FP16_WIDTH-1:0]         activations_in,

    // Weight input — full array in 1 cycle (N×M×2b = 512b)
    input  wire [N*M*2-1:0]                 weights_in,

    // Partial sum input (top edge, M columns × FP32)
    input  wire [M*`FP32_WIDTH-1:0]         partial_sums_in,

    // Partial sum output (bottom edge, M columns × FP32)
    output wire [M*`FP32_WIDTH-1:0]         partial_sums_out
);

    genvar i, j;

    wire [`FP16_WIDTH-1:0] act_wire [0:N-1][0:M-1];
    wire [`FP32_WIDTH-1:0]  ps_wire  [0:N-1][0:M-1];

    generate
        for (i = 0; i < N; i = i + 1) begin : row
            for (j = 0; j < M; j = j + 1) begin : col

                wire [`FP16_WIDTH-1:0] act_in_pe;
                wire [`FP32_WIDTH-1:0]  ps_in_pe;
                wire [1:0]              wt_in_pe;

                // Activation: left edge or from left neighbor
                if (j == 0)
                    assign act_in_pe = activations_in[i*`FP16_WIDTH +: `FP16_WIDTH];
                else
                    assign act_in_pe = act_wire[i][j-1];

                // Partial sum: top edge or from top neighbor
                if (i == 0)
                    assign ps_in_pe = partial_sums_in[j*`FP32_WIDTH +: `FP32_WIDTH];
                else
                    assign ps_in_pe = ps_wire[i-1][j];

                // Weight: unique 2b per PE from flat 512b bus
                assign wt_in_pe = weights_in[(i*M + j)*2 +: 2];

                pe_fp16t3 pe (
                    .clk(clk), .rst_n(rst_n),
                    .weight_load(weight_load), .accum_clear(accum_clear), .stall(stall),
                    .activation_in(act_in_pe), .weight_in(wt_in_pe),
                    .partial_sum_in(ps_in_pe),
                    .activation_out(act_wire[i][j]),
                    .partial_sum_out(ps_wire[i][j])
                );

            end
        end
    endgenerate

    generate
        for (j = 0; j < M; j = j + 1) begin : ps_out_col
            assign partial_sums_out[j*`FP32_WIDTH +: `FP32_WIDTH] = ps_wire[N-1][j];
        end
    endgenerate

endmodule
