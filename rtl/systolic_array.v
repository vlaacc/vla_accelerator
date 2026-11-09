// systolic_array.v — N×M systolic array of FP16 PEs
// Weight-stationary dataflow:
//   1. Preload weights column-by-column from left
//   2. Stream activations from left edge, propagate rightward each cycle
//   3. Accumulate partial sums downward
//   4. Drain partial sums from bottom edge
//
// Parameters: N_PE_ROWS rows, M_PE_COLS columns (from sparsenn_params.vh)

`include "sparsenn_params.vh"

module systolic_array #(
    parameter N = `N_PE_ROWS,
    parameter M = `M_PE_COLS
) (
    input  wire                             clk,
    input  wire                             rst_n,

    // Control
    input  wire                             weight_load,      // Enable weight preload
    input  wire                             accum_clear,      // Clear all accumulators
    input  wire                             stall,            // Stall all PEs

    // Activation input (left edge, N rows)
    input  wire [N*`FP16_WIDTH-1:0]         activations_in,

    // Weight input (top edge, one column at a time during preload, per-ROw)
    input  wire [N*`FP16_WIDTH-1:0]         weights_in,

    // Partial sum input (top edge, M columns)
    input  wire [M*`FP32_WIDTH-1:0]         partial_sums_in,

    // Partial sum output (bottom edge, M columns)
    output wire [M*`FP32_WIDTH-1:0]         partial_sums_out   // Bottom edge
);

    genvar i, j;

    // ── Wire grid for inter-PE connections ──
    // activation_wire[i][j]: activation flowing right from PE[i][j]
    // ps_wire[i][j]: partial sum flowing down from PE[i][j]
    wire [`FP16_WIDTH-1:0] act_wire [0:N-1][0:M-1];
    wire [`FP32_WIDTH-1:0]  ps_wire  [0:N-1][0:M-1];

    generate
        for (i = 0; i < N; i = i + 1) begin : row
            for (j = 0; j < M; j = j + 1) begin : col

                wire [`FP16_WIDTH-1:0] act_in_pe;
                wire [`FP32_WIDTH-1:0]  ps_in_pe;
                wire [`FP16_WIDTH-1:0]  wt_in_pe;

                // Activation input
                if (j == 0)
                    assign act_in_pe = activations_in[i*`FP16_WIDTH +: `FP16_WIDTH];
                else
                    assign act_in_pe = act_wire[i][j-1];

                // Partial sum input
                if (i == 0)
                    assign ps_in_pe = partial_sums_in[j*`FP32_WIDTH +: `FP32_WIDTH];
                else
                    assign ps_in_pe = ps_wire[i-1][j];

                // Weight input
                assign wt_in_pe = weights_in[i*`FP16_WIDTH +: `FP16_WIDTH];

                pe_fp16 pe (
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

    // ── Output partial sums from bottom edge ──
    generate
        for (j = 0; j < M; j = j + 1) begin : ps_out_col
            assign partial_sums_out[j*`FP32_WIDTH +: `FP32_WIDTH] = ps_wire[N-1][j];
        end
    endgenerate

endmodule
