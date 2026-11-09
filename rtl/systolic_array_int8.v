// systolic_array_int8.v — N_ROWS × M_COLS INT8 array, weight-stationary
//
// ── Topology: activation BROADCAST, partial sum SYSTOLIC ──
//   N_ROWS (=K)  : reduction index.  Row i holds the weights W[i][0..M-1] for one k.
//   M_COLS (=N)  : output channel index.  Column j accumulates out[j] = sum_i W[i][j]*A[i].
//   activations_in[i] is broadcast to EVERY PE in row i (not shifted right).
//   partial sums shift DOWN the column (PE(i,j) -> PE(i+1,j)), which is the only
//   systolic path.
//
// ── Why broadcast and not a right-shifting activation chain ──
// With a shift chain, PE(i,j) sees the activation 2j cycles late (pe_int8 registers
// the activation twice on its way across).  Working the recurrence out for the
// bottom edge gives
//
//     S_{N-1,j}(t) = sum_i W[i][j] * A_i(t - N - 1 + i) + partial_sums_in[j](t - N)
//
// so with the standard row-skewed feed `A_i(x) = A[x - i][i]` the result for output
// row m is C[m][j] but appears at t = m + N + 1 + 2j — i.e. each column is skewed by
// 2j relative to column 0.  Aligning 32 columns would need a ~32x62-word de-skew
// buffer (~0.6 mm^2 of flops per 4 meshes).  Broadcasting costs nothing and makes the
// skew identically zero:
//
//     S_{N-1,j}(t) = sum_i W[i][j] * A_i(t - N - 1 + i) + partial_sums_in[j](t - N)
//                  = C[t - N - 1][j]      for A_i(x) = A[x - i][i]
//
// Every column now emits output row m in the SAME cycle, so a 32 b x 32 column word
// can be drained straight into the accumulator SRAM.
//
// ── Weight preload ──
// weights_in carries one k-column (N_ROWS x 8 b); weight_col_sel is one-hot and picks
// the output column that latches it.  A full array load therefore takes M_COLS cycles.
//
// ── Feed protocol (see tb_systolic_array_int8.v) ──
// Cycle x presents the anti-diagonal { A[x - i][i] : i } of the activation tile.
// Output row m is valid on partial_sums_out at cycle m + N_ROWS + 1.

module systolic_array_int8 #(
    parameter N_ROWS = 64,     // Number of PE rows (activation dimension)
    parameter M_COLS = 64      // Number of PE columns (weight/output dimension)
) (
    input  wire                         clk,
    input  wire                         rst_n,

    // Control
    input  wire                         weight_load,       // Enable weight preload
    input  wire [M_COLS-1:0]            weight_col_sel,    // One-hot column select during preload
    input  wire                         accum_clear,       // Clear all accumulators
    input  wire                         stall,             // Stall all PEs

    // Activation input (broadcast to every column of row i)
    input  wire [N_ROWS*8-1:0]          activations_in,

    // Weight input (one k-column of N_ROWS x 8b, latched by the selected column)
    input  wire [N_ROWS*8-1:0]          weights_in,

    // Partial sum input (top edge, M_COLS x 32b)
    input  wire [M_COLS*32-1:0]         partial_sums_in,

    // Partial sum output (bottom edge, M_COLS x 32b)
    output wire [M_COLS*32-1:0]         partial_sums_out
);

    genvar i, j;

    // partial_sum_wire[i][j]: partial sum flowing DOWN out of PE(row=i, col=j)
    wire [31:0] partial_sum_wire [0:N_ROWS][0:M_COLS-1];   // +1 for output edge

    generate
        for (i = 0; i < N_ROWS; i = i + 1) begin : row
            // ── reset / clear replication ──
            // Flattened, rst_n and accum_clear each drove every PE in the block
            // (4 arrays x 1024 PEs -> ~131k flops).  repair_design degenerates on a
            // net that size and detailed route never finishes its first iteration.
            // One buffered copy per row keeps each net at ~1k loads.  The buffer
            // is an explicit library cell: a bare wire/buf primitive gets folded
            // away by the mapper and the single 131k-load net comes back.
            wire rst_row, clr_row;
            BUFx24_ASAP7_75t_R u_rst_buf (.A(rst_n),      .Y(rst_row));
            BUFx24_ASAP7_75t_R u_clr_buf (.A(accum_clear), .Y(clr_row));

            for (j = 0; j < M_COLS; j = j + 1) begin : col

                // ── Activation: broadcast, identical for every column of this row ──
                wire [7:0] act_in_wire;
                assign act_in_wire = activations_in[i*8 +: 8];

                // ── Weight input mux ──
                // All rows in column j get the same weight during preload
                wire [7:0] wgt_in_wire;
                assign wgt_in_wire = weights_in[i*8 +: 8];

                // ── Partial sum input mux ──
                // Top edge: from array input; internal: from top neighbor
                wire [31:0] psum_in_wire;
                if (i == 0)
                    assign psum_in_wire = partial_sums_in[j*32 +: 32];
                else
                    assign psum_in_wire = partial_sum_wire[i-1][j];

                // ── PE instantiation ──
                wire pe_weight_load;
                assign pe_weight_load = weight_load && weight_col_sel[j];

                pe_int8 u_pe (
                    .clk             (clk),
                    .rst_n           (rst_row),
                    .weight_load     (pe_weight_load),
                    .accum_clear     (clr_row),
                    .stall           (stall),
                    .activation_in   (act_in_wire),
                    .weight_in       (wgt_in_wire),
                    .partial_sum_in  (psum_in_wire),
                    .partial_sum_out (partial_sum_wire[i][j])
                );

            end
        end

        // ── Output connections ──
        for (j = 0; j < M_COLS; j = j + 1) begin : output_col
            assign partial_sums_out[j*32 +: 32] = partial_sum_wire[N_ROWS-1][j];
        end

    endgenerate

endmodule
