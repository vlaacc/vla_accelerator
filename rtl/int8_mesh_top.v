// int8_mesh_top.v — 4 x 32x32 INT8 weight-stationary meshes + their own SRAM (ASAP7 1RW)
//
// ── Array semantics (rtl/systolic_array_int8.v) ──
//   N_ROWS (=K)  : activations are BROADCAST to every PE of row i (no horizontal chain)
//   M_COLS (=N)  : partial sums enter at the top edge of each column and shift DOWN
//   PE(i,j) holds W[i][j]; out[j] = sum_i W[i][j] * A[i]
//   weight preload: one COLUMN per cycle, selected by the one-hot weight_col_sel
//   feed: activations_in[i] at cycle x must be A[x - i][i] (anti-diagonal skew) so
//         that all M_COLS outputs for a given row leave the bottom edge together.
//
// ── Why M-split ──
// The 4 meshes cover the SAME K/N block and differ only in the m row they process.
// Therefore all 4 hold the SAME 32K x 32N weight block (1024 x 8 b = 8 Kbit) and it is
// BROADCAST once, instead of 4 independent weight streams under an N-split.  The cost
// is 4 x 256 b = 1024 b/cycle of activation, which is why this block owns a dedicated
// act macro group (sram_act_bank_int8, instantiated in int8_engine.v).
// See docs/superpowers/plans/2026-09-22-asap7-sram-1rw-macro-selection.md §2.5 D.

module int8_mesh_top #(
    parameter MESHES = 4,
    parameter K      = 32,      // array rows  (contraction, activation shift direction)
    parameter N      = 32       // array cols  (outputs, partial-sum shift direction)
) (
    input  wire                                    clk,
    input  wire                                    rst_n,      // 500 MHz domain

    // ── weight load (shared block, broadcast to all MESHES) ──
    // 8 Kbit total = 32 columns x 256 b.  `wgt_ld_data` carries one 256 b column when
    // wgt_ld_en is high, addressing wgt_ld_col (or auto-incrementing if wgt_ld_col_we).
    input  wire                                    wgt_ld_en,
    input  wire [4:0]                              wgt_ld_col,
    input  wire [K*8-1:0]                          wgt_ld_data,

    // ── activation stream: one 256 b vector per mesh (different m rows) ──
    input  wire [MESHES*K*8-1:0]                   act_in,

    // ── partial sums out: one 1024 b vector per mesh ──
    output wire [MESHES*N*32-1:0]                  psum_out,

    // ── control ──
    input  wire                                    accum_clear,
    output wire                                    preload_done,
    output wire [4:0]                              dbg_col
);
    localparam WCOL_W = K*8;                 // 256 b per weight column

    // ── 8 Kbit weight register file, shared by all MESHES ──
    // One always block put all 8192 flops on the single reset net; per-entry blocks
    // with their own buffered reset keep each copy at 256 loads.
    reg [WCOL_W-1:0] wgt_buf [0:31];
    genvar gw;
    generate
        for (gw = 0; gw < 32; gw = gw + 1) begin : wbuf
            wire rst_w;
            BUFx24_ASAP7_75t_R u_rst_w (.A(rst_n), .Y(rst_w));
            always @(posedge clk or negedge rst_w) begin
                if (!rst_w)                       wgt_buf[gw] <= {WCOL_W{1'b0}};
                else if (wgt_ld_en && (wgt_ld_col == gw)) wgt_buf[gw] <= wgt_ld_data;
            end
        end
    endgenerate

    // ── preload FSM: walk the one-hot column select 0..31 ──
    reg        preloading;
    reg [4:0]  pre_col;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            preloading <= 1'b0;
            pre_col    <= 5'd0;
        end else begin
            if (accum_clear) begin
                preloading <= 1'b1;      // re-preload the array on every fresh accumulation
                pre_col    <= 5'd0;
            end else if (preloading) begin
                if (pre_col == 5'd31) preloading <= 1'b0;
                else                  pre_col <= pre_col + 1'b1;
            end
        end
    end
    assign preload_done = ~preloading;
    assign dbg_col      = pre_col;

    // ── the 4 meshes: same weights, different activations ──
    genvar g;
    generate
        for (g = 0; g < MESHES; g = g + 1) begin : mesh
            wire [N-1:0] col_sel;
            assign col_sel = preloading ? ({{(N-1){1'b0}}, 1'b1} << pre_col) : {N{1'b0}};

            systolic_array_int8 #(.N_ROWS(K), .M_COLS(N)) u_arr (
                .clk             (clk),
                .rst_n           (rst_n),
                .weight_load     (preloading),
                .weight_col_sel  (col_sel),
                .accum_clear     (accum_clear),
                .stall           (1'b0),
                .activations_in  (act_in[g*K*8 +: K*8]),   // 4 DIFFERENT m rows
                .weights_in      (wgt_buf[pre_col]),       // ONE broadcast copy
                .partial_sums_in ({N*32{1'b0}}),           // cross-row accumulation lives
                .partial_sums_out(psum_out[g*N*32 +: N*32])// ...in the acc SRAM, not here
            );
        end
    endgenerate
endmodule
