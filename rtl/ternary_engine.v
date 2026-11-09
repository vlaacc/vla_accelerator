// ternary_engine.v — 16-channel tile-major LUT compute block (unified W4 + ternary)
// Each channel runs lut_compute_unit_mixed (the 4-stage Horner reused for W4 and
// ternary 2x2-stage, runtime-switchable via wt_mode) and streams one redundant
// per-tile dot (tile_s, tile_c).  The K=128 group accumulation is owned by the
// external register-block accumulator (rtl/acc_block_reg.v).
`include "sparse_ternary_params.vh"

(* keep_hierarchy *) module ternary_engine #(
    parameter N_CH_PER_ENGINE = 16,
    parameter CH_OFFSET       = 0
) (
    input  wire                                       clk,
    input  wire                                       rst_n,
    input  wire                                       wt_mode,       // 1=ternary 0=W4
    input  wire [N_CH_PER_ENGINE * `ST_WT_WORD - 1:0] weight_segment,
    input  wire [`ST_PIPE_SUB*`ST_PIPE_BITS*`ST_BUILD_TOTAL_REGS-1:0] build_out,   // shifted subset-sums
    input  wire                          compute_en,
    input  wire                          weight_signed,
    output wire [N_CH_PER_ENGINE * 16 - 1:0] tile_s,       // per-tile redundant dot sum
    output wire [N_CH_PER_ENGINE * 16 - 1:0] tile_c,       // per-tile redundant dot carry
    output wire                              tile_valid    // per-tile 1-cycle pulse
);
    wire [N_CH_PER_ENGINE-1:0] ch_valid;
    genvar ch;
    generate
        for (ch = 0; ch < N_CH_PER_ENGINE; ch = ch + 1) begin : ch_chain
            wire [`ST_WT_WORD-1:0] ch_w =
                weight_segment[ch * `ST_WT_WORD +: `ST_WT_WORD];
            lut_compute_unit_mixed u_lut (
                .clk(clk), .rst_n(rst_n), .wt_mode(wt_mode),
                .compute_en(compute_en), .weight_signed(weight_signed),
                .weight(ch_w), .group_partial_sums(build_out),
                .tile_s(tile_s[ch*16 +: 16]),
                .tile_c(tile_c[ch*16 +: 16]),
                .tile_valid(ch_valid[ch]));
        end
    endgenerate
    assign tile_valid = |ch_valid;
endmodule
