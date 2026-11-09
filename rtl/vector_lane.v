// vector_lane.v — dual-datapath 32 b SIMD lane (integer + floating point)
//
// The 4096 b vector is 128 of these.  Each lane carries BOTH datapaths and an
// instruction mux picks between them, because the epilogue needs both:
//
//   integer (quantization) : acc_sram holds INT32 accumulators; dequant / requant /
//       bias-add are `(acc - zp) * scale >>> sh` + saturate — all integer/fixed point.
//       Using a float unit here is not just larger, it is WRONG: the accumulator
//       magnitude is ~2^22 and FP16's 11-bit significand cannot represent it.
//   floating point         : the genuine non-linearities (exp / gelu / sqrt / softmax
//       / LayerNorm) still need IEEE arithmetic.
//
// Measured 45nm reference areas (yosys synth -flatten + abc, Nangate45):
//   integer lane (dq_lane)           4,687.5 um2
//   floating lane (fp32_add+fp32_mul) 8,879.9 um2   (2,502.5 + 6,377.4)
//   => 13,567.4 um2 per lane, 128 lanes ~= 1.74 mm2 of arithmetic
// The multiplier is NOT shared between the two datapaths and the integer multiplier
// is retained — both were explicit decisions (see the ASAP7 1RW plan §7.2 B2/B3).

`include "sparsenn_params.vh"

module vector_lane (
    input  wire                   clk,
    input  wire                   rst_n,

    // instruction select
    input  wire [1:0]             op,       // 0=INT dequant, 1=FP add+scale, 2=FP mul
    input  wire                   is_fp,    // 1 = route through the FP datapath

    // operands (acc-word lane `i` of the 4096 b vector)
    input  wire signed [31:0]     a_in,     // INT32 accumulator / FP32 operand A
    input  wire signed [31:0]     b_in,     // FP32 operand B (unused by INT ops)

    // integer-side parameters (per-group quantization)
    input  wire signed [15:0]     zp,       // zero point
    input  wire signed [15:0]     scale,    // Q1.15 multiplier
    input  wire [4:0]             sh,       // arithmetic right shift

    output wire signed [31:0]     int_out,
    output wire        [`FP32_WIDTH-1:0] fp_add_out,
    output wire        [`FP32_WIDTH-1:0] fp_mul_out,
    output reg  signed [31:0]     lane_out  // muxed result
);
    // ── integer datapath: (a - zp) * scale >>> sh, saturating ──
    // 3-stage: sub | mult | shift+saturate
    reg signed [31:0] d_r;
    reg signed [15:0] s_r;
    reg        [4:0]  sh_r;
    reg signed [47:0] p_r;

    wire signed [47:0] prod = d_r * s_r;
    wire signed [47:0] shf  = prod >>> sh_r;
    wire signed [31:0] sat  = (shf >  48'sd2147483647) ?  32'sd2147483647
                            : (shf < -48'sd2147483648) ? -32'sd2147483648
                            : shf[31:0];

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            d_r <= 0; s_r <= 0; sh_r <= 0; p_r <= 0;
        end else begin
            d_r  <= a_in - zp;
            s_r  <= scale;
            sh_r <= sh;
            p_r  <= prod;
        end
    end
    assign int_out = sat;

    // ── floating datapath: the existing 5-stage fp32 units, unchanged ──
    fp32_add u_add (.clk(clk), .rst_n(rst_n), .a(a_in), .b(b_in), .result(fp_add_out));
    fp32_mul u_mul (.clk(clk), .rst_n(rst_n), .a(a_in), .b(b_in), .result(fp_mul_out));

    // ── instruction mux ──
    reg signed [31:0] fp_sel_r;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            lane_out <= 0;
            fp_sel_r <= 0;
        end else begin
            fp_sel_r <= (op == 2'd2) ? fp_mul_out : fp_add_out;
            lane_out <= is_fp ? fp_sel_r : int_out;
        end
    end
endmodule
