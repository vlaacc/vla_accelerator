// vector_unit_4096.v — 4096 b SIMD vector (128 lanes x 32 b), integer + FP datapaths
//
// ── Why 4096 b ──
// The epilogue reads/modifies acc_sram, whose 1RW port is 8192 b wide and whose word
// is 4096 b (= 128 ch x 32 b).  A narrow (512 b) vector would take 8x as many passes
// and dominate the schedule: measured overhead drops from +118.8% to +31.3% (ternary)
// when the vector is widened to 4096 b.  See the ASAP7 1RW plan §5.4.
//
// ── Why both datapaths, not shared ──
// See rtl/vector_lane.v.  Dequant/requant is integer (an FP16 significand cannot hold
// a ~2^22 accumulator), the non-linearities are floating point, and both the
// multiplier-sharing and I2F-round-trip alternatives were explicitly rejected.
//
// ── Interface ──
// The datapath is deliberately FSM-free so it can be driven by the existing
// vector_unit FSM: `acc_rd_data` comes straight from the acc SRAM port, `acc_wr_data`
// goes straight back, and the reduction trees expose max/sum of the current word.

`include "sparsenn_params.vh"

module vector_unit_4096 #(
    parameter LANES = 128,
    parameter ACC_W = 32
) (
    input  wire                          clk,
    input  wire                          rst_n,

    // ── per-group quantization parameters (integer epilogue) ──
    input  wire signed [15:0]            zp,
    input  wire signed [15:0]            scale,       // Q1.15
    input  wire [4:0]                    sh,

    // ── operand / result: exactly one acc SRAM word ──
    input  wire [LANES*ACC_W-1:0]        acc_rd_data, // read from acc_sram
    input  wire [LANES*ACC_W-1:0]        b_operand,   // second operand (elementwise ops)
    output wire [LANES*ACC_W-1:0]        acc_wr_data, // write back to acc_sram

    // ── instruction ──
    input  wire [1:0]                    op,          // 0=int dequant, 1=fp add, 2=fp mul
    input  wire                          is_fp,

    // ── reduction over the whole word (softmax / LayerNorm statistics) ──
    output wire [LANES*ACC_W-1:0]        vec_out,
    output wire signed [ACC_W-1:0]       red_sum,
    output wire signed [ACC_W-1:0]       red_max
);
    // ── 128 lanes ──
    wire signed [ACC_W-1:0] lane_out [0:LANES-1];
    genvar g;
    generate
        for (g = 0; g < LANES; g = g + 1) begin : lane
            wire signed [ACC_W-1:0] lane_result;
            // Per-lane reset buffer: flattened, rst_n drove every flop in the
            // vector (~109k) plus the mesh's ~131k, which repair_design cannot
            // buffer.  One copy per lane keeps each net at ~850 loads; the cell is
            // explicit because a wire-level buffer gets optimised away.
            wire rst_lane;
            BUFx24_ASAP7_75t_R u_rst_buf (.A(rst_n), .Y(rst_lane));
            vector_lane u_lane (
                .clk(clk), .rst_n(rst_lane),
                .op(op), .is_fp(is_fp),
                .a_in (acc_rd_data[g*ACC_W +: ACC_W]),
                .b_in (b_operand  [g*ACC_W +: ACC_W]),
                .zp(zp), .scale(scale), .sh(sh),
                .int_out(), .fp_add_out(), .fp_mul_out(),
                .lane_out(lane_result)
            );
            assign lane_out[g]          = lane_result;
            assign vec_out[g*ACC_W +: ACC_W] = lane_result;
        end
    endgenerate

    assign acc_wr_data = vec_out;

    // ── reduction trees (pipelined: 7 levels for 128 lanes) ──
    // Sum tree: 128 -> 1, 32 b adders, registered per level.
    // Max tree: 128 -> 1, signed comparators, registered per level.
    // Both are kept here (rather than reusing the non-linear units) because softmax
    // and LayerNorm need them on EVERY element, so they must be full-rate.
    function automatic signed [ACC_W-1:0] iadd;
        input signed [ACC_W-1:0] x, y;
        begin iadd = x + y; end
    endfunction
    function automatic signed [ACC_W-1:0] imax;
        input signed [ACC_W-1:0] x, y;
        begin imax = (x > y) ? x : y; end
    endfunction

    localparam LEVELS = 7;                      // 2^7 = 128
    wire signed [ACC_W-1:0] sum_lv [0:LEVELS][0:LANES-1];
    wire signed [ACC_W-1:0] max_lv [0:LEVELS][0:LANES-1];
    genvar lv, n;
    generate
        for (n = 0; n < LANES; n = n + 1) begin : lv0
            assign sum_lv[0][n] = lane_out[n];
            assign max_lv[0][n] = lane_out[n];
        end
        for (lv = 1; lv <= LEVELS; lv = lv + 1) begin : level
            for (n = 0; n < (LANES >> lv); n = n + 1) begin : node
                reg signed [ACC_W-1:0] s_r, m_r;
                // 127 tree nodes x 64 flops otherwise share the single reset net.
                wire rst_node;
                BUFx24_ASAP7_75t_R u_rst_node (.A(rst_n), .Y(rst_node));
                always @(posedge clk or negedge rst_node) begin
                    if (!rst_node) begin
                        s_r <= 0;
                        m_r <= 0;
                    end else begin
                        s_r <= iadd(sum_lv[lv-1][2*n], sum_lv[lv-1][2*n+1]);
                        m_r <= imax(max_lv[lv-1][2*n], max_lv[lv-1][2*n+1]);
                    end
                end
                assign sum_lv[lv][n] = s_r;
                assign max_lv[lv][n] = m_r;
            end
        end
    endgenerate
    assign red_sum = sum_lv[LEVELS][0];
    assign red_max = max_lv[LEVELS][0];
endmodule
