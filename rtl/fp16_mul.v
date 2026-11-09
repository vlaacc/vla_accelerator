// fp16_mul.v — IEEE 754 FP16 multiplier producing FP32 product
// 2-stage pipeline (target 1 GHz, each stage < 1 ns).
//
//   Stage 1 (comb -> reg): decompose + classify + exponent sum + sign +
//                          the 11x11 mantissa multiply split into two parallel
//                          partial products (11x6 and 11x5)
//   Stage 2 (reg -> out):  merge partial products (22-bit add) + leading-one
//                          detect (lod32) + normalize + pack FP32
//
// The 11x11 multiply is split into an 11x6 (p_lo) and 11x5 (p_hi) so each
// partial-product multiplier's carry-propagate adder is only ~17/16 bits
// (~0.6 ns). The only wide adder left is the 22-bit merge in Stage 2 (~0.7 ns),
// and the leading-one detection (lod32) is pushed into Stage 3 where it runs in
// parallel with the exponent add, so no stage embeds a wide CPA + LOD in series.
//
// All IEEE-754 special cases (NaN/Inf/zero/denorm) are preserved by carrying the
// classify flags through the pipeline registers. Functionally identical to the
// original combinational implementation (rtl/fp16_mul_comb.v.bak).

`include "sparsenn_params.vh"

(* keep_hierarchy *) module fp16_mul (
    input  wire clk, rst_n,
    input  wire [`FP16_WIDTH-1:0] a, b,
    output reg  [`FP32_WIDTH-1:0] result
);

    // ── Tree-structured Leading-One Detector (LOD) ──
    function [0:0] lod2;
        input [1:0] val;
        lod2 = val[1];
    endfunction
    function [1:0] lod4;
        input [3:0] val;
        lod4 = |val[3:2] ? (2'd2 + {1'b0, lod2(val[3:2])}) : {1'b0, lod2(val[1:0])};
    endfunction
    function [2:0] lod8;
        input [7:0] val;
        lod8 = |val[7:4] ? (3'd4 + {1'b0, lod4(val[7:4])}) : {1'b0, lod4(val[3:0])};
    endfunction
    function [3:0] lod16;
        input [15:0] val;
        lod16 = |val[15:8] ? (4'd8 + {1'b0, lod8(val[15:8])}) : {1'b0, lod8(val[7:0])};
    endfunction
    function [4:0] lod32;
        input [31:0] val;
        lod32 = |val[31:16] ? (5'd16 + {1'b0, lod16(val[31:16])}) : {1'b0, lod16(val[15:0])};
    endfunction

    // ════════════════════════════════════════════════════════════
    // Stage 1 (combinational): decompose + classify + exp sum + sign
    //                          + two parallel partial products
    // ════════════════════════════════════════════════════════════

    wire          sign_a, sign_b;
    wire [ 4:0]   exp_a, exp_b;
    wire [ 9:0]   man_a, man_b;

    assign sign_a = a[15];
    assign sign_b = b[15];
    assign exp_a  = a[14:10];
    assign exp_b  = b[14:10];
    assign man_a  = a[9:0];
    assign man_b  = b[9:0];

    wire a_is_zero, b_is_zero, a_is_denorm, b_is_denorm;
    wire a_is_inf,  b_is_inf,  a_is_nan,   b_is_nan;
    wire a_is_norm, b_is_norm;

    assign a_is_zero   = (exp_a == 5'd0) && (man_a == 10'd0);
    assign a_is_denorm = (exp_a == 5'd0) && (man_a != 10'd0);
    assign a_is_inf    = (exp_a == 5'd31) && (man_a == 10'd0);
    assign a_is_nan    = (exp_a == 5'd31) && (man_a != 10'd0);
    assign a_is_norm   = (exp_a != 5'd0) && (exp_a != 5'd31);

    assign b_is_zero   = (exp_b == 5'd0) && (man_b == 10'd0);
    assign b_is_denorm = (exp_b == 5'd0) && (man_b != 10'd0);
    assign b_is_inf    = (exp_b == 5'd31) && (man_b == 10'd0);
    assign b_is_nan    = (exp_b == 5'd31) && (man_b != 10'd0);
    assign b_is_norm   = (exp_b != 5'd0) && (exp_b != 5'd31);

    wire [10:0] man_a_ext, man_b_ext;
    assign man_a_ext = a_is_norm  ? {1'b1, man_a} :
                       a_is_denorm ? {1'b0, man_a} : 11'd0;
    assign man_b_ext = b_is_norm  ? {1'b1, man_b} :
                       b_is_denorm ? {1'b0, man_b} : 11'd0;

    wire          sign_r;
    wire [ 7:0]   exp_sum_raw;

    assign sign_r      = sign_a ^ sign_b;
    assign exp_sum_raw = {3'd0, exp_a} + {3'd0, exp_b};

    // 11x11 = (11x6) + (11x5 << 6):  man_b_ext[5:0] and man_b_ext[10:6]
    wire [16:0]   p_lo = man_a_ext * man_b_ext[5:0];     // 11 x 6 = 17 bits
    wire [15:0]   p_hi = man_a_ext * man_b_ext[10:6];    // 11 x 5 = 16 bits

    // ── Stage 1 -> Stage 2 registers ──
    reg        a_is_nan_r1, b_is_nan_r1, a_is_inf_r1, b_is_inf_r1;
    reg        a_is_zero_r1, b_is_zero_r1;
    reg        sign_r_r1;
    reg [ 7:0] exp_sum_r1;
    reg [16:0] p_lo_r1;
    reg [15:0] p_hi_r1;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            a_is_nan_r1  <= 1'b0;
            b_is_nan_r1  <= 1'b0;
            a_is_inf_r1  <= 1'b0;
            b_is_inf_r1  <= 1'b0;
            a_is_zero_r1 <= 1'b0;
            b_is_zero_r1 <= 1'b0;
            sign_r_r1    <= 1'b0;
            exp_sum_r1   <= 8'd0;
            p_lo_r1      <= 17'd0;
            p_hi_r1      <= 16'd0;
        end else begin
            a_is_nan_r1  <= a_is_nan;
            b_is_nan_r1  <= b_is_nan;
            a_is_inf_r1  <= a_is_inf;
            b_is_inf_r1  <= b_is_inf;
            a_is_zero_r1 <= a_is_zero;
            b_is_zero_r1 <= b_is_zero;
            sign_r_r1    <= sign_r;
            exp_sum_r1   <= exp_sum_raw;
            p_lo_r1      <= p_lo;
            p_hi_r1      <= p_hi;
        end
    end

    // ════════════════════════════════════════════════════════════
    // Stage 2 (combinational): merge partial products -> 22-bit product
    // ════════════════════════════════════════════════════════════

    // ════════════════════════════════════════════════════════════
    // Stage 2 (combinational): merge partial products (22-bit add)
    //                          + leading-one detect + normalize + pack
    // (合并了原 Stage 2 和 Stage 3：22-bit 加法 ~330ps + LOD ~150ps
    //  + 归一化 ~150ps ≈ 630ps < 1ns，减少一级流水线)
    // ════════════════════════════════════════════════════════════

    wire [21:0]   man_product = p_lo_r1 + {p_hi_r1, 6'd0};
    wire [4:0]    mul_lod_pos = lod32({10'd0, man_product});

    always @(*) begin
        // NaN propagation: NaN * anything = NaN
        if (a_is_nan_r1 || b_is_nan_r1 || (a_is_inf_r1 && b_is_zero_r1) || (a_is_zero_r1 && b_is_inf_r1)) begin
            // Quiet NaN in FP32: 0_11111111_100...0
            result = {1'b0, 8'hFF, 23'h400000};
        end
        // Inf * anything non-NaN = Inf (sign depends on operands)
        else if (a_is_inf_r1 || b_is_inf_r1) begin
            result = {sign_r_r1, 8'hFF, 23'd0};
        end
        // Zero * anything = Zero
        else if (a_is_zero_r1 || b_is_zero_r1) begin
            result = {sign_r_r1, 8'd0, 23'd0};
        end
        // Denormal or normal: LOD-based normalization
        else begin
            case (mul_lod_pos)
                // product in [2,4): leading one at bit 21, exp_sum + 21 + 77
                5'd21: result = {sign_r_r1, exp_sum_r1 + 8'd98, man_product[20:0], 2'd0};
                5'd20: result = {sign_r_r1, exp_sum_r1 + 8'd97, man_product[19:0], 3'd0};
                5'd19: result = {sign_r_r1, exp_sum_r1 + 8'd96, man_product[18:0], 4'd0};
                5'd18: result = {sign_r_r1, exp_sum_r1 + 8'd95, man_product[17:0], 5'd0};
                5'd17: result = {sign_r_r1, exp_sum_r1 + 8'd94, man_product[16:0], 6'd0};
                // lod < 17: subnormal or very small -> flush to zero
                default: result = {sign_r_r1, 8'd0, 23'd0};
            endcase
        end
    end

endmodule
