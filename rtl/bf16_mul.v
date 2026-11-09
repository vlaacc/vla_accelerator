// bf16_mul.v — IEEE 754 BF16 (bfloat16) multiplier producing FP32 product
//
// Computes: result = a * b  (both BF16, result FP32)
//
// BF16 format (bit[15]=sign, bit[14:7]=exp bias 127, bit[6:0]=man 7-bit):
//   - zero   : exp==0   && man==0
//   - denorm : exp==0   && man!=0   -> value = 0.man * 2^-126
//   - normal : 0<exp<255           -> value = 1.man * 2^(exp-127)
//   - inf    : exp==255 && man==0
//   - nan    : exp==255 && man!=0
//
// 2-stage pipeline:
//   Stage 1: decompose (sign/exp/man) + classify + 8x8 mantissa multiply
//            + exponent add  (registered)
//   Stage 2: LOD + normalize + FP32 pack, with overflow->Inf and
//            underflow/FTZ->zero  (registered)
//
// Because BF16 (7-bit mantissa) x BF16 fits exactly in FP32 (23-bit mantissa),
// normal x normal is EXACT.  Denormal results (magnitude < 2^-126) are flushed
// to zero (FTZ); overflow past FP32 max is reported as +/-Inf.

`include "sparsenn_params.vh"

module bf16_mul (
    input  wire         clk,
    input  wire         rst_n,
    input  wire [15:0]  a,        // BF16
    input  wire [15:0]  b,        // BF16
    output wire [31:0]  result    // FP32
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

    // ── Stage 1 (combinational): decompose ──
    wire          sign_a, sign_b;
    wire [ 7:0]   exp_a, exp_b;
    wire [ 6:0]   man_a, man_b;

    assign sign_a = a[15];
    assign sign_b = b[15];
    assign exp_a  = a[14:7];
    assign exp_b  = b[14:7];
    assign man_a  = a[6:0];
    assign man_b  = b[6:0];

    // ── Classify operands ──
    wire a_is_zero, b_is_zero, a_is_denorm, b_is_denorm;
    wire a_is_inf,  b_is_inf,  a_is_nan,   b_is_nan;
    wire a_is_norm, b_is_norm;

    assign a_is_zero   = (exp_a == 8'd0)   && (man_a == 7'd0);
    assign a_is_denorm = (exp_a == 8'd0)   && (man_a != 7'd0);
    assign a_is_inf    = (exp_a == 8'd255) && (man_a == 7'd0);
    assign a_is_nan    = (exp_a == 8'd255) && (man_a != 7'd0);
    assign a_is_norm   = (exp_a != 8'd0)   && (exp_a != 8'd255);

    assign b_is_zero   = (exp_b == 8'd0)   && (man_b == 7'd0);
    assign b_is_denorm = (exp_b == 8'd0)   && (man_b != 7'd0);
    assign b_is_inf    = (exp_b == 8'd255) && (man_b == 7'd0);
    assign b_is_nan    = (exp_b == 8'd255) && (man_b != 7'd0);
    assign b_is_norm   = (exp_b != 8'd0)   && (exp_b != 8'd255);

    // ── Extract mantissas with hidden bits (8-bit) ──
    wire [7:0] man_a_ext, man_b_ext;
    assign man_a_ext = a_is_norm   ? {1'b1, man_a} :
                       a_is_denorm ? {1'b0, man_a} : 8'd0;
    assign man_b_ext = b_is_norm   ? {1'b1, man_b} :
                       b_is_denorm ? {1'b0, man_b} : 8'd0;

    // ── Effective exponent (denormal uses true exponent 1-bias = -126) ──
    wire [7:0] eff_exp_a, eff_exp_b;
    assign eff_exp_a = a_is_denorm ? 8'd1 : exp_a;
    assign eff_exp_b = b_is_denorm ? 8'd1 : exp_b;

    // ── Stage 1 results (combinational) ──
    wire         sign_r      = sign_a ^ sign_b;
    wire [8:0]   exp_sum     = {1'b0, eff_exp_a} + {1'b0, eff_exp_b}; // 0..510
    wire [15:0]  man_product = man_a_ext * man_b_ext;                  // 8x8 = 16

    // Exception priority: NaN > Inf > Zero > normal (matches fp16_mul)
    wire any_nan  = a_is_nan || b_is_nan || (a_is_inf && b_is_zero) || (a_is_zero && b_is_inf);
    wire any_inf  = !any_nan && (a_is_inf || b_is_inf);
    wire any_zero = !any_nan && !any_inf && (a_is_zero || b_is_zero);

    // ── Stage 1 registers ──
    reg         r_sign;
    reg [ 8:0]  r_exp_sum;
    reg [15:0]  r_man_product;
    reg         r_nan, r_inf, r_zero;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            r_sign       <= 1'b0;
            r_exp_sum    <= 9'd0;
            r_man_product<= 16'd0;
            r_nan        <= 1'b0;
            r_inf        <= 1'b0;
            r_zero       <= 1'b1;
        end else begin
            r_sign       <= sign_r;
            r_exp_sum    <= exp_sum;
            r_man_product<= man_product;
            r_nan        <= any_nan;
            r_inf        <= any_inf;
            r_zero       <= any_zero;
        end
    end

    // ── Stage 2 (combinational): LOD + normalize + pack ──
    wire [3:0]  mul_lod_pos = lod16(r_man_product);

    // FP32 biased exponent = exp_sum + L - 141  (derived below)
    //   value = man_product/2^L * 2^(exp_sum + L - 268) = 1.frac * 2^(fp32_exp - 127)
    //   => fp32_exp = exp_sum + L - 141
    wire [9:0]  exp_tmp      = {1'b0, r_exp_sum} + {6'd0, mul_lod_pos}; // 0..525
    wire        overflow     = (exp_tmp >= 10'd396);   // fp32_exp >= 255
    wire        underflow    = (exp_tmp <= 10'd141);   // fp32_exp <= 0 (FTZ)
    wire [9:0]  fp32_exp_tmp = exp_tmp - 10'd141;      // 1..254 when valid

    // Mantissa: shift leading one to bit 15, take low 15 bits, pad to 23
    wire [30:0] man_shifted  = r_man_product << (4'd15 - mul_lod_pos);
    wire [22:0] man23        = {man_shifted[14:0], 8'd0};

    wire [31:0] result_comb =
        r_nan     ? {1'b0,   8'hFF, 23'h400000} :   // quiet NaN
        r_inf     ? {r_sign, 8'hFF, 23'd0}      :   // +/-Inf
        r_zero    ? {r_sign, 8'd0,  23'd0}      :   // +/-0
        overflow  ? {r_sign, 8'hFF, 23'd0}      :   // overflow -> +/-Inf
        underflow ? {r_sign, 8'd0,  23'd0}      :   // underflow -> FTZ
                    {r_sign, fp32_exp_tmp[7:0], man23};

    // ── Stage 2 register (output) ──
    reg [31:0] r_result;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            r_result <= 32'd0;
        else
            r_result <= result_comb;
    end

    assign result = r_result;

endmodule
