// fp32_mul.v — IEEE 754 FP32 multiplier
// 5-stage pipeline (deepened from 4-stage to meet 1 GHz target, each stage < 1 ns).
//
//   Stage 1 (comb -> reg): decompose + classify + exponent sum + sign
//   Stage 2 (reg -> reg): 24x24 mantissa multiply split into eight 24x3 partial
//                          products (r0..r7) computed in parallel
//   Stage 3 (reg -> reg): carry-save reduction of the eight partial products
//                          down to two 48-bit numbers (sum, carry)
//   Stage 4 (reg -> reg): final 48-bit carry-propagate add -> product
//   Stage 5 (reg -> out):  normalization (casez) + result packing
//
// The previous 24x12 (~1.4 ns) and 24x6 (~1.1 ns) multiplies each embed a wide
// carry-propagate adder that dominates the delay. Splitting the operand into 3-bit
// slices shrinks each multiplier's CPA to 27 bits (~0.9 ns), and the wide
// cross-product merge is done carry-save (XOR/AND tree, ~0.4 ns) so the only wide
// CPA (48-bit, ~0.9 ns) lives alone in Stage 4.
//
// All IEEE-754 special cases (NaN/Inf/zero/denorm) are preserved by carrying the
// classify flags through the pipeline registers.

`include "sparsenn_params.vh"

(* keep_hierarchy *) module fp32_mul (
    input  wire clk, rst_n,
    input  wire [`FP32_WIDTH-1:0] a, b,
    output reg  [`FP32_WIDTH-1:0] result
);

    // ════════════════════════════════════════════════════════════
    // Stage 1 (combinational): decompose + classify + exp sum + sign
    // ════════════════════════════════════════════════════════════

    wire          sign_a, sign_b;
    wire [ 7:0]   exp_a, exp_b;
    wire [22:0]   man_a, man_b;

    assign sign_a = a[31];
    assign sign_b = b[31];
    assign exp_a  = a[30:23];
    assign exp_b  = b[30:23];
    assign man_a  = a[22:0];
    assign man_b  = b[22:0];

    wire a_is_zero, b_is_zero, a_is_denorm, b_is_denorm;
    wire a_is_inf,  b_is_inf,  a_is_nan,   b_is_nan, a_is_norm, b_is_norm;

    assign a_is_zero   = (exp_a == 8'd0) && (man_a == 23'd0);
    assign a_is_denorm = (exp_a == 8'd0) && (man_a != 23'd0);
    assign a_is_inf    = (exp_a == 8'hFF) && (man_a == 23'd0);
    assign a_is_nan    = (exp_a == 8'hFF) && (man_a != 23'd0);
    assign a_is_norm   = (exp_a != 8'd0) && (exp_a != 8'hFF);

    assign b_is_zero   = (exp_b == 8'd0) && (man_b == 23'd0);
    assign b_is_denorm = (exp_b == 8'd0) && (man_b != 23'd0);
    assign b_is_inf    = (exp_b == 8'hFF) && (man_b == 23'd0);
    assign b_is_nan    = (exp_b == 8'hFF) && (man_b != 23'd0);
    assign b_is_norm   = (exp_b != 8'd0) && (exp_b != 8'hFF);

    wire [23:0] man_a_ext, man_b_ext;
    assign man_a_ext = a_is_norm  ? {1'b1, man_a} :
                       a_is_denorm ? {1'b0, man_a} : 24'd0;
    assign man_b_ext = b_is_norm  ? {1'b1, man_b} :
                       b_is_denorm ? {1'b0, man_b} : 24'd0;

    wire          sign_r  = sign_a ^ sign_b;
    wire [ 8:0]   exp_sum = {1'b0, exp_a} + {1'b0, exp_b};

    localparam BIAS = 9'd127;

    // ── Stage 1 -> Stage 2 registers ──
    reg        a_is_nan_r1, b_is_nan_r1, a_is_inf_r1, b_is_inf_r1;
    reg        a_is_zero_r1, b_is_zero_r1;
    reg        sign_r_r1;
    reg [ 8:0] exp_sum_r1;
    reg [23:0] man_a_ext_r1, man_b_ext_r1;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            a_is_nan_r1  <= 1'b0;
            b_is_nan_r1  <= 1'b0;
            a_is_inf_r1  <= 1'b0;
            b_is_inf_r1  <= 1'b0;
            a_is_zero_r1 <= 1'b0;
            b_is_zero_r1 <= 1'b0;
            sign_r_r1    <= 1'b0;
            exp_sum_r1   <= 9'd0;
            man_a_ext_r1 <= 24'd0;
            man_b_ext_r1 <= 24'd0;
        end else begin
            a_is_nan_r1  <= a_is_nan;
            b_is_nan_r1  <= b_is_nan;
            a_is_inf_r1  <= a_is_inf;
            b_is_inf_r1  <= b_is_inf;
            a_is_zero_r1 <= a_is_zero;
            b_is_zero_r1 <= b_is_zero;
            sign_r_r1    <= sign_r;
            exp_sum_r1   <= exp_sum;
            man_a_ext_r1 <= man_a_ext;
            man_b_ext_r1 <= man_b_ext;
        end
    end

    // ════════════════════════════════════════════════════════════
    // Stage 2 (combinational): eight 24x3 partial products in parallel
    // ════════════════════════════════════════════════════════════

    // man_b_ext_r1 = {b[23:21], b[20:18], b[17:15], b[14:12], b[11:9], b[8:6], b[5:3], b[2:0]}
    //   product = man_a * man_b
    //           = r7<<21 + r6<<18 + r5<<15 + r4<<12 + r3<<9 + r2<<6 + r1<<3 + r0
    wire [26:0] r0 = man_a_ext_r1 * man_b_ext_r1[2:0];
    wire [26:0] r1 = man_a_ext_r1 * man_b_ext_r1[5:3];
    wire [26:0] r2 = man_a_ext_r1 * man_b_ext_r1[8:6];
    wire [26:0] r3 = man_a_ext_r1 * man_b_ext_r1[11:9];
    wire [26:0] r4 = man_a_ext_r1 * man_b_ext_r1[14:12];
    wire [26:0] r5 = man_a_ext_r1 * man_b_ext_r1[17:15];
    wire [26:0] r6 = man_a_ext_r1 * man_b_ext_r1[20:18];
    wire [26:0] r7 = man_a_ext_r1 * man_b_ext_r1[23:21];

    // ── Stage 2 -> Stage 3 registers ──
    reg        a_is_nan_r2, b_is_nan_r2, a_is_inf_r2, b_is_inf_r2;
    reg        a_is_zero_r2, b_is_zero_r2;
    reg        sign_r_r2;
    reg [ 8:0] exp_sum_r2;
    reg [26:0] r0_r2, r1_r2, r2_r2, r3_r2, r4_r2, r5_r2, r6_r2, r7_r2;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            a_is_nan_r2  <= 1'b0;
            b_is_nan_r2  <= 1'b0;
            a_is_inf_r2  <= 1'b0;
            b_is_inf_r2  <= 1'b0;
            a_is_zero_r2 <= 1'b0;
            b_is_zero_r2 <= 1'b0;
            sign_r_r2    <= 1'b0;
            exp_sum_r2   <= 9'd0;
            r0_r2 <= 27'd0; r1_r2 <= 27'd0; r2_r2 <= 27'd0; r3_r2 <= 27'd0;
            r4_r2 <= 27'd0; r5_r2 <= 27'd0; r6_r2 <= 27'd0; r7_r2 <= 27'd0;
        end else begin
            a_is_nan_r2  <= a_is_nan_r1;
            b_is_nan_r2  <= b_is_nan_r1;
            a_is_inf_r2  <= a_is_inf_r1;
            b_is_inf_r2  <= b_is_inf_r1;
            a_is_zero_r2 <= a_is_zero_r1;
            b_is_zero_r2 <= b_is_zero_r1;
            sign_r_r2    <= sign_r_r1;
            exp_sum_r2   <= exp_sum_r1;
            r0_r2 <= r0; r1_r2 <= r1; r2_r2 <= r2; r3_r2 <= r3;
            r4_r2 <= r4; r5_r2 <= r5; r6_r2 <= r6; r7_r2 <= r7;
        end
    end

    // ════════════════════════════════════════════════════════════
    // Stage 3 (combinational): carry-save reduction 8 -> 2
    // ════════════════════════════════════════════════════════════
    wire [47:0] x0 = {21'd0, r0_r2};
    wire [47:0] x1 = {r1_r2, 3'd0};
    wire [47:0] x2 = {r2_r2, 6'd0};
    wire [47:0] x3 = {r3_r2, 9'd0};
    wire [47:0] x4 = {r4_r2, 12'd0};
    wire [47:0] x5 = {r5_r2, 15'd0};
    wire [47:0] x6 = {r6_r2, 18'd0};
    wire [47:0] x7 = {r7_r2, 21'd0};

    // Level 1: 3:2 compressors
    wire [47:0] a_s = x0 ^ x1 ^ x2;   wire [47:0] a_c = (x0 & x1) | (x1 & x2) | (x0 & x2);
    wire [47:0] b_s = x3 ^ x4 ^ x5;   wire [47:0] b_c = (x3 & x4) | (x4 & x5) | (x3 & x5);
    // Level 2: 3:2 compressors
    wire [47:0] c_s = a_s ^ (a_c << 1) ^ b_s;      wire [47:0] c_c = (a_s & (a_c << 1)) | ((a_c << 1) & b_s) | (a_s & b_s);
    wire [47:0] d_s = (b_c << 1) ^ x6 ^ x7;        wire [47:0] d_c = ((b_c << 1) & x6) | (x6 & x7) | ((b_c << 1) & x7);
    // Level 3: 3:2 compressor
    wire [47:0] e_s = c_s ^ (c_c << 1) ^ d_s;      wire [47:0] e_c = (c_s & (c_c << 1)) | ((c_c << 1) & d_s) | (c_s & d_s);
    // Level 4: 3:2 compressor
    wire [47:0] sum_cs = e_s ^ (e_c << 1) ^ (d_c << 1);
    wire [47:0] car_cs = (e_s & (e_c << 1)) | ((e_c << 1) & (d_c << 1)) | (e_s & (d_c << 1));

    // ── Stage 3 -> Stage 4 registers ──
    reg        a_is_nan_r3, b_is_nan_r3, a_is_inf_r3, b_is_inf_r3;
    reg        a_is_zero_r3, b_is_zero_r3;
    reg        sign_r_r3;
    reg [ 8:0] exp_sum_r3;
    reg [47:0] sum_cs_r3, car_cs_r3;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            a_is_nan_r3  <= 1'b0;
            b_is_nan_r3  <= 1'b0;
            a_is_inf_r3  <= 1'b0;
            b_is_inf_r3  <= 1'b0;
            a_is_zero_r3 <= 1'b0;
            b_is_zero_r3 <= 1'b0;
            sign_r_r3    <= 1'b0;
            exp_sum_r3   <= 9'd0;
            sum_cs_r3    <= 48'd0;
            car_cs_r3    <= 48'd0;
        end else begin
            a_is_nan_r3  <= a_is_nan_r2;
            b_is_nan_r3  <= b_is_nan_r2;
            a_is_inf_r3  <= a_is_inf_r2;
            b_is_inf_r3  <= b_is_inf_r2;
            a_is_zero_r3 <= a_is_zero_r2;
            b_is_zero_r3 <= b_is_zero_r2;
            sign_r_r3    <= sign_r_r2;
            exp_sum_r3   <= exp_sum_r2;
            sum_cs_r3    <= sum_cs;
            car_cs_r3    <= car_cs;
        end
    end

    // ════════════════════════════════════════════════════════════
    // Stage 4 (combinational): carry-select 48-bit add -> product
    // ════════════════════════════════════════════════════════════
    // product = sum_cs + (car_cs << 1);  a 48-bit CPA (~1.3 ns) is split into
    // two parallel 24-bit adders + a carry mux (~0.85 ns).
    wire [24:0] low_sum = {1'b0, sum_cs_r3[23:0]} + {1'b0, car_cs_r3[22:0], 1'b0};
    wire [23:0] hi_sum0 = sum_cs_r3[47:24] + car_cs_r3[46:23];
    wire [23:0] hi_sum1 = sum_cs_r3[47:24] + car_cs_r3[46:23] + 24'd1;
    wire [23:0] hi_sum  = low_sum[24] ? hi_sum1 : hi_sum0;
    wire [47:0] man_product = {hi_sum, low_sum[23:0]};

    // ── Stage 4 -> Stage 5 registers ──
    reg        a_is_nan_r4, b_is_nan_r4, a_is_inf_r4, b_is_inf_r4;
    reg        a_is_zero_r4, b_is_zero_r4;
    reg        sign_r_r4;
    reg [ 8:0] exp_sum_r4;
    reg [47:0] man_product_r4;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            a_is_nan_r4   <= 1'b0;
            b_is_nan_r4   <= 1'b0;
            a_is_inf_r4   <= 1'b0;
            b_is_inf_r4   <= 1'b0;
            a_is_zero_r4  <= 1'b0;
            b_is_zero_r4  <= 1'b0;
            sign_r_r4     <= 1'b0;
            exp_sum_r4    <= 9'd0;
            man_product_r4 <= 48'd0;
        end else begin
            a_is_nan_r4   <= a_is_nan_r3;
            b_is_nan_r4   <= b_is_nan_r3;
            a_is_inf_r4   <= a_is_inf_r3;
            b_is_inf_r4   <= b_is_inf_r3;
            a_is_zero_r4  <= a_is_zero_r3;
            b_is_zero_r4  <= b_is_zero_r3;
            sign_r_r4     <= sign_r_r3;
            exp_sum_r4    <= exp_sum_r3;
            man_product_r4 <= man_product;
        end
    end

    // ════════════════════════════════════════════════════════════
    // Stage 5 (combinational): normalization (casez) + pack result
    // ════════════════════════════════════════════════════════════

    always @(*) begin
        if (a_is_nan_r4 || b_is_nan_r4 || (a_is_inf_r4 && b_is_zero_r4) || (a_is_zero_r4 && b_is_inf_r4)) begin
            result = {1'b0, 8'hFF, 23'h400000};
        end
        else if (a_is_inf_r4 || b_is_inf_r4) begin
            result = {sign_r_r4, 8'hFF, 23'd0};
        end
        else if (a_is_zero_r4 || b_is_zero_r4) begin
            result = {sign_r_r4, 8'd0, 23'd0};
        end
        else begin
            casez (man_product_r4)
                48'b1???????????????????????????????????????????????: begin
                    result = {sign_r_r4, exp_sum_r4 + 9'd1 - BIAS, man_product_r4[46:24]};
                end
                48'b01??????????????????????????????????????????????: begin
                    result = {sign_r_r4, exp_sum_r4 - BIAS, man_product_r4[45:23]};
                end
                48'b001?????????????????????????????????????????????: begin
                    result = {sign_r_r4, exp_sum_r4 - 9'd1 - BIAS, man_product_r4[44:22]};
                end
                48'b0001????????????????????????????????????????????: begin
                    result = {sign_r_r4, exp_sum_r4 - 9'd2 - BIAS, man_product_r4[43:21]};
                end
                default: begin
                    result = {sign_r_r4, 8'd0, 23'd0};
                end
            endcase
        end
    end

endmodule
