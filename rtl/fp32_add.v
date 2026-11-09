// fp32_add.v — IEEE 754 FP32 adder with normalization
// 5-stage pipeline (deepened from 4-stage to meet 1 GHz target, each stage < 1 ns).
//
//   Stage 1 (comb -> reg): decompose (sign/exp/man) + classify (nan/inf/zero/norm/denorm)
//   Stage 2 (reg -> reg):  exponent compare + operand ordering + effective_sub
//   Stage 3 (reg -> reg):  align shift of the smaller mantissa (barrel shifter)
//   Stage 4 (reg -> reg):  28-bit mantissa add/subtract + LOD (leading-one detect)
//   Stage 5 (reg -> out):  normalization shift + result packing
//
// The previously-bottleneck Stage 1 (decompose+classify+exp-compare+align-shift, ~1.3 ns)
// is split into three stages (1/2/3). The 28-bit adder and the lod32 tree are merged into
// a single Stage 4 because neither alone approached 1 ns.
//
// All IEEE-754 special cases (NaN/Inf/zero/denorm/underflow-flush) are preserved by
// carrying the classify flags through the pipeline registers.

`include "sparsenn_params.vh"

(* keep_hierarchy *) module fp32_add (
    input  wire clk, rst_n,
    input  wire [`FP32_WIDTH-1:0] a, b,
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
    // Stage 1 (combinational): decompose + classify
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
    wire a_is_inf,  b_is_inf,  a_is_nan,   b_is_nan,  a_is_norm, b_is_norm;

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

    // Extended mantissas (24 bits with hidden bit, plus guard bits → 27 total)
    wire [26:0] man_a_ext, man_b_ext;
    assign man_a_ext = a_is_norm  ? {1'b1, man_a, 3'd0} :
                       a_is_denorm ? {1'b0, man_a, 3'd0} : 27'd0;
    assign man_b_ext = b_is_norm  ? {1'b1, man_b, 3'd0} :
                       b_is_denorm ? {1'b0, man_b, 3'd0} : 27'd0;

    // ── Stage 1 -> Stage 2 registers ──
    reg        a_is_nan_r1, b_is_nan_r1, a_is_inf_r1, b_is_inf_r1;
    reg        sign_a_r1, sign_b_r1;
    reg [ 7:0] exp_a_r1, exp_b_r1;
    reg [26:0] man_a_ext_r1, man_b_ext_r1;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            a_is_nan_r1   <= 1'b0;
            b_is_nan_r1   <= 1'b0;
            a_is_inf_r1   <= 1'b0;
            b_is_inf_r1   <= 1'b0;
            sign_a_r1     <= 1'b0;
            sign_b_r1     <= 1'b0;
            exp_a_r1      <= 8'd0;
            exp_b_r1      <= 8'd0;
            man_a_ext_r1  <= 27'd0;
            man_b_ext_r1  <= 27'd0;
        end else begin
            a_is_nan_r1   <= a_is_nan;
            b_is_nan_r1   <= b_is_nan;
            a_is_inf_r1   <= a_is_inf;
            b_is_inf_r1   <= b_is_inf;
            sign_a_r1     <= sign_a;
            sign_b_r1     <= sign_b;
            exp_a_r1      <= exp_a;
            exp_b_r1      <= exp_b;
            man_a_ext_r1  <= man_a_ext;
            man_b_ext_r1  <= man_b_ext;
        end
    end

    // ════════════════════════════════════════════════════════════
    // Stage 2 (combinational): exponent compare + operand ordering
    // ════════════════════════════════════════════════════════════
    wire       a_larger = (exp_a_r1 > exp_b_r1) ||
                          ((exp_a_r1 == exp_b_r1) && (man_a_ext_r1 >= man_b_ext_r1));

    wire [ 7:0] exp_large  = a_larger ? exp_a_r1 : exp_b_r1;
    wire [ 7:0] exp_small  = a_larger ? exp_b_r1 : exp_a_r1;
    wire [ 7:0] exp_diff   = exp_large - exp_small;

    wire        sign_large  = a_larger ? sign_a_r1 : sign_b_r1;
    wire        sign_small  = a_larger ? sign_b_r1 : sign_a_r1;
    wire [26:0] man_large   = a_larger ? man_a_ext_r1 : man_b_ext_r1;
    wire [26:0] man_small   = a_larger ? man_b_ext_r1 : man_a_ext_r1;

    wire        effective_sub = sign_large ^ sign_small;

    // ── Stage 2 -> Stage 3 registers ──
    reg        a_is_nan_r2, b_is_nan_r2, a_is_inf_r2, b_is_inf_r2;
    reg        sign_a_r2, sign_b_r2, sign_large_r2, effective_sub_r2;
    reg [ 7:0] exp_large_r2, exp_diff_r2;
    reg [26:0] man_large_r2, man_small_r2;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            a_is_nan_r2     <= 1'b0;
            b_is_nan_r2     <= 1'b0;
            a_is_inf_r2     <= 1'b0;
            b_is_inf_r2     <= 1'b0;
            sign_a_r2       <= 1'b0;
            sign_b_r2       <= 1'b0;
            sign_large_r2   <= 1'b0;
            effective_sub_r2<= 1'b0;
            exp_large_r2    <= 8'd0;
            exp_diff_r2     <= 8'd0;
            man_large_r2    <= 27'd0;
            man_small_r2    <= 27'd0;
        end else begin
            a_is_nan_r2     <= a_is_nan_r1;
            b_is_nan_r2     <= b_is_nan_r1;
            a_is_inf_r2     <= a_is_inf_r1;
            b_is_inf_r2     <= b_is_inf_r1;
            sign_a_r2       <= sign_a_r1;
            sign_b_r2       <= sign_b_r1;
            sign_large_r2   <= sign_large;
            effective_sub_r2<= effective_sub;
            exp_large_r2    <= exp_large;
            exp_diff_r2     <= exp_diff;
            man_large_r2    <= man_large;
            man_small_r2    <= man_small;
        end
    end

    // ════════════════════════════════════════════════════════════
    // Stage 3 (combinational): align shift of smaller mantissa
    // ════════════════════════════════════════════════════════════
    wire [26:0] man_small_shifted = (exp_diff_r2 >= 8'd27) ? 27'd0 : (man_small_r2 >> exp_diff_r2);

    // ── Stage 3 -> Stage 4 registers ──
    reg        a_is_nan_r3, b_is_nan_r3, a_is_inf_r3, b_is_inf_r3;
    reg        sign_a_r3, sign_b_r3, sign_large_r3, effective_sub_r3;
    reg [ 7:0] exp_large_r3;
    reg [26:0] man_large_r3, man_small_shifted_r3;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            a_is_nan_r3        <= 1'b0;
            b_is_nan_r3        <= 1'b0;
            a_is_inf_r3        <= 1'b0;
            b_is_inf_r3        <= 1'b0;
            sign_a_r3          <= 1'b0;
            sign_b_r3          <= 1'b0;
            sign_large_r3      <= 1'b0;
            effective_sub_r3   <= 1'b0;
            exp_large_r3       <= 8'd0;
            man_large_r3       <= 27'd0;
            man_small_shifted_r3 <= 27'd0;
        end else begin
            a_is_nan_r3        <= a_is_nan_r2;
            b_is_nan_r3        <= b_is_nan_r2;
            a_is_inf_r3        <= a_is_inf_r2;
            b_is_inf_r3        <= b_is_inf_r2;
            sign_a_r3          <= sign_a_r2;
            sign_b_r3          <= sign_b_r2;
            sign_large_r3      <= sign_large_r2;
            effective_sub_r3   <= effective_sub_r2;
            exp_large_r3       <= exp_large_r2;
            man_large_r3       <= man_large_r2;
            man_small_shifted_r3 <= man_small_shifted;
        end
    end

    // ════════════════════════════════════════════════════════════
    // Stage 4 (combinational): 28-bit add/subtract + LOD
    // ════════════════════════════════════════════════════════════
    wire [27:0] man_sum_raw = effective_sub_r3 ?
        {1'b0, man_large_r3} - {1'b0, man_small_shifted_r3} :
        {1'b0, man_large_r3} + {1'b0, man_small_shifted_r3};

    wire [4:0] add_lod_pos = lod32({4'd0, man_sum_raw});

    // ── Stage 4 -> Stage 5 registers ──
    reg        a_is_nan_r4, b_is_nan_r4, a_is_inf_r4, b_is_inf_r4;
    reg        sign_a_r4, sign_b_r4, sign_large_r4;
    reg [ 7:0] exp_large_r4;
    reg [27:0] man_sum_raw_r4;
    reg [ 4:0] add_lod_pos_r4;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            a_is_nan_r4    <= 1'b0;
            b_is_nan_r4    <= 1'b0;
            a_is_inf_r4    <= 1'b0;
            b_is_inf_r4    <= 1'b0;
            sign_a_r4      <= 1'b0;
            sign_b_r4      <= 1'b0;
            sign_large_r4  <= 1'b0;
            exp_large_r4   <= 8'd0;
            man_sum_raw_r4 <= 28'd0;
            add_lod_pos_r4 <= 5'd0;
        end else begin
            a_is_nan_r4    <= a_is_nan_r3;
            b_is_nan_r4    <= b_is_nan_r3;
            a_is_inf_r4    <= a_is_inf_r3;
            b_is_inf_r4    <= b_is_inf_r3;
            sign_a_r4      <= sign_a_r3;
            sign_b_r4      <= sign_b_r3;
            sign_large_r4  <= sign_large_r3;
            exp_large_r4   <= exp_large_r3;
            man_sum_raw_r4 <= man_sum_raw;
            add_lod_pos_r4 <= add_lod_pos;
        end
    end

    // ════════════════════════════════════════════════════════════
    // Stage 5 (combinational): normalization shift + pack result
    // ════════════════════════════════════════════════════════════
    wire [31:0] add_man_shifted = man_sum_raw_r4 << (5'd27 - add_lod_pos_r4);

    always @(*) begin
        if (a_is_nan_r4 || b_is_nan_r4) begin
            result = {1'b0, 8'hFF, 23'h400000};  // Quiet NaN
        end
        else if (a_is_inf_r4 && b_is_inf_r4 && (sign_a_r4 != sign_b_r4)) begin
            result = {1'b0, 8'hFF, 23'h400000};  // NaN
        end
        else if (a_is_inf_r4) begin
            result = {sign_a_r4, 8'hFF, 23'd0};
        end
        else if (b_is_inf_r4) begin
            result = {sign_b_r4, 8'hFF, 23'd0};
        end
        else if (man_sum_raw_r4 == 28'd0) begin
            result = {1'b0, 8'd0, 23'd0};  // +0
        end
        else begin
            if (add_lod_pos_r4 < 5'd19) begin
                result = {sign_large_r4, 8'd0, 23'd0};  // underflow -> flush
            end else begin
                result = {sign_large_r4,
                          exp_large_r4 + {3'd0, add_lod_pos_r4} - 8'd26,
                          add_man_shifted[26:4]};
            end
        end
    end

endmodule
