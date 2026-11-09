// fp32_div.v — IEEE 754 FP32 divider using shift-subtract (binary long division)
// References Berkeley HardFloat DivSqrtRawFN_small (used by Gemmini Normalizer).
// 27-cycle latency, single-issue handshake: inReady/inValid → outValid.
// result = a / b

`include "sparsenn_params.vh"

module fp32_div (
    input  wire                     clk,
    input  wire                     rst_n,

    input  wire                     inValid,
    output wire                     inReady,
    input  wire [`FP32_WIDTH-1:0]   a,
    input  wire [`FP32_WIDTH-1:0]   b,

    output reg  [`FP32_WIDTH-1:0]   result,
    output reg                      outValid
);

    // ── Decompose inputs ──
    wire sign_a = a[31];  wire sign_b = b[31];
    wire [7:0] exp_a = a[30:23];  wire [7:0] exp_b = b[30:23];
    wire [22:0] man_a = a[22:0];  wire [22:0] man_b = b[22:0];

    wire a_nan = (exp_a == 8'hFF) && (|man_a);
    wire b_nan = (exp_b == 8'hFF) && (|man_b);
    wire a_inf = (exp_a == 8'hFF) && !(|man_a);
    wire b_inf = (exp_b == 8'hFF) && !(|man_b);
    wire a_zero = (exp_a == 8'd0) && !(|man_a);
    wire b_zero = (exp_b == 8'd0) && !(|man_b);

    wire special = a_nan || b_nan || b_zero || (a_inf && b_inf);
    wire result_is_zero = a_zero || b_inf;
    wire result_is_inf  = a_inf && !b_inf;

    // ── Internal state ──
    reg [4:0]  cycle;          // 0=idle, 1..26=active, 27=done
    reg [31:0] sign_r;
    reg [9:0]  exp_diff;       // exponent difference (signed)
    reg [24:0] divisor;        // normalized divisor (1.xxxxx, 25 bits)
    reg [49:0] remainder;      // 2× divisor width
    reg [24:0] quotient;       // accumulated quotient bits

    wire [49:0] rem_shifted = {remainder[48:0], 1'b0};
    wire        sub_take    = rem_shifted[49:25] >= divisor;
    wire [49:0] rem_next    = sub_take ? {rem_shifted[49:25] - divisor, rem_shifted[24:0]}
                                       : rem_shifted;

    assign inReady = (cycle == 5'd0);

    // ── Normalize divisor to [1.0, 2.0) ──
    wire [24:0] divisor_norm = {2'b01, man_b};  // 1.mantissa_b

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            cycle <= 0; result <= 0; outValid <= 0;
            sign_r <= 0; exp_diff <= 0;
            divisor <= 0; remainder <= 0; quotient <= 0;
        end else begin
            outValid <= 0;

            // ── Capture inputs ──
            if (inValid && inReady) begin
                sign_r <= {sign_a ^ sign_b, 1'b0};
                exp_diff <= $signed({1'b0, exp_a}) - $signed({1'b0, exp_b});
                divisor <= divisor_norm;

                if (special) begin
                    cycle <= 5'd27;  // skip to output
                end else begin
                    cycle <= 5'd1;
                    // remainder = {1.mantissa_a, 25'b0}  (50 bits, left-aligned)
                    remainder <= {2'b01, man_a, 25'd0};
                    quotient  <= 25'd0;
                end
            end

            // ── Iterative division: 26 cycles ──
            if (cycle >= 1 && cycle <= 26) begin
                cycle <= cycle + 5'd1;
                remainder <= rem_next;
                if (sub_take)
                    quotient <= quotient | (25'd1 << (5'd25 - cycle));
            end

            // ── Normalize and output ──
            if (cycle == 5'd27) begin
                outValid <= 1'b1;
                cycle <= 0;

                if (special) begin
                    result <= {1'b0, 8'hFF, 23'h400000};  // NaN
                end else if (result_is_zero) begin
                    result <= {sign_a ^ sign_b, 8'd0, 23'd0};
                end else if (result_is_inf) begin
                    result <= {sign_a ^ sign_b, 8'hFF, 23'd0};
                end else begin
                    // Normalize: find leading 1 in 25-bit quotient
                    // Mantissa = quotient bits below leading 1, padded to 23 bits
                    // Exponent = exp_a - exp_b + 127 - normalization_shift
                    reg [4:0] lz;  // leading zero count (0 if Q[24]=1, 1 if Q[23]=1, ...)
                    reg [22:0] final_man;
                    reg [ 7:0] final_exp;

                    casez (quotient)
                        25'b1????????????????????????: begin lz = 5'd0;  final_man = quotient[23:1];  end
                        25'b01???????????????????????: begin lz = 5'd1;  final_man = quotient[22:0];  end
                        25'b001??????????????????????: begin lz = 5'd2;  final_man = {quotient[21:0], 1'b0}; end
                        25'b0001?????????????????????: begin lz = 5'd3;  final_man = {quotient[20:0], 2'b0}; end
                        25'b00001????????????????????: begin lz = 5'd4;  final_man = {quotient[19:0], 3'b0}; end
                        25'b000001???????????????????: begin lz = 5'd5;  final_man = {quotient[18:0], 4'b0}; end
                        default:                       begin lz = 5'd6;  final_man = {quotient[17:0], 5'b0}; end
                    endcase

                    // result_exp = exp_a - exp_b + 127 - lz
                    // exp_diff is 10-bit signed. Extract 8-bit biased exponent.
                    final_exp = exp_diff[7:0] + 8'd127 - {3'd0, lz};
                    result <= {sign_a ^ sign_b, final_exp, final_man};
                end
            end
        end
    end

endmodule
