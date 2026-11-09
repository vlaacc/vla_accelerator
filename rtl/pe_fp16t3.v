// pe_fp16t3.v — Processing Element: FP16 activation × Ternary weight (+1/0/-1)
// No FP16 multiplier needed — uses sign-flip + FP32 add/sub.
// Reuses the existing fp32_add module.

`include "sparsenn_params.vh"

module pe_fp16t3 (
    input  wire                        clk,
    input  wire                        rst_n,

    input  wire                        weight_load,
    input  wire                        accum_clear,
    input  wire                        stall,

    input  wire [`FP16_WIDTH-1:0]      activation_in,
    input  wire [1:0]                  weight_in,         // 2b ternary: 00=0, 01=+1, 10=-1
    input  wire [`FP32_WIDTH-1:0]      partial_sum_in,

    output reg  [`FP16_WIDTH-1:0]      activation_out,
    output reg  [`FP32_WIDTH-1:0]      partial_sum_out
);

    // Internal registers
    reg  [1:0]                 weight_reg;
    reg  [`FP16_WIDTH-1:0]     activation_reg;
    reg  [`FP32_WIDTH-1:0]     accum_reg;

    // FP16 → FP32 conversion (exp + 112, man zero-pad)
    wire [`FP32_WIDTH-1:0] act_fp32;
    wire sign   = activation_reg[15];
    wire [4:0]  exp   = activation_reg[14:10];
    wire [9:0]  man   = activation_reg[9:0];
    wire is_zero = (exp == 5'd0) && (man == 10'd0);
    wire is_denorm = (exp == 5'd0) && (man != 10'd0);
    wire is_nan_inf = (exp == 5'd31);

    assign act_fp32 = is_nan_inf ? {sign, 8'hFF, man, 13'd0} :
                      is_zero     ? {sign, 8'd0, 23'd0} :
                      is_denorm   ? {sign, 8'd0, man, 13'd0} :
                      {sign, exp + 8'd112, man, 13'd0};

    // Ternary decode → sign-flipped operand for fp32_add
    wire is_pos = (weight_reg == 2'b01);
    wire is_neg = (weight_reg == 2'b10);
    wire accum_en = is_pos || is_neg;  // weight = 0 → passthrough

    wire [`FP32_WIDTH-1:0] addend = is_neg ? {~act_fp32[31], act_fp32[30:0]} : act_fp32;

    // FP32 accumulate
    wire [`FP32_WIDTH-1:0] add_result;
    fp32_add add_unit (
        .a     (addend),
        .b     (partial_sum_in),
        .result(add_result)
    );

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            weight_reg     <= 2'd0;
            activation_reg <= `FP16_WIDTH'd0;
            accum_reg      <= `FP32_WIDTH'd0;
        end else if (!stall) begin
            if (weight_load)
                weight_reg <= weight_in;

            activation_reg <= activation_in;

            if (accum_clear)
                accum_reg <= `FP32_WIDTH'd0;
            else if (accum_en)
                accum_reg <= add_result;
            else
                accum_reg <= partial_sum_in;
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            activation_out  <= `FP16_WIDTH'd0;
            partial_sum_out <= `FP32_WIDTH'd0;
        end else if (!stall) begin
            activation_out  <= activation_in;
            partial_sum_out <= accum_clear ? partial_sum_in :
                               accum_en     ? add_result : partial_sum_in;
        end
    end

endmodule
