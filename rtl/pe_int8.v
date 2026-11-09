// pe_int8.v — Single Processing Element: INT8 multiply + INT32 accumulate
// Weight-stationary: weight preloaded once; the activation arrives every cycle.
// Outputs the partial sum DOWN the column (registered).  There is deliberately no
// activation passthrough: rtl/systolic_array_int8.v broadcasts the activation across
// each row instead of chaining it, which is what keeps the per-column output skew at
// zero (see that file's header).

module pe_int8 (
    input  wire             clk,
    input  wire             rst_n,

    // Control
    input  wire             weight_load,     // Latch new stationary weight
    input  wire             accum_clear,     // Clear accumulation register
    input  wire             stall,           // Stall pipeline

    // Data
    input  wire [7:0]       activation_in,   // INT8 from left neighbor
    input  wire [7:0]       weight_in,       // INT8 weight (latched on weight_load)
    input  wire [31:0]      partial_sum_in,  // INT32 from top neighbor

    output reg  [31:0]      partial_sum_out  // To bottom neighbor (registered)
);

    // ── Internal registers ──
    reg  [7:0]  weight_reg;
    reg  [7:0]  activation_reg;
    reg  [31:0] accum_reg;

    // ── Arithmetic ──
    wire [15:0] mult_result;   // 8b × 8b = 16b signed
    wire [31:0] add_result;    // 16b + 32b = 32b

    // INT8 signed multiply: 8b × 8b → 16b
    assign mult_result = $signed(weight_reg) * $signed(activation_reg);

    // Sign-extend 16b product to 32b, add to incoming partial sum
    assign add_result = $signed(partial_sum_in) + {{16{mult_result[15]}}, mult_result};

    // ── Registers ──
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            weight_reg     <= 8'd0;
            activation_reg <= 8'd0;
            accum_reg      <= 32'd0;
        end else if (!stall) begin
            if (weight_load)
                weight_reg <= weight_in;

            activation_reg <= activation_in;

            if (accum_clear)
                accum_reg <= 32'd0;
            else
                accum_reg <= add_result;
        end
    end

    // ── Pipeline output: partial sum down the column ──
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            partial_sum_out <= 32'd0;
        end else if (!stall) begin
            partial_sum_out <= accum_clear ? partial_sum_in : add_result;
        end
    end

endmodule
