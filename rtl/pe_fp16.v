// pe_fp16.v — Single Processing Element: FP16 multiply + FP32 accumulate
// Pipelined MAC. In a weight-stationary systolic array the PE receives an
// activation from the left, a preloaded stationary weight, and a partial sum
// from above. It forwards the activation to the right and the accumulated
// partial sum downward.
//
// Pipeline (register stages):
//   fp16_mul : 1 register stage   (2 "logic" stages, 1 reg boundary)
//   fp32_add : 4 register stages  (5 "logic" stages, 4 reg boundaries)
//   MAC latency = 1 + 4 = 5 cycles (activation_in -> add_result)
//
//   The activation path is delayed by the same 5 cycles through an explicit
//   shift chain (act_delay[0:4]) so activation_out stays wavefront-aligned
//   with partial_sum_out. partial_sum_in is delayed 1 cycle (ps_align) so it
//   lines up with the multiplier's output before entering the adder.
//
//   Throughput: 1 result/cycle (full pipeline).  Latency: 5 cycles per PE.
//
// Note: fp16_mul / fp32_add have no stall input (they are free-running
// pipelines). The PE-level registers are gated by `stall`; this is sufficient
// because `stall` is currently never asserted by the FP16 controller.

`include "sparsenn_params.vh"

module pe_fp16 (
    input  wire                        clk,
    input  wire                        rst_n,

    // Control
    input  wire                        weight_load,      // Latch new stationary weight
    input  wire                        accum_clear,      // Pass partial_sum through (drop own contribution)
    input  wire                        stall,            // Stall pipeline for sparse skip cycles

    // Data signals
    input  wire [`FP16_WIDTH-1:0]      activation_in,    // From left neighbor
    input  wire [`FP16_WIDTH-1:0]      weight_in,        // New weight value (latched on weight_load)
    input  wire [`FP32_WIDTH-1:0]      partial_sum_in,   // From top neighbor

    output reg  [`FP16_WIDTH-1:0]      activation_out,   // To right neighbor (registered)
    output reg  [`FP32_WIDTH-1:0]      partial_sum_out   // To bottom neighbor (registered)
);

    // ── Pipeline latencies (register-stage counts, measured) ──
    localparam MUL_LATENCY = 1;                          // fp16_mul: 1 register stage
    localparam ADD_LATENCY = 4;                          // fp32_add: 4 register stages
    localparam ACT_DELAY    = MUL_LATENCY + ADD_LATENCY; // = 5, matches MAC latency

    // ── Stationary weight (preloaded, not part of the flowing pipeline) ──
    reg  [`FP16_WIDTH-1:0] weight_reg;

    // ── Activation delay chain (keeps activation aligned with partial_sum) ──
    reg  [`FP16_WIDTH-1:0] act_delay [0:ACT_DELAY-1];

    // ── partial_sum alignment register (compensates the mul's 1-cycle latency) ──
    reg  [`FP32_WIDTH-1:0] ps_align;

    // ── accum_clear alignment register (compensates the mul's 1-cycle latency) ──
    reg                    clear_d1;

    // ── Arithmetic unit wires ──
    wire [`FP32_WIDTH-1:0] mult_result;
    wire [`FP32_WIDTH-1:0] mult_eff;   // multiply contribution, zeroed on accum_clear
    wire [`FP32_WIDTH-1:0] add_result;

    // ── FP16 multiply: weight * activation -> FP32 (1 register stage) ──
    fp16_mul mul_unit (
        .clk   (clk),
        .rst_n (rst_n),
        .a     (weight_reg),
        .b     (activation_in),
        .result(mult_result)
    );

    // accum_clear -> drop this PE's weight*activation contribution (pass-through)
    assign mult_eff = clear_d1 ? `FP32_WIDTH'd0 : mult_result;

    // ── FP32 accumulate: partial_sum + (weight*activation) (4 register stages) ──
    fp32_add add_unit (
        .clk   (clk),
        .rst_n (rst_n),
        .a     (mult_eff),
        .b     (ps_align),
        .result(add_result)
    );

    integer i;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            weight_reg      <= `FP16_WIDTH'd0;
            clear_d1        <= 1'b0;
            ps_align        <= `FP32_WIDTH'd0;
            for (i = 0; i < ACT_DELAY; i = i + 1)
                act_delay[i] <= `FP16_WIDTH'd0;
            activation_out  <= `FP16_WIDTH'd0;
            partial_sum_out <= `FP32_WIDTH'd0;
        end else if (!stall) begin
            if (weight_load)
                weight_reg <= weight_in;

            // Activation shift chain: delay activation_in by ACT_DELAY cycles.
            act_delay[0] <= activation_in;
            for (i = 1; i < ACT_DELAY; i = i + 1)
                act_delay[i] <= act_delay[i-1];

            // Align partial_sum and accum_clear with the multiplier output.
            ps_align <= partial_sum_in;
            clear_d1 <= accum_clear;

            // Registered outputs (both delayed identically, wavefront aligned).
            activation_out  <= act_delay[ACT_DELAY-1];
            partial_sum_out <= add_result;
        end
    end

endmodule
