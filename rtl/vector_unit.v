// vector_unit.v — Vector processing for softmax, layernorm, ReLU, GELU
// 16-lane fully parallel architecture.
// Each lane has dedicated fp32_add, fp32_mul, fp32_exp, fp32_gelu, fp32_sqrt.
// Reduction trees (max/sum) are combinational. Shared fp32_recip for post-reduction.
//
// Op codes: 0=ReLU 1=Softmax 2=LayerNorm 3=GELU 4=VecMul 5=VecScale 6=VecSqrt 7=VecAdd

`include "sparsenn_params.vh"

module vector_unit #(
    parameter LANES = 16
) (
    input  wire                     clk,
    input  wire                     rst_n,

    // Data in (from SA drain)
    input  wire                     data_valid,
    input  wire [LANES*`FP32_WIDTH-1:0] data_in,
    output wire                     data_ready,

    // Second operand (for element-wise multiply)
    input  wire                     data_valid_b,
    input  wire [LANES*`FP32_WIDTH-1:0] data_in_b,
    output wire                     data_ready_b,

    // Data out (to acc bank or DMA)
    output wire                     result_valid,
    output wire [LANES*`FP32_WIDTH-1:0] result_out,
    input  wire                     result_ready,

    // Control
    input  wire                     start,
    input  wire [2:0]               op,        // 0=ReLU 1=Softmax 2=LN 3=GELU 4=VecMul 5=VecScale 6=VecSqrt 7=VecAdd
    input  wire [`FP32_WIDTH-1:0]   scale,     // layer norm gamma / vec scale factor (broadcast)
    input  wire                     stream_mode = 1'b0, // 1 = VecScale streams 1 chunk/cycle (bypass FSM)
    output wire                     done,
    output wire                     busy,

    // Debug
    output wire [4:0]               dbg_state,
    output wire [`FP32_WIDTH-1:0]   dbg_accum,
    output wire [`FP32_WIDTH-1:0]   dbg_stat,
    output wire [3:0]               dbg_idx
);

    // ── State encoding ──
    localparam S_IDLE       = 5'd0;
    localparam S_LOAD       = 5'd1;
    localparam S_LOAD_B     = 5'd2;
    localparam S_REDUCE_MAX = 5'd3;
    localparam S_REDUCE_SUM = 5'd4;
    localparam S_SUB_SCALAR = 5'd5;
    localparam S_EXP_ALL    = 5'd6;
    localparam S_SQUARE_ALL = 5'd7;
    localparam S_GELU_ALL   = 5'd8;
    localparam S_SQRT_ALL   = 5'd9;
    localparam S_ELTW_MUL   = 5'd10;
    localparam S_ELTW_ADD   = 5'd11;
    localparam S_SCALE_ALL  = 5'd12;
    localparam S_RECIP      = 5'd13;
    localparam S_MEAN_DIV   = 5'd14;
    localparam S_ADJUST_VAR = 5'd15;
    localparam S_SQRT_STAT  = 5'd16;
    localparam S_STAT_MUL   = 5'd17;
    localparam S_LN_SCALE   = 5'd18;
    localparam S_OUTPUT     = 5'd19;
    localparam S_DONE       = 5'd20;

    // ── Registers ──
    reg  [4:0]              state, next_state;
    reg  [`FP32_WIDTH-1:0]  rf    [0:LANES-1];
    reg  [`FP32_WIDTH-1:0]  rf_b  [0:LANES-1];
    reg  [`FP32_WIDTH-1:0]  rf_ln [0:LANES-1];  // saved mean-subtracted values for LN
    reg  [`FP32_WIDTH-1:0]  stat;
    reg                     ln_pass;   // 0=first LN sum (mean), 1=second LN sum (variance)

    // ── Pipeline wait counter ──
    // fp32_add/fp32_mul are 5-stage pipelines; the FSM must dwell a per-state
    // number of cycles before the pipelined result is valid. wait_cnt is loaded
    // with (delay-1) on state entry and counts down; a state completes (and its
    // datapath result is sampled) when wait_cnt == 0.
    reg [4:0] wait_cnt;

    // ── Streaming VecScale (op=5) pipeline ──
    // When stream_mode is high, VecScale bypasses the FSM and streams:
    //   data_valid every cycle → fp32_mul (4 register stages) → result 4 cycles later.
    // Throughput is 1 chunk/cycle (vs 8 cycles/chunk for the serial FSM).
    wire stream_scale = stream_mode && (op == 3'd5);
    reg  [3:0] stream_valid;   // valid-bit shift register (fp32_mul latency = 4)

    // ── Constants ──
    wire [`FP32_WIDTH-1:0] EPS = 32'h3727C5AC;  // 1e-6

    // Per-state dwell time in cycles (for the pipeline wait counter).
    // 5-cycle states use fp32_add/fp32_mul; the 4-level sum tree needs 4*5=20.
    function [4:0] state_delay;
        input [4:0] s;
        begin
            case (s)
                S_SUB_SCALAR, S_SQUARE_ALL, S_ELTW_MUL, S_ELTW_ADD,
                S_SCALE_ALL, S_LN_SCALE, S_STAT_MUL, S_ADJUST_VAR:
                    state_delay = 5'd5;
                S_REDUCE_SUM:
                    state_delay = 5'd20;   // 4-level fp32_add sum tree (4*5)
                default:
                    state_delay = 5'd1;    // combinational (1-cycle) + handshake states
            endcase
        end
    endfunction

    // ── Per-lane unit output wires ──
    wire [`FP32_WIDTH-1:0] lane_add_out  [0:LANES-1];
    wire [`FP32_WIDTH-1:0] lane_mul_out  [0:LANES-1];
    wire [`FP32_WIDTH-1:0] lane_exp_out  [0:LANES-1];
    wire [`FP32_WIDTH-1:0] lane_gelu_out [0:LANES-1];
    wire [`FP32_WIDTH-1:0] lane_sqrt_out [0:LANES-1];

    // ── Reduction tree wires ──
    // Max tree
    wire [`FP32_WIDTH-1:0] max_l0 [0:7];
    wire [`FP32_WIDTH-1:0] max_l1 [0:3];
    wire [`FP32_WIDTH-1:0] max_l2 [0:1];
    wire [`FP32_WIDTH-1:0] max_result;

    // Sum tree
    wire [`FP32_WIDTH-1:0] sum_l0_out [0:7];
    wire [`FP32_WIDTH-1:0] sum_l1_out [0:3];
    wire [`FP32_WIDTH-1:0] sum_l2_out [0:1];
    wire [`FP32_WIDTH-1:0] sum_result;

    // Shared recip
    wire [`FP32_WIDTH-1:0] recip_out;

    // ═══════════════════════════════════════════════════════════════
    // Per-lane generate: 16 lanes, each with 5 arithmetic units
    // ═══════════════════════════════════════════════════════════════

    genvar g;
    generate
        for (g = 0; g < LANES; g = g + 1) begin : lane

            // ── Input muxing for fp32_add ──
            wire [`FP32_WIDTH-1:0] lane_add_a;
            wire [`FP32_WIDTH-1:0] lane_add_b;
            assign lane_add_a = (state == S_SUB_SCALAR)             ? rf[g] :
                                (state == S_ELTW_ADD)               ? rf[g] :
                                (state == S_ADJUST_VAR && g == 0)   ? stat :
                                32'd0;
            assign lane_add_b = (state == S_SUB_SCALAR)             ? {~stat[31], stat[30:0]} :
                                (state == S_ELTW_ADD)               ? rf_b[g] :
                                (state == S_ADJUST_VAR && g == 0)   ? EPS :
                                32'd0;
            fp32_add u_add (.clk(clk), .rst_n(rst_n), .a(lane_add_a), .b(lane_add_b), .result(lane_add_out[g]));

            // ── Input muxing for fp32_mul ──
            wire [`FP32_WIDTH-1:0] lane_mul_a;
            wire [`FP32_WIDTH-1:0] lane_mul_b;
            assign lane_mul_a = (stream_scale)                      ? data_in[g*`FP32_WIDTH +: `FP32_WIDTH] :
                                (state == S_SQUARE_ALL)             ? rf[g] :
                                (state == S_ELTW_MUL)               ? rf[g] :
                                (state == S_SCALE_ALL)              ? rf[g] :
                                (state == S_LN_SCALE)               ? rf_ln[g] :
                                (state == S_STAT_MUL && g == 0)     ? stat :
                                32'd0;
            assign lane_mul_b = (stream_scale)                      ? scale :
                                (state == S_SQUARE_ALL)             ? rf[g] :
                                (state == S_ELTW_MUL)               ? rf_b[g] :
                                (state == S_SCALE_ALL)              ? ((op == 3'd5) ? scale : stat) :
                                (state == S_LN_SCALE)               ? stat :
                                (state == S_STAT_MUL && g == 0)     ? scale :
                                32'd0;
            fp32_mul u_mul (.clk(clk), .rst_n(rst_n), .a(lane_mul_a), .b(lane_mul_b), .result(lane_mul_out[g]));

            // ── Input muxing for fp32_exp ──
            wire [`FP32_WIDTH-1:0] lane_exp_in;
            assign lane_exp_in = (state == S_EXP_ALL) ? rf[g] : 32'd0;
            fp32_exp u_exp (.x(lane_exp_in), .result(lane_exp_out[g]));

            // ── Input muxing for fp32_gelu ──
            wire [`FP32_WIDTH-1:0] lane_gelu_in;
            assign lane_gelu_in = (state == S_GELU_ALL) ? rf[g] : 32'd0;
            fp32_gelu u_gelu (.x(lane_gelu_in), .result(lane_gelu_out[g]));

            // ── Input muxing for fp32_sqrt ──
            wire [`FP32_WIDTH-1:0] lane_sqrt_in;
            assign lane_sqrt_in = (state == S_SQRT_ALL)              ? rf[g] :
                                  (state == S_SQRT_STAT && g == 0)   ? stat :
                                  32'd0;
            fp32_sqrt u_sqrt (.x(lane_sqrt_in), .result(lane_sqrt_out[g]));

        end
    endgenerate

    // ═══════════════════════════════════════════════════════════════
    // Max reduction tree: 15 comparators, 4 levels, combinational
    // ═══════════════════════════════════════════════════════════════

    generate
        // Level 0: 16→8
        for (g = 0; g < 8; g = g + 1) begin : max_l0_gen
            wire signed [31:0] a_s = $signed({~rf[g*2][31], rf[g*2][30:0]});
            wire signed [31:0] b_s = $signed({~rf[g*2+1][31], rf[g*2+1][30:0]});
            assign max_l0[g] = (a_s > b_s) ? rf[g*2] : rf[g*2+1];
        end
        // Level 1: 8→4
        for (g = 0; g < 4; g = g + 1) begin : max_l1_gen
            wire signed [31:0] a_s = $signed({~max_l0[g*2][31], max_l0[g*2][30:0]});
            wire signed [31:0] b_s = $signed({~max_l0[g*2+1][31], max_l0[g*2+1][30:0]});
            assign max_l1[g] = (a_s > b_s) ? max_l0[g*2] : max_l0[g*2+1];
        end
        // Level 2: 4→2
        for (g = 0; g < 2; g = g + 1) begin : max_l2_gen
            wire signed [31:0] a_s = $signed({~max_l1[g*2][31], max_l1[g*2][30:0]});
            wire signed [31:0] b_s = $signed({~max_l1[g*2+1][31], max_l1[g*2+1][30:0]});
            assign max_l2[g] = (a_s > b_s) ? max_l1[g*2] : max_l1[g*2+1];
        end
        // Level 3: 2→1
        if (1) begin : max_l3_gen
            wire signed [31:0] a_s = $signed({~max_l2[0][31], max_l2[0][30:0]});
            wire signed [31:0] b_s = $signed({~max_l2[1][31], max_l2[1][30:0]});
            assign max_result = (a_s > b_s) ? max_l2[0] : max_l2[1];
        end
    endgenerate

    // ═══════════════════════════════════════════════════════════════
    // Sum reduction tree: 15 fp32_add, 4 levels, combinational
    // ═══════════════════════════════════════════════════════════════

    generate
        // Level 0: 16→8
        for (g = 0; g < 8; g = g + 1) begin : sum_l0_gen
            fp32_add u_sum0 (.clk(clk), .rst_n(rst_n), .a(rf[g*2]), .b(rf[g*2+1]), .result(sum_l0_out[g]));
        end
        // Level 1: 8→4
        for (g = 0; g < 4; g = g + 1) begin : sum_l1_gen
            fp32_add u_sum1 (.clk(clk), .rst_n(rst_n), .a(sum_l0_out[g*2]), .b(sum_l0_out[g*2+1]), .result(sum_l1_out[g]));
        end
        // Level 2: 4→2
        for (g = 0; g < 2; g = g + 1) begin : sum_l2_gen
            fp32_add u_sum2 (.clk(clk), .rst_n(rst_n), .a(sum_l1_out[g*2]), .b(sum_l1_out[g*2+1]), .result(sum_l2_out[g]));
        end
        // Level 3: 2→1
        if (1) begin : sum_l3_gen
            fp32_add u_sum3 (.clk(clk), .rst_n(rst_n), .a(sum_l2_out[0]), .b(sum_l2_out[1]), .result(sum_result));
        end
    endgenerate

    // ═══════════════════════════════════════════════════════════════
    // Shared fp32_recip (single-cycle approximation)
    // ═══════════════════════════════════════════════════════════════

    wire [`FP32_WIDTH-1:0] recip_in;
    assign recip_in = (state == S_RECIP) ? stat : 32'd0;
    fp32_recip u_recip (.x(recip_in), .result(recip_out));

    // ═══════════════════════════════════════════════════════════════
    // FSM: Combinational next-state logic
    // ═══════════════════════════════════════════════════════════════

    always @(*) begin
        next_state = state;
        /* verilator lint_off CASEINCOMPLETE */
        case (state)
            S_IDLE: if (start && !stream_scale) next_state = S_LOAD;

            S_LOAD: if (data_valid) begin
                case (op)
                    3'd0: next_state = S_OUTPUT;       // ReLU
                    3'd1: next_state = S_REDUCE_MAX;    // Softmax: find max
                    3'd2: next_state = S_REDUCE_SUM;    // LayerNorm: 1st sum (mean)
                    3'd3: next_state = S_GELU_ALL;      // GELU
                    3'd4: next_state = S_LOAD_B;        // VecMul
                    3'd5: next_state = S_SCALE_ALL;     // VecScale
                    3'd6: next_state = S_SQRT_ALL;      // VecSqrt
                    3'd7: next_state = S_LOAD_B;        // VecAdd
                    default: next_state = S_OUTPUT;
                endcase
            end

            S_LOAD_B: if (data_valid_b) begin
                next_state = (op == 3'd7) ? S_ELTW_ADD : S_ELTW_MUL;
            end

            S_REDUCE_MAX: if (wait_cnt == 0) next_state = S_SUB_SCALAR;

            S_REDUCE_SUM: if (wait_cnt == 0) begin
                if (op == 3'd1)
                    next_state = S_RECIP;           // softmax: sum of exps → 1/sum
                else if (!ln_pass)
                    next_state = S_MEAN_DIV;        // LN: 1st sum → mean = sum/N
                else
                    next_state = S_ADJUST_VAR;      // LN: 2nd sum → var+eps
            end

            S_SUB_SCALAR: if (wait_cnt == 0) begin
                if (op == 3'd1)
                    next_state = S_EXP_ALL;         // softmax: x-max → exp(x-max)
                else
                    next_state = S_SQUARE_ALL;      // LN: x-mean → (x-mean)^2
            end

            S_EXP_ALL:    if (wait_cnt == 0) next_state = S_REDUCE_SUM;   // softmax: sum of exps
            S_SQUARE_ALL: if (wait_cnt == 0) next_state = S_REDUCE_SUM;   // LN: sum of squares

            S_RECIP: if (wait_cnt == 0) begin
                if (op == 3'd1)
                    next_state = S_SCALE_ALL;       // softmax: exp_i * 1/sum
                else
                    next_state = S_STAT_MUL;        // LN: stat * gamma
            end

            S_MEAN_DIV:   if (wait_cnt == 0) next_state = S_SUB_SCALAR;  // LN: subtract mean
            S_ADJUST_VAR: if (wait_cnt == 0) next_state = S_SQRT_STAT;    // LN: sqrt(var+eps)
            S_SQRT_STAT:  if (wait_cnt == 0) next_state = S_RECIP;        // LN: 1/sqrt
            S_STAT_MUL:   if (wait_cnt == 0) next_state = S_LN_SCALE;     // LN: rf * stat

            S_GELU_ALL:  if (wait_cnt == 0) next_state = S_OUTPUT;
            S_SQRT_ALL:  if (wait_cnt == 0) next_state = S_OUTPUT;
            S_ELTW_MUL:  if (wait_cnt == 0) next_state = S_OUTPUT;
            S_ELTW_ADD:  if (wait_cnt == 0) next_state = S_OUTPUT;
            S_SCALE_ALL: if (wait_cnt == 0) next_state = S_OUTPUT;
            S_LN_SCALE:  if (wait_cnt == 0) next_state = S_OUTPUT;

            S_OUTPUT: if (result_ready) next_state = S_DONE;
            S_DONE:   next_state = S_IDLE;
            default:  next_state = S_IDLE;
        endcase
    end

    // ═══════════════════════════════════════════════════════════════
    // FSM: Sequential state + datapath
    // ═══════════════════════════════════════════════════════════════

    integer i;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state    <= S_IDLE;
            stat     <= 32'd0;
            ln_pass  <= 1'b0;
            wait_cnt <= 5'd0;
            for (i = 0; i < LANES; i = i + 1) begin
                rf[i]    <= 32'd0;
                rf_b[i]  <= 32'd0;
                rf_ln[i] <= 32'd0;
            end
        end else begin
            state <= next_state;

            // Pipeline wait counter: load (delay-1) on state entry, then count down.
            if (next_state != state)
                wait_cnt <= state_delay(next_state) - 5'd1;
            else if (wait_cnt != 5'd0)
                wait_cnt <= wait_cnt - 5'd1;

            case (state)
                S_IDLE: begin
                    ln_pass <= 1'b0;
                end

                S_LOAD: begin
                    if (data_valid) begin
                        for (i = 0; i < LANES; i = i + 1)
                            rf[i] <= data_in[i*`FP32_WIDTH +: `FP32_WIDTH];
                    end
                end

                S_LOAD_B: begin
                    for (i = 0; i < LANES; i = i + 1)
                        rf_b[i] <= data_in_b[i*`FP32_WIDTH +: `FP32_WIDTH];
                end

                S_REDUCE_MAX: if (wait_cnt == 0) stat <= max_result;

                S_REDUCE_SUM: if (wait_cnt == 0) begin
                    stat <= sum_result;
                    if (op == 3'd2) ln_pass <= 1'b1;  // mark first LN sum done
                end

                S_SUB_SCALAR: if (wait_cnt == 0) begin
                    for (i = 0; i < LANES; i = i + 1) begin
                        rf[i]    <= lane_add_out[i];
                        rf_ln[i] <= lane_add_out[i];  // save for LN final scale
                    end
                end

                S_EXP_ALL: if (wait_cnt == 0) begin
                    for (i = 0; i < LANES; i = i + 1)
                        rf[i] <= lane_exp_out[i];
                end

                S_SQUARE_ALL: if (wait_cnt == 0) begin
                    for (i = 0; i < LANES; i = i + 1)
                        rf[i] <= lane_mul_out[i];
                end

                S_GELU_ALL: if (wait_cnt == 0) begin
                    for (i = 0; i < LANES; i = i + 1)
                        rf[i] <= lane_gelu_out[i];
                end

                S_SQRT_ALL: if (wait_cnt == 0) begin
                    for (i = 0; i < LANES; i = i + 1)
                        rf[i] <= lane_sqrt_out[i];
                end

                S_ELTW_MUL: if (wait_cnt == 0) begin
                    for (i = 0; i < LANES; i = i + 1)
                        rf[i] <= lane_mul_out[i];
                end

                S_ELTW_ADD: if (wait_cnt == 0) begin
                    for (i = 0; i < LANES; i = i + 1)
                        rf[i] <= lane_add_out[i];
                end

                S_SCALE_ALL: if (wait_cnt == 0) begin
                    for (i = 0; i < LANES; i = i + 1)
                        rf[i] <= lane_mul_out[i];
                end

                S_LN_SCALE: if (wait_cnt == 0) begin
                    for (i = 0; i < LANES; i = i + 1)
                        rf[i] <= lane_mul_out[i];
                end

                S_RECIP:      if (wait_cnt == 0) stat <= recip_out;
                S_MEAN_DIV:   if (wait_cnt == 0) stat <= {stat[31], stat[30:23] - 8'd4, stat[22:0]};  // /16
                S_ADJUST_VAR: if (wait_cnt == 0) stat <= lane_add_out[0];   // lane[0] does stat+eps
                S_SQRT_STAT:  if (wait_cnt == 0) stat <= lane_sqrt_out[0];  // lane[0] sqrt
                S_STAT_MUL:   if (wait_cnt == 0) stat <= lane_mul_out[0];   // lane[0] mul

                S_OUTPUT: begin
                    if (op == 3'd0) begin  // ReLU: max(x, 0)
                        for (i = 0; i < LANES; i = i + 1)
                            if (rf[i][31]) rf[i] <= 32'd0;
                    end
                end

                S_DONE: ln_pass <= 1'b0;

                default: ;
            endcase
        end
    end

    // ── Streaming VecScale valid shift register ──
    // stream_valid[i] is asserted i+1 cycles after a data_valid beat, tracking
    // the fp32_mul pipeline so result_valid aligns with lane_mul_out.
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) stream_valid <= 4'd0;
        else        stream_valid <= {stream_valid[2:0], stream_scale & data_valid};
    end

    // ═══════════════════════════════════════════════════════════════
    // Output assignments
    // ═══════════════════════════════════════════════════════════════

    wire [`FP32_WIDTH-1:0] out_lane [0:LANES-1];
    generate
        for (g = 0; g < LANES; g = g + 1) begin : out_vec
            assign out_lane[g] = rf[g];
        end
    endgenerate

    assign result_out = stream_scale ? {
        lane_mul_out[15], lane_mul_out[14], lane_mul_out[13], lane_mul_out[12],
        lane_mul_out[11], lane_mul_out[10], lane_mul_out[ 9], lane_mul_out[ 8],
        lane_mul_out[ 7], lane_mul_out[ 6], lane_mul_out[ 5], lane_mul_out[ 4],
        lane_mul_out[ 3], lane_mul_out[ 2], lane_mul_out[ 1], lane_mul_out[ 0]
    } : {
        out_lane[15], out_lane[14], out_lane[13], out_lane[12],
        out_lane[11], out_lane[10], out_lane[ 9], out_lane[ 8],
        out_lane[ 7], out_lane[ 6], out_lane[ 5], out_lane[ 4],
        out_lane[ 3], out_lane[ 2], out_lane[ 1], out_lane[ 0]
    };

    assign result_valid = stream_scale ? stream_valid[3] : (state == S_OUTPUT);
    assign data_ready   = stream_scale ? 1'b1 : (state == S_LOAD);
    assign data_ready_b = (state == S_LOAD_B);
    assign done         = (state == S_DONE);
    assign busy         = stream_scale ? (|stream_valid) : (state != S_IDLE && state != S_DONE);

    // Debug
    assign dbg_state = state;
    assign dbg_accum = stat;
    assign dbg_stat  = stat;
    assign dbg_idx   = {12'd0, ln_pass};  // show LN pass on LSB (for debugging)

endmodule
