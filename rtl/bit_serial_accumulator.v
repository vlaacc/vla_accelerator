// bit_serial_accumulator.v — MSB-first bit-serial w8×w4 dot product
//
// Computes  result = sum_{i=0..3} signed(weight[i]) * signed(act[i])
// by iterating over the 8 weight bits, one bit per cycle (MSB first).
//
// TIME-MULTIPLEXED serial implementation: a single 16:1 MUX, a single shifter,
// and a single adder are reused over 8 cycles by a 3-bit state machine, instead
// of unrolling the 8 bit-steps into 8 pipeline stages.  Latency (8 cycles) and
// throughput (1 dot product / 8 cycles) are unchanged, but the datapath area
// drops ~8× (one shifter+adder instead of eight).
//
// Reuses the 16 subset sums already computed by build_unit (all 2^4 sums of
// the 4 activations).  At each bit position b (MSB-first), the 4 weight bits
// form a 4-bit mask that selects one subset sum via a 16:1 MUX; that value is
// then shift-accumulated:  acc = (acc << 1) + subset_sum[mask].
//
// Signed weights are handled by negating the MSB term (two's-complement):
//   weight = -w[7]·2^7 + sum_{b=0..6} w[b]·2^b
// so the first step subtracts subset_sum[mask7] instead of adding it (Horner):
//   acc₀   = -subset_sum[mask₇]
//   accₖ   = (accₖ₋₁ << 1) + subset_sum[mask_{7-k}]
//   result = acc₇
//
// The accumulator is a fixed 14-bit register (matching the previous pipeline's
// final width).  The true dot-product range is ±4096 (4×128×8), and even a loose
// bound on the intermediate Horner values (8160) stays within 14-bit signed
// range, so no intermediate overflow occurs.
//
// subset_sum[mask] is selected with a case statement (constant indices) so the
// synthesizer builds a balanced 16:1 MUX tree instead of a barrel shifter,
// mirroring lut_compute_unit.v's read_gps pattern.  The mask is extracted from
// weight with a variable single-bit select (one 8:1 mux per weight bit lane).
//
// Inputs subset_sum/weight must be held STABLE for the full 8 cycles of a dot
// product (compute_en asserted 8×), exactly as with the previous pipeline; this
// keeps the interface fully compatible with lut_compute_unit.v.
//
// Packed interface (matching build_unit's per-group 112-bit slice):
//   subset_sum[7*i +: 7]  = i-th subset sum (mask i), signed 7-bit
//   weight[8*i +: 8]      = i-th w8 weight, signed 8-bit

module bit_serial_accumulator (
    input  wire                clk,
    input  wire                rst_n,
    input  wire                compute_en,
    input  wire [111:0]        subset_sum,   // 16 signed subset sums (build_unit)
    input  wire [31:0]         weight,       // 4 signed w8 weights
    output reg  [15:0]         result        // signed 16-bit dot product
);

    // ── Bit counter: 0 = MSB (sign bit), 7 = LSB ──
    reg [2:0] bit_cnt;

    // ── Current 4-bit mask (MSB-first) ──
    // bit_cnt selects weight bit position (7 - bit_cnt) of each of the 4 weights:
    //   bit_cnt=0 → weight[31]/weight[23]/weight[15]/weight[7]  (bit 7 = MSB)
    //   bit_cnt=7 → weight[24]/weight[16]/weight[8] /weight[0]  (bit 0 = LSB)
    // Explicit case (constant indices) rather than a variable bit-select, so the
    // synthesizer builds a clean one-hot mux with no barrel shifter or width
    // expansion — mirroring the subset-sum MUX below.
    reg [3:0] mask;
    always @(*) begin
        case (bit_cnt)
            3'd0: mask = {weight[31], weight[23], weight[15], weight[7]};
            3'd1: mask = {weight[30], weight[22], weight[14], weight[6]};
            3'd2: mask = {weight[29], weight[21], weight[13], weight[5]};
            3'd3: mask = {weight[28], weight[20], weight[12], weight[4]};
            3'd4: mask = {weight[27], weight[19], weight[11], weight[3]};
            3'd5: mask = {weight[26], weight[18], weight[10], weight[2]};
            3'd6: mask = {weight[25], weight[17], weight[9],  weight[1]};
            3'd7: mask = {weight[24], weight[16], weight[8],  weight[0]};
            default: mask = 4'd0;
        endcase
    end

    // ── 16:1 MUX (case → balanced mux tree, no dynamic bit-select) ──
    // Constant [hi:lo] slices; subset_sum passed as an argument (rather than
    // read as a module net) to keep iverilog/Verilator behavior identical.
    function [6:0] select_sum;
        input [111:0] ss;
        input [3:0]   msk;
        begin
            case (msk)
                4'd0:  select_sum = ss[6:0];
                4'd1:  select_sum = ss[13:7];
                4'd2:  select_sum = ss[20:14];
                4'd3:  select_sum = ss[27:21];
                4'd4:  select_sum = ss[34:28];
                4'd5:  select_sum = ss[41:35];
                4'd6:  select_sum = ss[48:42];
                4'd7:  select_sum = ss[55:49];
                4'd8:  select_sum = ss[62:56];
                4'd9:  select_sum = ss[69:63];
                4'd10: select_sum = ss[76:70];
                4'd11: select_sum = ss[83:77];
                4'd12: select_sum = ss[90:84];
                4'd13: select_sum = ss[97:91];
                4'd14: select_sum = ss[104:98];
                4'd15: select_sum = ss[111:105];
                default: select_sum = 7'd0;
            endcase
        end
    endfunction

    wire [6:0] selected = select_sum(subset_sum, mask);

    // ── Single shifter + accumulator (14-bit, reused over 8 cycles) ──
    reg  signed [13:0] acc;
    wire signed [13:0] selected_ext = {{7{selected[6]}}, selected};

    // MSB (bit_cnt==0): negate (two's-complement weight MSB term).
    // Otherwise: acc = (acc << 1) + selected (operands sign-extended to 14-bit).
    wire signed [13:0] next_acc = (bit_cnt == 3'd0) ? -selected_ext
                                                    : ((acc << 1) + selected_ext);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            acc     <= 14'sd0;
            bit_cnt <= 3'd0;
            result  <= 16'sd0;
        end else if (compute_en) begin
            acc <= next_acc;
            if (bit_cnt == 3'd7) begin
                result  <= {{2{next_acc[13]}}, next_acc};  // sign-extend 14→16
                bit_cnt <= 3'd0;
            end else begin
                bit_cnt <= bit_cnt + 1'b1;
            end
        end
    end

endmodule
