// sub_sum_fifo.v — Register-based FIFO decoupling LUT engine from SIMD vector unit
//
// Buffers one full "group" of sub_sum between the LUT engine (producer) and the
// SIMD vector unit (consumer):
//   WIDTH = R_CORES * N_CH_PER_CORE * SUBSUM_WIDTH = 2 * 256 * 16 = 8192 bits
//
// A register implementation is used (no SRAM): 8192b x DEPTH flip-flops.
//   DEPTH=4 => 32 Kbit = 4 KB of storage.
//
// Backpressure contract:
//   full  -> the W R I T E must not be attempted (the caller's write enable is
//            gated with `!full`, which SILENTLY DROPS the entry)
//   empty -> consumer (SIMD) must stall, no group read
//
// full/empty are derived combinationally from a registered occupancy `count`,
// giving a standard 1-cycle-latency status.
//
// ⚠️ `full` alone is NOT a working backpressure contract, and `full` is
//    effectively a late signal: the producer must be stopped EARLIER than the
//    instant the FIFO fills, because entries already in the producer's pipeline
//    keep arriving for ST_RES_DELAY cycles after the stop is asserted.  The
//    occupancy is therefore exported as `level` so the caller can stop the
//    producer while there is still headroom — see FIFO_BP_LEVEL in
//    rtl/sparse_ternary_top.v and the analysis/harness in sim/tb_fifo_rate.v.
//
`timescale 1ns / 1ps

module sub_sum_fifo #(
    parameter WIDTH = 8192,
    parameter DEPTH = 4
) (
    input  wire                 clk,
    input  wire                 rst_n,

    input  wire                 wr_en,
    input  wire [WIDTH-1:0]     wr_data,

    input  wire                 rd_en,
    output wire [WIDTH-1:0]     rd_data,

    output wire                 full,
    output wire                 empty,
    output wire [$clog2(DEPTH):0] level   // current occupancy (0..DEPTH)
);

    localparam PTR_W = (DEPTH > 1) ? $clog2(DEPTH) : 1;

    reg [WIDTH-1:0] mem [0:DEPTH-1];
    reg [PTR_W-1:0] wr_ptr;
    reg [PTR_W-1:0] rd_ptr;
    reg [PTR_W:0]   count;   // valid entries: 0 .. DEPTH

    wire wr_ok = wr_en && !full;
    wire rd_ok = rd_en && !empty;

    // Wrap at DEPTH-1, NOT at the pointer's own width.  `PTR_W = $clog2(DEPTH)` is the
    // smallest width that can HOLD DEPTH-1, which for a non-power-of-two depth is wider
    // than the storage: at DEPTH=24 PTR_W is 5, so a plain `ptr + 1` counts 0..31 while
    // `mem` is only `[0:23]`.  Pointers 24..31 then index out of range -- the write is
    // discarded and the read returns X.  Measured: an acc row that is never written,
    // which is how this was found (FIFO DEPTH 24 -> sim-st-top-w4 reads got=x).
    // DEPTH=32 hid it because 32 is a power of two and the two wraps coincide.
    wire [PTR_W-1:0] wr_nxt = (wr_ptr == PTR_W'(DEPTH-1)) ? {PTR_W{1'b0}} : (wr_ptr + 1'b1);
    wire [PTR_W-1:0] rd_nxt = (rd_ptr == PTR_W'(DEPTH-1)) ? {PTR_W{1'b0}} : (rd_ptr + 1'b1);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wr_ptr <= {PTR_W{1'b0}};
            rd_ptr <= {PTR_W{1'b0}};
            count  <= {PTR_W+1{1'b0}};
        end else begin
            if (wr_ok) begin
                mem[wr_ptr] <= wr_data;
                wr_ptr      <= wr_nxt;
            end
            if (rd_ok)
                rd_ptr <= rd_nxt;

            case ({wr_ok, rd_ok})
                2'b10:   count <= count + 1'b1;
                2'b01:   count <= count - 1'b1;
                default: count <= count;
            endcase
        end
    end

    assign rd_data = mem[rd_ptr];
    assign full    = (count == DEPTH);
    assign empty   = (count == 0);
    assign level   = count;

endmodule
