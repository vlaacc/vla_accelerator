// accumulator_bank.v — FP32 partial-sum accumulator storage
// Stores partial sums between tiles.
// After each tile completes, partial sums drain from the systolic array
// bottom edge into the accumulator bank.
// Before the next tile, partial sums are reloaded from the accumulator bank
// into the systolic array top edge.

`include "sparsenn_params.vh"

module accumulator_bank #(
    parameter DEPTH = `ACC_BUF_DEPTH,
    parameter WIDTH = `N_PE_ROWS * `FP32_WIDTH
) (
    input  wire                     clk,
    input  wire                     rst_n,

    // Write (drain from systolic array bottom)
    input  wire                     wr_en,
    input  wire [$clog2(DEPTH)-1:0] wr_addr,
    input  wire [WIDTH-1:0]         wr_data,

    // Read (reload to systolic array top)
    input  wire                     rd_en,
    input  wire [$clog2(DEPTH)-1:0] rd_addr,
    output reg  [WIDTH-1:0]         rd_data,

    // Accumulation mode (add incoming data instead of overwrite)
    input  wire                     accum_en,
    output wire                     ready
);

    reg [WIDTH-1:0] mem [0:DEPTH-1];

    // ── Write (with optional accumulate mode) ──
    always @(posedge clk) begin
        if (wr_en) begin
            if (accum_en)
                mem[wr_addr] <= mem[wr_addr] + wr_data;
            else
                mem[wr_addr] <= wr_data;
        end
    end

    // ── Read ──
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            rd_data <= {WIDTH{1'b0}};
        else if (rd_en)
            rd_data <= mem[rd_addr];
    end

    assign ready = 1'b1;

endmodule
