// weight_buffer.v — Double-buffered weight scratchpad
// Implements ping-pong buffering to overlap weight DMA with computation.
// While one buffer (ping) streams weights to the systolic array, the other (pong)
// receives new weight data from the host via DMA.
//
// Interface mirrors activation_buffer.v for symmetric control.

`include "sparsenn_params.vh"

module weight_buffer #(
    parameter DEPTH = `WEIGHT_BUF_DEPTH,
    parameter WIDTH = `M_PE_COLS * `FP16_WIDTH
) (
    input  wire                     clk,
    input  wire                     rst_n,

    // Write port (from host/DMA reader)
    input  wire                     wr_en,
    input  wire [$clog2(DEPTH)-1:0] wr_addr,
    input  wire [WIDTH-1:0]         wr_data,
    input  wire                     wr_bank_sel,    // Which bank to write: 0=ping, 1=pong

    // Read port (to systolic array weight input)
    input  wire                     rd_en,
    input  wire [$clog2(DEPTH)-1:0] rd_addr,
    output wire [WIDTH-1:0]         rd_data,
    input  wire                     rd_bank_sel,    // Which bank to read: 0=ping, 1=pong

    // Buffer swap control
    input  wire                     swap,            // Unused (bank selects driven explicitly)
    output wire                     buffer_ready     // Active read buffer has data
);

    reg [WIDTH-1:0] ping [0:DEPTH-1];
    reg [WIDTH-1:0] pong [0:DEPTH-1];

    // ── Write to selected bank ──
    always @(posedge clk) begin
        if (wr_en) begin
            if (wr_bank_sel)
                pong[wr_addr] <= wr_data;
            else
                ping[wr_addr] <= wr_data;
        end
    end

    // ── Read from selected bank (combinational) ──
    assign rd_data = rd_bank_sel ? pong[rd_addr] : ping[rd_addr];

    // ── Buffer ready: always 1 (async read) ──
    assign buffer_ready = 1'b1;

endmodule
