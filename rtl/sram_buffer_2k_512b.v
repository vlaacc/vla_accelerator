// sram_buffer_2k_512b.v — 2048×512b SRAM buffer for FP16T3 weight storage
// Behavioral DFF implementation (Phase 1). Phase 2 replaces with
// OpenRAM scratchpad_1024x128 macros (8 per bank, 16 total with ping-pong).
//
// Interface matches weight_buffer.v conventions for drop-in integration.

module sram_buffer_2k_512b #(
    parameter DEPTH = 2048,
    parameter WIDTH = 512
) (
    input  wire                     clk,
    input  wire                     rst_n,

    input  wire                     wr_en,
    input  wire [$clog2(DEPTH)-1:0] wr_addr,
    input  wire [WIDTH-1:0]         wr_data,

    input  wire                     rd_en,
    input  wire [$clog2(DEPTH)-1:0] rd_addr,
    output wire [WIDTH-1:0]         rd_data
);

    reg [WIDTH-1:0] mem [0:DEPTH-1];

    always @(posedge clk) begin
        if (wr_en)
            mem[wr_addr] <= wr_data;
    end

    assign rd_data = mem[rd_addr];

endmodule
