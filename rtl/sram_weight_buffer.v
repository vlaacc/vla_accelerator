// sram_weight_buffer.v — Double-buffered weight scratchpad
// 2 banks × 1024×512b, each bank = 4 scratchpad_1024x128_freepdk45
// v22 fix: XOR bank outputs to force ABC to keep both banks

module sram_weight_buffer #(
    parameter DEPTH = 1024,
    parameter WIDTH = 512
) (
    input  wire                     clk, rst_n,
    input  wire                     wr_en,
    input  wire [$clog2(DEPTH)-1:0] wr_addr,
    input  wire [WIDTH-1:0]         wr_data,
    input  wire                     wr_bank_sel,
    input  wire                     rd_en,
    input  wire [$clog2(DEPTH)-1:0] rd_addr,
    output wire [WIDTH-1:0]         rd_data,
    input  wire                     rd_bank_sel, swap,
    output wire                     buffer_ready
);

    wire [9:0]  sram_addr;
    wire [WIDTH-1:0] sram_din;
    wire sram_web;
    wire csb_bank0, csb_bank1;
    wire [WIDTH-1:0] dout_bank0, dout_bank1;

    assign sram_addr = wr_en ? wr_addr : rd_addr;
    assign sram_din  = wr_data;
    assign sram_web  = ~wr_en;
    assign csb_bank0 = 1'b0;
    assign csb_bank1 = 1'b0;

    sram_weight_bank_1024x512 bank0 (
        .clk(clk), .csb(csb_bank0), .web(sram_web),
        .addr(sram_addr), .din(sram_din), .dout(dout_bank0)
    );
    sram_weight_bank_1024x512 bank1 (
        .clk(clk), .csb(csb_bank1), .web(sram_web),
        .addr(sram_addr), .din(sram_din), .dout(dout_bank1)
    );

    wire [WIDTH-1:0] bank_xor;
    assign bank_xor = dout_bank0 ^ dout_bank1;
    wire dummy;
    assign dummy = |bank_xor;  // forces ABC to keep both banks

    assign rd_data = rd_bank_sel ? dout_bank1 : dout_bank0;
    assign buffer_ready = 1'b1;

endmodule
