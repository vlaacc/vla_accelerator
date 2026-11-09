// sram_acc_buffer.v — Double-buffered accumulator SRAM (per-core)
// 2 banks × 512×512b, each bank = 4 scratchpad_1024x128_freepdk45 macros (total 8 per core)
// Interface matches weight/activation buffer conventions.

`include "sparsenn_params.vh"

module sram_acc_buffer #(
    parameter DEPTH = 512,           // Per-bank depth (addressed by 9 bits)
    parameter WIDTH = 512            // Fixed 512b (4 × scratchpad_1024x128)
) (
    input  wire                     clk,
    input  wire                     rst_n,
    input  wire                     wr_en,
    input  wire [$clog2(DEPTH)-1:0] wr_addr,
    input  wire [WIDTH-1:0]         wr_data,
    input  wire                     wr_bank_sel,    // 0=ping, 1=pong
    input  wire                     rd_en,
    input  wire [$clog2(DEPTH)-1:0] rd_addr,
    output wire [WIDTH-1:0]         rd_data,
    input  wire                     rd_bank_sel,    // 0=ping, 1=pong
    input  wire                     swap,            // Unused
    output wire                     buffer_ready
);

    wire [8:0]  sram_addr;
    wire [WIDTH-1:0] sram_din;
    wire sram_web;

    assign sram_addr = wr_en ? wr_addr : rd_addr;
    assign sram_din  = wr_data;
    assign sram_web  = ~wr_en;

    wire csb_bank0 = (wr_en ?  wr_bank_sel : rd_bank_sel);
    wire csb_bank1 = (wr_en ? ~wr_bank_sel : ~rd_bank_sel);

    wire [WIDTH-1:0] dout_bank0;
    wire [WIDTH-1:0] dout_bank1;

    sram_acc_bank_512x512 bank0 (
        .clk(clk), .csb(csb_bank0), .web(sram_web),
        .addr(sram_addr), .din(sram_din), .dout(dout_bank0)
    );

    sram_acc_bank_512x512 bank1 (
        .clk(clk), .csb(csb_bank1), .web(sram_web),
        .addr(sram_addr), .din(sram_din), .dout(dout_bank1)
    );

    assign rd_data = rd_bank_sel ? dout_bank1 : dout_bank0;
    assign buffer_ready = 1'b1;

endmodule
