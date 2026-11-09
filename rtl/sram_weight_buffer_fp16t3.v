// sram_weight_buffer_fp16t3.v — Double-buffered weight SRAM for FP16T3
// 2 banks × 2048×512b, each bank = 8 scratchpad_1024x128 macros (total 16)
// Single-port SRAM MUX: write (DMA) and read (SA) are time-multiplexed.
// Interface matches weight_buffer.v conventions.

`include "sparsenn_params.vh"

module sram_weight_buffer_fp16t3 #(
    parameter DEPTH = 2048,
    parameter WIDTH = 512
) (
    input  wire                     clk,
    input  wire                     rst_n,
    input  wire                     wr_en,
    input  wire [$clog2(DEPTH)-1:0] wr_addr,
    input  wire [WIDTH-1:0]         wr_data,
    input  wire                     wr_bank_sel,
    input  wire                     rd_en,
    input  wire [$clog2(DEPTH)-1:0] rd_addr,
    output wire [WIDTH-1:0]         rd_data,
    input  wire                     rd_bank_sel,
    input  wire                     swap,
    output wire                     buffer_ready
);

    // SRAM addr/din: mux between write and read
    wire [10:0] sram_addr;
    wire [WIDTH-1:0] sram_din;
    wire sram_csb, sram_web;

    // Write when wr_en=1, else read
    assign sram_addr = wr_en ? {1'b0, wr_addr} : {1'b0, rd_addr};
    assign sram_din  = wr_data;
    assign sram_csb  = 1'b0;              // always enabled
    assign sram_web  = ~wr_en;            // 0=write, 1=read

    wire [WIDTH-1:0] dout_bank0;
    wire [WIDTH-1:0] dout_bank1;

    // Per-bank chip select: only selected bank responds
    wire csb_bank0 = sram_csb | (wr_en ? wr_bank_sel : rd_bank_sel);
    wire csb_bank1 = sram_csb | (wr_en ? ~wr_bank_sel : ~rd_bank_sel);

    sram_weight_bank_2048x512 bank0 (
        .clk(clk), .csb(csb_bank0), .web(sram_web),
        .addr(sram_addr), .din(sram_din), .dout(dout_bank0)
    );

    sram_weight_bank_2048x512 bank1 (
        .clk(clk), .csb(csb_bank1), .web(sram_web),
        .addr(sram_addr), .din(sram_din), .dout(dout_bank1)
    );

    assign rd_data = rd_bank_sel ? dout_bank1 : dout_bank0;
    assign buffer_ready = 1'b1;

endmodule
