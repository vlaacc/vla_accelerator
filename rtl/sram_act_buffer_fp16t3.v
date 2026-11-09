// sram_act_buffer_fp16t3.v — Double-buffered activation SRAM for FP16T3
// 2 banks × 2048×256b, each bank = 4 scratchpad_1024x128 macros (total 8)
// Interface matches activation_buffer.v conventions.

`include "sparsenn_params.vh"

module sram_act_buffer_fp16t3 #(
    parameter DEPTH = 2048,
    parameter WIDTH = 256
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

    wire [10:0] sram_addr;
    wire [WIDTH-1:0] sram_din;
    wire sram_web;

    assign sram_addr = wr_en ? {1'b0, wr_addr} : {1'b0, rd_addr};
    assign sram_din  = wr_data;
    assign sram_web  = ~wr_en;

    wire csb_bank0 = wr_en ? wr_bank_sel : rd_bank_sel;
    wire csb_bank1 = wr_en ? ~wr_bank_sel : ~rd_bank_sel;

    wire [WIDTH-1:0] dout_bank0;
    wire [WIDTH-1:0] dout_bank1;

    sram_act_bank_2048x256 bank0 (
        .clk(clk), .csb(csb_bank0), .web(sram_web),
        .addr(sram_addr), .din(sram_din), .dout(dout_bank0)
    );

    sram_act_bank_2048x256 bank1 (
        .clk(clk), .csb(csb_bank1), .web(sram_web),
        .addr(sram_addr), .din(sram_din), .dout(dout_bank1)
    );

    assign rd_data = rd_bank_sel ? dout_bank1 : dout_bank0;
    assign buffer_ready = 1'b1;

endmodule
