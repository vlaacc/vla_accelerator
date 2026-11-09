// sram_acc_buffer_fp16t3.v — Accumulator bank SRAM for FP16T3
// 1024×512b, 4 × accum_512x128_freepdk45 macros
// Interface matches accumulator_bank.v conventions.

`include "sparsenn_params.vh"

module sram_acc_buffer_fp16t3 #(
    parameter DEPTH = 1024,
    parameter WIDTH = 512
) (
    input  wire                     clk,
    input  wire                     rst_n,
    input  wire                     wr_en,
    input  wire [$clog2(DEPTH)-1:0] wr_addr,
    input  wire [WIDTH-1:0]         wr_data,
    input  wire                     rd_en,
    input  wire [$clog2(DEPTH)-1:0] rd_addr,
    output wire [WIDTH-1:0]         rd_data,
    input  wire                     accum_en,
    output wire                     ready
);

    wire [9:0]  sram_addr;
    wire [WIDTH-1:0] sram_din;
    wire sram_csb, sram_web;

    assign sram_addr = wr_en ? wr_addr : rd_addr;
    assign sram_din  = wr_data;
    assign sram_csb  = 1'b0;
    assign sram_web  = ~wr_en;

    sram_acc_bank_1024x512 bank (
        .clk(clk), .csb(sram_csb), .web(sram_web),
        .addr(sram_addr), .din(sram_din), .dout(rd_data)
    );

    assign ready = 1'b1;

endmodule
