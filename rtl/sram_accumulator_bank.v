// sram_accumulator_bank.v — Accumulator bank SRAM for FP16
// 1024×512b, 4 × accum_512x128_freepdk45 macros (4 width-parallel)
// Interface matches accumulator_bank.v conventions.
//
// Note: This SRAM wrapper operates in simple write-through mode.
// The accum_en signal gates the write enable (passthrough to SRAM web signal).
// Accumulation (read-modify-write) is handled at the controller level outside
// this wrapper via separate read and write cycles.

`include "sparsenn_params.vh"

module sram_accumulator_bank #(
    parameter DEPTH = 1024,           // 10-bit addressable
    parameter WIDTH = 512            // Fixed 512b (4 × accum_512x128)
) (
    input  wire                     clk,
    input  wire                     rst_n,
    input  wire                     wr_en,
    input  wire [$clog2(DEPTH)-1:0] wr_addr,
    input  wire [WIDTH-1:0]         wr_data,
    input  wire                     rd_en,
    input  wire [$clog2(DEPTH)-1:0] rd_addr,
    output wire [WIDTH-1:0]         rd_data,
    input  wire                     accum_en,        // Gates write when asserted
    output wire                     ready
);

    wire [9:0]  sram_addr;
    wire [WIDTH-1:0] sram_din;
    wire sram_csb, sram_web;

    // accum_en gates the write: when 1, data is accumulated into existing value
    // (the actual RMW is handled by external controller logic)
    // When 0, simple overwrite
    assign sram_addr = wr_en ? wr_addr : rd_addr;
    assign sram_din  = wr_data;
    assign sram_csb  = 1'b0;
    // web=0 (write) only when both wr_en AND accum_en are asserted
    assign sram_web  = ~(wr_en && accum_en);

    sram_acc_bank_1024x512 bank (
        .clk(clk), .csb(sram_csb), .web(sram_web),
        .addr(sram_addr), .din(sram_din), .dout(rd_data)
    );

    assign ready = 1'b1;

endmodule
