// sram_weight_bank_2048x512.v — Single bank 2048×512b weight SRAM
// 8 × scratchpad_1024x128_freepdk45: 4 width-parallel × 2 depth-cascade (addr[10]=CS)
// Blackboxed during Yosys synthesis, instantiated as macro by OpenROAD.

module sram_weight_bank_2048x512 (
    input  wire        clk,
    input  wire        csb,        // chip select (active low)
    input  wire        web,        // write enable (active low)
    input  wire [10:0] addr,       // addr[10]=row select
    input  wire [511:0] din,
    output wire [511:0] dout
);

    wire       csb_row0 = csb |  addr[10];
    wire       csb_row1 = csb | ~addr[10];
    wire [9:0] row_addr = addr[9:0];

    wire [511:0] dout_row0;
    wire [511:0] dout_row1;

    genvar i;
    generate
        for (i = 0; i < 4; i = i + 1) begin : row0_slice
            (* blackbox *) scratchpad_1024x128_freepdk45 u_row0 (
                .clk0 (clk), .csb0(csb_row0), .web0(web),
                .addr0(row_addr), .din0(din[i*128 +: 128]),
                .dout0(dout_row0[i*128 +: 128])
            );
        end
        for (i = 0; i < 4; i = i + 1) begin : row1_slice
            (* blackbox *) scratchpad_1024x128_freepdk45 u_row1 (
                .clk0 (clk), .csb0(csb_row1), .web0(web),
                .addr0(row_addr), .din0(din[i*128 +: 128]),
                .dout0(dout_row1[i*128 +: 128])
            );
        end
    endgenerate

    assign dout = addr[10] ? dout_row1 : dout_row0;

endmodule
