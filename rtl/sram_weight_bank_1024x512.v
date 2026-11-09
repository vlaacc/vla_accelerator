// sram_weight_bank_1024x512.v — Single bank 1024×512b weight SRAM
// 4 × scratchpad_1024x128_freepdk45: 4 width-parallel (no depth cascade)
// Blackboxed during Yosys synthesis, instantiated as macro by OpenROAD.

module sram_weight_bank_1024x512 (
    input  wire        clk,
    input  wire        csb,        // chip select (active low)
    input  wire        web,        // write enable (active low)
    input  wire [9:0]  addr,
    input  wire [511:0] din,
    output wire [511:0] dout
);

    genvar i;
    generate
        for (i = 0; i < 4; i = i + 1) begin : slice
            (* blackbox, keep *) scratchpad_1024x128_freepdk45 u (
                .clk0(clk), .csb0(csb), .web0(web),
                .addr0(addr), .din0(din[i*128 +: 128]),
                .dout0(dout[i*128 +: 128])
            );
        end
    endgenerate

endmodule
