// sram_acc_bank_1024x512.v — Accumulator bank 1024×512b SRAM
// 4 × accum_512x128_freepdk45: 4 width-parallel

module sram_acc_bank_1024x512 (
    input  wire        clk,
    input  wire        csb,
    input  wire        web,
    input  wire [9:0]  addr,
    input  wire [511:0] din,
    output wire [511:0] dout
);

    genvar i;
    generate
        for (i = 0; i < 4; i = i + 1) begin : slice
            (* blackbox *) accum_512x128_freepdk45 u (
                .clk0(clk), .csb0(csb), .web0(web),
                .addr0(addr), .din0(din[i*128 +: 128]),
                .dout0(dout[i*128 +: 128])
            );
        end
    endgenerate

endmodule
