// sram_acc_bank_512x512.v — Single bank 512×512b accumulator SRAM
// 4 × scratchpad_1024x128_freepdk45: 4 width-parallel (addr[8:0], addr[9]=0)

module sram_acc_bank_512x512 (
    input  wire        clk,
    input  wire        csb,        // chip select (active low)
    input  wire        web,        // write enable (active low)
    input  wire [8:0]  addr,       // 512 depth
    input  wire [511:0] din,
    output wire [511:0] dout
);

    // Extend to scratchpad 10-bit address (upper bit tied low)
    wire [9:0] sp_addr = {1'b0, addr};

    genvar i;
    generate
        for (i = 0; i < 4; i = i + 1) begin : slice
            (* blackbox, keep *) scratchpad_1024x128_freepdk45 u (
                .clk0(clk), .csb0(csb), .web0(web),
                .addr0(sp_addr), .din0(din[i*128 +: 128]),
                .dout0(dout[i*128 +: 128])
            );
        end
    endgenerate

endmodule
