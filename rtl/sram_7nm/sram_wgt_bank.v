// PROCESS: ASAP7 (7nm predictive) - SRAM macros are srambank_*.
// 32 x srambank_64x4x64_6t122 (256 words x 64 b) width-assembled => 2048 b, 128 words.
// Only the low 128 words are addressed (ADDRESS[7] tied 0).
module sram_wgt_bank (
    input  wire           clk,
    input  wire           csb,           // active low chip select
    input  wire           web,           // active low write enable (1RW: read XOR write)
    input  wire [6:0]     addr,          // 128 words
    input  wire [2047:0]  din,
    output wire [2047:0]  dout
);

    genvar i;
    generate
        for (i = 0; i < 32; i = i + 1) begin : slice
            (* blackbox, keep *) srambank_64x4x64_6t122 u (
                .banksel (~csb),
                .write   (~csb & ~web),
                .read    (~csb &  web),
                .ADDRESS ({1'b0, addr}),
                .wd      (din [i*64 +: 64]),
                .dataout (dout[i*64 +: 64]),
                .clk     (clk)
            );
        end
    endgenerate

endmodule
