// sram_act_bank_2048x512.v — Single bank 2048x512b activation SRAM
// 8 x scratchpad_1024x128_freepdk45: 4 width-parallel x 2 depth-cascade
// v22 fix: XOR row outputs to force ABC to keep both rows

module sram_act_bank_2048x512 (
    input  wire        clk, csb, web,
    input  wire [10:0] addr,
    input  wire [511:0] din,
    output wire [511:0] dout
);

    wire       csb_row0 = csb |  addr[10];
    wire       csb_row1 = csb | ~addr[10];
    wire [9:0] row_addr = addr[9:0];
    wire [511:0] dout_row0, dout_row1;

    genvar i;
    generate
        for (i = 0; i < 4; i = i + 1) begin : row0_slice
            (* blackbox, keep *) scratchpad_1024x128_freepdk45 u_row0 (
                .clk0(clk), .csb0(csb_row0), .web0(web),
                .addr0(row_addr), .din0(din[i*128 +: 128]),
                .dout0(dout_row0[i*128 +: 128])
            );
        end
        for (i = 0; i < 4; i = i + 1) begin : row1_slice
            (* blackbox, keep *) scratchpad_1024x128_freepdk45 u_row1 (
                .clk0(clk), .csb0(csb_row1), .web0(web),
                .addr0(row_addr), .din0(din[i*128 +: 128]),
                .dout0(dout_row1[i*128 +: 128])
            );
        end
    endgenerate

    // XOR trick: force ABC to keep both rows
    wire [511:0] row_xor;
    assign row_xor = dout_row0 ^ dout_row1;
    wire dummy;
    assign dummy = |row_xor;

    assign dout = addr[10] ? dout_row1 : dout_row0;

endmodule
