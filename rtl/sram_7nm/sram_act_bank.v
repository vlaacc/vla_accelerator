// PROCESS: ASAP7 (7nm predictive) — SRAM macros are srambank_*.
//
// sram_act_bank.v — LUT-engine activation bank, 512 b per access.
//
// 8 x srambank_64x4x64_6t122 (M256, 256 words x 64 b) width-assembled => 512 b.
//
// M256, NOT M1024: the acc row contract caps dim_I at 256, so the act address never
// exceeds 128 -- a 1024-deep macro would be 8x deeper than anything addressable.
// The macro's free top address bit is used by the top-level region split (the core
// reads one half of the depth while the DMA fills the other), so `addr` is the full
// 8-bit macro address and passes straight through.  No depth cascade.
//
// The LUT activation store is TWO of these banks sharing one address (see
// act_port_mux.v); bank select is `web` only, since every access asserts csb.
//
// Library-specific module. The rtl/sram_45nm/ twin has the SAME name and port list
// but is built from scratchpad_1024x128_freepdk45.

module sram_act_bank (
    input  wire         clk,
    input  wire         csb,            // active low chip select
    input  wire         web,            // active low write enable (1RW: read XOR write)
    input  wire [7:0]   addr,           // 256 rows; top bit = region (read/write half)
    input  wire [511:0] din,
    output wire [511:0] dout
);

    genvar i;
    generate
        for (i = 0; i < 8; i = i + 1) begin : slice
            (* blackbox, keep *) srambank_64x4x64_6t122 u (
                .clk     (clk),
                .banksel (~csb),
                .write   (~csb & ~web),
                .read    (~csb &  web),
                .ADDRESS (addr),
                .wd      (din [i*64 +: 64]),
                .dataout (dout[i*64 +: 64])
            );
        end
    endgenerate

endmodule
