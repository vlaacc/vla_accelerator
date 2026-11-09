// PROCESS: ASAP7 (7nm predictive) — SRAM macros are srambank_*.
//
// sram_act_bank_int8.v — INT8-side activation SRAM, 1024 b (its own address domain).
//
// (Renamed from sram_act_bank_int8_1024x1024.v; the geometry is in the body, and the
//  name now matches sram_acc_bank_int8 / the sram_45nm twin.)
//
// The LUT engine already consumes the full width of the main act SRAM, so the INT8
// meshes cannot read activation data concurrently from it.  This is a SEPARATE macro
// group with its own address, which is what lets the INT8 meshes run alongside the LUT.
//
// M-split rationale: the 4 meshes cover the SAME K/N block and differ only in the
// m row, so they all need the SAME weight block (32K x 32N = 8 Kbit, broadcast once)
// but 4 DIFFERENT activation rows -> 4 x 256 b = 1024 b/cycle.
// See docs/superpowers/plans/2026-09-22-asap7-sram-1rw-macro-selection.md §2.5 D.
//
// 16 x srambank_256x4x64_6t122 (M1024, 1024 words x 64 b) width-assembled => 1024 b.
// Logical depth 1024 == macro depth, so no depth cascade.
// (Still M1024 here, unlike the LUT act bank: the INT8 act row index is not bounded
//  by the LUT acc-row contract, so it keeps the deeper macro.)
//
// Library-specific module. The rtl/sram_45nm/ twin has the SAME name and port list
// but is built from scratchpad_1024x128_freepdk45.

module sram_act_bank_int8 (
    input  wire          clk,
    input  wire          csb,           // active low chip select
    input  wire          web,           // active low write enable (1RW: read XOR write)
    input  wire [9:0]    addr,          // 1024 rows == macro depth
    input  wire [1023:0] din,
    output wire [1023:0] dout
);

    genvar i;
    generate
        for (i = 0; i < 16; i = i + 1) begin : slice
            (* blackbox, keep *) srambank_256x4x64_6t122 u (
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
