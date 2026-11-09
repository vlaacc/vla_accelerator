// PROCESS: ASAP7 (7nm predictive) — SRAM macros are srambank_*.
// 64 x srambank_256x4x64_6t122 (M1024, 1024 words x 64 b) width-assembled => 4096 b
// Capacity 4096 x 1024 = 4 Mbit = 512 KB (unchanged).
//
// WHY 64x256x4x64 instead of 256x64x4x64:
//   512 KB needs 256 x 16 Kbit (64x4x64) but only 64 x 64 Kbit (256x4x64).
//   Same capacity, 4x fewer macros, 4x narrower data bus (4096 vs 16384),
//   30% less macro area and 73% fewer macro pins.  The pin count is what the
//   asap7 DRT violates (Lef58 V3 cut enclosure at the SRAM pins), so this both
//   shrinks the problem 4x and removes 75% of the macros from the floorplan.
//
// ONE shared address drives all 64 macros.  acc_rmw_unit owns the port while
// computing and the drain scan owns it after `done`.

module sram_acc_bank (
    input  wire           clk,
    input  wire           csb,           // active low chip select
    input  wire           web,           // active low write enable (1RW: read XOR write)
    input  wire [9:0]     addr,          // 1024 words == macro depth
    input  wire [4095:0]  din,
    output wire [4095:0]  dout
);

    genvar i;
    generate
        for (i = 0; i < 64; i = i + 1) begin : slice
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
