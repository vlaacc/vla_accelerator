// PROCESS: ASAP7 (7nm predictive) — SRAM macros are srambank_*.
//
// sram_acc_bank_int8.v — INT8-side accumulator SRAM (ASAP7 1RW), 4096 b port.
//
// Dedicated to the 4 INT8 meshes so they do not contend with the LUT engine's acc
// SRAM.  Sized identically to the LUT side's per-half width (4096 b / 1 Mbit /
// 64 macros), which is not a coincidence: the INT8 MAC rate is half the LUT's, and
// the acc demand `8 B x MAC / G` scales with it 1:1.
//
//   4 meshes x 32 output channels x 32 b = 4096 b per drained row
//   64 x srambank_64x4x64_6t122 (M256, 256 words x 64 b) width-assembled => 4096 b
//   Logical depth 256 rows == macro depth, so no depth cascade.  64 x 16 Kb = 1 Mbit.
//
// 1RW: the INT8 side also splits read (accumulate) and write (store) in time; its
// demand is only 2048 b/cycle (G=128), i.e. 50% of this port, so the split is free.
//
// Library-specific module. The rtl/sram_45nm/ twin has the SAME name and port list
// but is built from accum_512x128_freepdk45.

module sram_acc_bank_int8 (
    input  wire          clk,
    input  wire          csb,           // active low chip select
    input  wire          web,           // active low write enable (1RW: read XOR write)
    input  wire [7:0]    addr,          // 256 rows == macro depth
    input  wire [4095:0] din,
    output wire [4095:0] dout
);

    genvar i;
    generate
        for (i = 0; i < 64; i = i + 1) begin : slice
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
