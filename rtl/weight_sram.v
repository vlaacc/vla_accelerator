// weight_sram.v — weight storage SRAM for the tile-major weight path
//
// Holds one k_group of weights (DEPTH words × WIDTH bits = 16 KB) and multiplexes a single
// SRAM port between two users:
//   * DMA fill (write): 512b/beat, DEPTH beats, sequential fill_cnt (reset on tlast).
//   * cache refill (read): weight_tile_cache drives read_addr, dout returns 1 cycle later.
//
// The two phases are sequential in the simple (non-ping-pong) version: S_LOAD_WGT fills the
// whole buffer, then S_GROUP reads it via the cache refill.  (Overlap of fill with compute —
// ping-pong or dual-port — is a later optimization.)
//
// Note: this instantiates a behavioral SRAM for simulation; the integration swaps it for
// scratchpad_1024x128_freepdk45 macros (4 width-parallel, blackboxed at synthesis).
module weight_sram #(
    parameter DEPTH = 256,        // words (16 KB at WIDTH=512)
    parameter WIDTH = 512
) (
    input  wire clk,
    input  wire rst_n,
    // DMA fill (write)
    input  wire             fill_en,     // s_axis_wgt_tvalid
    input  wire             fill_last,   // s_axis_wgt_tlast (final beat)
    input  wire [WIDTH-1:0] fill_data,
    // cache refill (read)
    input  wire [$clog2(DEPTH)-1:0] read_addr,
    output reg  [WIDTH-1:0] dout
);
    reg [$clog2(DEPTH)-1:0] fill_cnt;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)              fill_cnt <= 0;
        else if (fill_en)        fill_cnt <= fill_last ? 0 : fill_cnt + 1'b1;
    end

    // single-port mux: fill (write) has priority, otherwise read
    wire [$clog2(DEPTH)-1:0] sram_addr = fill_en ? fill_cnt : read_addr;
    wire                     sram_web  = ~fill_en;
    wire [WIDTH-1:0]         sram_din  = fill_data;

    // behavioral SRAM (1-cycle read, synchronous write) — swap for scratchpad macros in P&R
    reg [WIDTH-1:0] mem [0:DEPTH-1];
    always @(posedge clk) begin
        if (!sram_web) mem[sram_addr] <= sram_din;
        dout <= mem[sram_addr];
    end

endmodule
