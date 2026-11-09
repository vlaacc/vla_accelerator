// weight_reg_file.v — Weight register file (double-buffered, ping-pong)
// Holds one k_group (K=128) of weights: 256ch x 8 tile x (16*N_BITS) bits.
// Two buffers: while the LUT engine computes from buf_sel, the weight DMA fills
// the other buffer (fill_buf = ~buf_sel) so the weight port can stream
// continuously (64 GB/s) without stalling the compute.
//
// Fill: DMA 512b beats, beat b -> tile = b[BEAT_W-1:HALF_W], half = b[HALF_W-1:0].
//   fill_last (tlast) marks the final beat: the last word is stored, beat resets,
//   and fill_done asserts.  fill_swap (from ctrl, on a j_pass/k_group boundary)
//   clears fill_done for the new fill target.
// Read: broadcast wt_out = all 256ch weights for the current tile, per cycle,
//   sourced from buf_sel.
module weight_reg_file #(
    parameter N_CH       = 256,
    parameter N_TILES    = 8,
    parameter W_CH_TILE  = 128,     // 16 * ST_WT_BITS (W8=128, W4=64)
    parameter DMA_W      = 512,
    parameter N_BEATS    = 512      // N_CH * N_TILES * W_CH_TILE / DMA_W
) (
    input  wire                      clk,
    input  wire                      rst_n,
    input  wire                      fill_en,      // s_axis_wgt_tvalid
    input  wire                      fill_last,    // s_axis_wgt_tlast (final beat)
    input  wire [DMA_W-1:0]          dma_data,
    input  wire                      buf_sel,      // which buffer the compute reads
    input  wire                      fill_swap,    // pulse: swapped buf_sel (clear fill_done)
    input  wire [$clog2(N_TILES)-1:0] tile_idx,
    output wire [N_CH*W_CH_TILE-1:0] wt_out,
    output wire                      fill_done     // fill_buf holds a complete group
);
    localparam HALVES   = N_CH * W_CH_TILE / DMA_W;   // 256*128/512 = 64
    localparam HALF_W   = $clog2(HALVES);             // 6
    localparam BEAT_W   = $clog2(N_BEATS);            // 9 (W8) / 8 (W4)

    reg [DMA_W-1:0] mem [0:1][0:N_TILES-1][0:HALVES-1];
    reg [BEAT_W-1:0] beat;
    reg             fill_done_r;

    wire fill_buf = ~buf_sel;
    wire [$clog2(N_TILES)-1:0] fill_tile = beat[BEAT_W-1:HALF_W];
    wire [HALF_W-1:0]         fill_half = beat[HALF_W-1:0];

    integer b, t, h;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            beat <= 0;
            fill_done_r <= 0;
            for (b = 0; b < 2; b = b + 1)
                for (t = 0; t < N_TILES; t = t + 1)
                    for (h = 0; h < HALVES; h = h + 1)
                        mem[b][t][h] <= 0;
        end else begin
            if (fill_en) begin
                mem[fill_buf][fill_tile][fill_half] <= dma_data;
                if (fill_last) begin
                    beat <= 0;          // final beat stored, wrap for next group
                    fill_done_r <= 1;
                end else begin
                    beat <= beat + 1'b1;
                end
            end
            if (fill_swap) fill_done_r <= 0;
        end
    end

    genvar hh;
    generate
        for (hh = 0; hh < HALVES; hh = hh + 1) begin : rd
            assign wt_out[hh*DMA_W +: DMA_W] = mem[buf_sel][tile_idx][hh];
        end
    endgenerate

    assign fill_done = fill_done_r;

endmodule
