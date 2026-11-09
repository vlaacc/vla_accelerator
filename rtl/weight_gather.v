// weight_gather.v — Per-channel weight gather (act_compress feature), runtime W4/ternary.
// For each channel, gathers K weights selected by idx_0..3 (the K non-zero
// activation indices), zero-fills the remaining slots, and passes through
// unchanged when compress_en is deasserted (dense fallback for N>K).
//
// The weight word per channel is 64 bit in both modes (ST_WT_WORD): 16x4-bit
// (W4) or 32x2-bit (ternary).  wt_mode picks the gather granularity (4b vs 2b)
// and the front-placement stride.  The upper 16 activation slots are zero in W4
// mode (zero-padded upstream), so idx_j never exceeds 15 there and idx_j*4 stays
// within the 64-bit word.
`include "sparse_ternary_params.vh"

module weight_gather #(
    parameter N_CH = 256,
    parameter K    = 4
) (
    input  wire                                wt_mode,   // 1=ternary(2b) 0=W4(4b)
    input  wire [N_CH*`ST_WT_WORD-1:0]         weight_in,
    input  wire [4:0]                          idx_0, idx_1, idx_2, idx_3,
    input  wire [5:0]                          nz_cnt,
    input  wire                                compress_en,
    output wire [N_CH*`ST_WT_WORD-1:0]         weight_out
);
    localparam WCH = `ST_WT_WORD;   // 64

    genvar c;
    generate
        for (c = 0; c < N_CH; c = c + 1) begin : ch
            // gather at both granularities (4b W4 / 2b ternary), mux by wt_mode
            wire [3:0] w4_0 = (nz_cnt > 6'd0) ? weight_in[c*WCH + idx_0*4 +: 4] : 4'd0;
            wire [3:0] w4_1 = (nz_cnt > 6'd1) ? weight_in[c*WCH + idx_1*4 +: 4] : 4'd0;
            wire [3:0] w4_2 = (nz_cnt > 6'd2) ? weight_in[c*WCH + idx_2*4 +: 4] : 4'd0;
            wire [3:0] w4_3 = (nz_cnt > 6'd3) ? weight_in[c*WCH + idx_3*4 +: 4] : 4'd0;
            wire [1:0] w2_0 = (nz_cnt > 6'd0) ? weight_in[c*WCH + idx_0*2 +: 2] : 2'd0;
            wire [1:0] w2_1 = (nz_cnt > 6'd1) ? weight_in[c*WCH + idx_1*2 +: 2] : 2'd0;
            wire [1:0] w2_2 = (nz_cnt > 6'd2) ? weight_in[c*WCH + idx_2*2 +: 2] : 2'd0;
            wire [1:0] w2_3 = (nz_cnt > 6'd3) ? weight_in[c*WCH + idx_3*2 +: 2] : 2'd0;
            wire [WCH-1:0] compact = wt_mode
                ? {56'd0, w2_3, w2_2, w2_1, w2_0}
                : {48'd0, w4_3, w4_2, w4_1, w4_0};
            assign weight_out[c*WCH +: WCH] = compress_en ? compact : weight_in[c*WCH +: WCH];
        end
    endgenerate
endmodule
