// int8_act_skew.v — row-major activation stream -> anti-diagonal array feed
//
// rtl/systolic_array_int8.v needs, at cycle x,
//
//     activations_in[i] = A[x - i][i]          (i = reduction index, x = cycle)
//
// but activations are stored row-major, so address m yields the whole 32 x 8 b word
// A[m][0..K-1] at cycle m.  The transform is a pure per-lane delay: lane i wants
// byte i of the word from i cycles ago.
//
//     lane 0 : word_act[0]          (combinational)
//     lane i : byte i, delayed i cycles
//
// so each lane is a shift register of depth i, fed by byte i of the incoming word.
// Total cost is K(K-1)/2 bytes — 496 B (3.9 Kbit) for K=32, ~508 um2 of flops.
// Compare with the alternative of keeping the horizontal activation chain and
// de-skewing the outputs, which needs a ~32x62-word buffer (~0.6 mm2 per 4 meshes).
//
// `valid` is a plain enable: the shift registers hold when it is low.

module int8_act_skew #(
    parameter K = 32
) (
    input  wire             clk,
    input  wire             rst_n,
    input  wire             valid,        // shift enable
    input  wire [K*8-1:0]   word_act,     // A[m][0..K-1] for the current row m
    output wire [K*8-1:0]   act_skew       // feeds activations_in
);
    genvar gi;
    generate
        for (gi = 0; gi < K; gi = gi + 1) begin : lane
            if (gi == 0) begin : head
                // deepest lane needs no delay at all
                assign act_skew[7:0] = word_act[7:0];
            end else begin : pipe
                // Per-lane copies of rst_n/valid.  Flattened, the four skews put
                // 3968 flops each on one reset net and 15872 on one enable net;
                // busting them per lane keeps every copy at a few hundred loads.
                wire rst_lane, val_lane;
                BUFx24_ASAP7_75t_R u_rst_lane (.A(rst_n), .Y(rst_lane));
                BUFx24_ASAP7_75t_R u_val_lane (.A(valid), .Y(val_lane));
                // Shift register holding the last `gi` samples of byte gi.  Shifting
                // left drops the oldest; the TOP byte is therefore the sample from
                // `gi` cycles ago, which is what this lane must emit.
                reg [8*gi-1:0] sr;
                always @(posedge clk or negedge rst_lane) begin
                    if (!rst_lane)  sr <= {(8*gi){1'b0}};
                    else if (val_lane) sr <= (sr << 8) | word_act[gi*8 +: 8];
                end
                assign act_skew[gi*8 +: 8] = sr[8*gi-1 -: 8];
            end
        end
    endgenerate
endmodule
