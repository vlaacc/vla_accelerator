// lut_compute_unit_mixed.v — unified 4-stage bit-serial Horner LUT, RUNTIME W4/ternary.
//
// wt_mode=1 → ternary: 32 weights x 2-bit ({-1,0,+1} two's-complement packed
//   2-per-4-bit-slot), 8 groups, the SAME 4-stage Horner reused as TWO 2-stage
//   decoupled chains:
//     stages 0-1 -> "hi" weights (groups 0-3, K=0..15)  -> dot_hi
//     stages 2-3 -> "lo" weights (groups 4-7, K=16..31) -> dot_lo
//   stage 2 RE-INITS (signed init) so dot_hi/dot_lo are separate, then summed.
// wt_mode=0 → W4: 16 weights x 4-bit, 4 groups, the standard 4-stage MSB-first
//   signed Horner -> one K=16 dot.
//
// The Horner datapath is shared (4 groups x 4 stages); only the weight-bit
// select, the subset-sum source group, and the stage-2 operation (init vs CSA)
// differ at runtime.  The final reduction tree is DUPLICATED (ternary 16->2 vs
// W4 8->2) and the output is muxed — the two trees have different input counts
// and arrival times, so sharing them is not worth an input mux.
//
// W4's ta-reg was removed: without it the W4 tree is 4 CSA levels (~0.4ns, still
// under the ~0.62ns Horner budget), and both modes now emit tile_s/tile_c at the
// SAME latency (S3 + 1 cycle), which the shared output register requires.
`include "sparse_ternary_params.vh"

module lut_compute_unit_mixed #(
    parameter W       = 4,             // Horner stages (always 4)
    parameter TILE_W  = 16
) (
    input  wire                                       clk,
    input  wire                                       rst_n,
    input  wire                                       wt_mode,       // 1=ternary(K=32,2b) 0=W4(K=16,4b)
    input  wire                                       compute_en,   // 1 pulse per tile
    input  wire                                       weight_signed,
    input  wire [`ST_WT_WORD-1:0]                     weight,       // 64b (16x4b / 32x2b)
    input  wire [`ST_PIPE_SUB*`ST_PIPE_BITS*`ST_BUILD_TOTAL_REGS-1:0] group_partial_sums, // shifted gps
    output reg  signed [TILE_W-1:0]                   tile_s,
    output reg  signed [TILE_W-1:0]                   tile_c,
    output reg                                        tile_valid
);
    localparam H_G = 4;               // Horner groups per stage (always 4)

    // ── S0: weight register ──
    reg [`ST_WT_WORD-1:0] weight_r;
    reg                   v_w;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) weight_r <= 0;
        else        weight_r <= weight;
    end
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) v_w <= 1'b0;
        else        v_w <= compute_en;
    end

    // ── weight shift register: wd[S] = weight_r delayed S ──
    wire [`ST_WT_WORD-1:0] wd [0:W-1];
    assign wd[0] = weight_r;
    genvar d;
    generate
        for (d = 1; d < W; d = d + 1) begin : wdel
            reg [`ST_WT_WORD-1:0] r;
            always @(posedge clk or negedge rst_n) begin
                if (!rst_n) r <= 0;
                else        r <= wd[d-1];
            end
            assign wd[d] = r;
        end
    endgenerate

    // ── 4-stage MSB-first Horner (carry-save), runtime W4/ternary ──
    reg signed [TILE_W-1:0] hsum [0:H_G-1][0:W-1];
    reg signed [TILE_W-1:0] hcar [0:H_G-1][0:W-1];
    genvar S, g, i;
    generate
        for (S = 0; S < W; S = S + 1) begin : bitstage
            // 32-bit mask: ternary uses all 32 (2b each); W4 uses low 16 (4b each),
            // high 16 = 0.  Weight-bit select: ternary reads MSB (S even) / LSB (S odd)
            // of each 2-bit weight; W4 reads bit (3-S) MSB-first of each 4-bit weight.
            wire [31:0] mask_s;
            for (i = 0; i < 32; i = i + 1) begin : mi
                if (i < 16) begin : w4_ok
                    // both encodings live in the 64 b word: 4 b x 16 (W4) / 2 b x 32 (ternary)
                    assign mask_s[i] = wt_mode ? wd[S][i*2 + ((S % 2 == 0) ? 1 : 0)]
                                               : wd[S][i*4 + (3 - S)];
                end else begin : ternary_only
                    // lanes 16..31 exist only in ternary mode.  Splitting them into their
                    // own generate branch keeps the W4 select (i*4+3 -> bit 127) from ever
                    // being elaborated against the 64 b word, which Icarus warns about.
                    assign mask_s[i] = wt_mode ? wd[S][i*2 + ((S % 2 == 0) ? 1 : 0)] : 1'b0;
                end
            end
            for (g = 0; g < H_G; g = g + 1) begin : grp
                // Subset-sum read kept at 16:1 (constant group base) to avoid a 128:1
                // mux from a runtime group offset.  Compute both the hi group (g) and
                // lo group (g+4) reads, then 2:1-mux the 7-bit result.  At S>=2 the hi
                // group is W4-only and the lo group is ternary-only, so gate each group's
                // data with the mode to keep the unused 16:1 mux dark (data path).
                wire [3:0]  m_hi  = mask_s[g*4 +: 4];
                wire [6:0]  sl;
                if (S >= 2) begin : lo_path
                    wire [3:0]  m_lo  = mask_s[(g+4)*4 +: 4];
                    wire [`ST_BUILD_REGS_PER_GROUP-1:0] gps_hi =
                        group_partial_sums[S*`ST_BUILD_TOTAL_REGS + g*`ST_BUILD_REGS_PER_GROUP +: `ST_BUILD_REGS_PER_GROUP]
                        & {`ST_BUILD_REGS_PER_GROUP{~wt_mode}};
                    wire [`ST_BUILD_REGS_PER_GROUP-1:0] gps_lo =
                        group_partial_sums[S*`ST_BUILD_TOTAL_REGS + (g+4)*`ST_BUILD_REGS_PER_GROUP +: `ST_BUILD_REGS_PER_GROUP]
                        & {`ST_BUILD_REGS_PER_GROUP{wt_mode}};
                    wire [6:0]  sl_hi = gps_hi[m_hi*7 +: 7];
                    wire [6:0]  sl_lo = gps_lo[m_lo*7 +: 7];
                    assign sl = wt_mode ? sl_lo : sl_hi;
                end else begin : hi_only
                    // S<2: hi group used in both modes, no gating.
                    wire [6:0]  sl_hi = group_partial_sums[S*`ST_BUILD_TOTAL_REGS + g*`ST_BUILD_REGS_PER_GROUP + m_hi*7 +: 7];
                    assign sl = sl_hi;
                end
                wire [15:0] sl_ext = {{9{sl[6]}}, sl};
                // init operation (signed negate + select) — carry-save form of -sl_ext / sl_ext
                wire [15:0] init_s = weight_signed ? ~sl_ext : sl_ext;
                wire [15:0] init_c = weight_signed ? 16'sd1  : 16'sd0;
                if (S == 0) begin : init_stage
                    // S0 always init (both modes)
                    always @(posedge clk or negedge rst_n) begin
                        if (!rst_n) begin hsum[g][S] <= 16'sd0; hcar[g][S] <= 16'sd0; end
                        else begin hsum[g][S] <= init_s; hcar[g][S] <= init_c; end
                    end
                end else if (S == 2) begin : dual_stage
                    // S2 dual: ternary re-inits (lo chain), W4 continues CSA
                    wire [15:0] ss   = {hsum[g][S-1][14:0], 1'b0};
                    wire [15:0] cc   = {hcar[g][S-1][14:0], 1'b0};
                    wire [15:0] nsum, ncar;
                    csa3 #(.W(16)) u_csa (.a(ss), .b(cc), .c(sl_ext), .s(nsum), .co(ncar));
                    always @(posedge clk or negedge rst_n) begin
                        if (!rst_n) begin hsum[g][S] <= 16'sd0; hcar[g][S] <= 16'sd0; end
                        else if (wt_mode) begin hsum[g][S] <= init_s; hcar[g][S] <= init_c; end
                        else             begin hsum[g][S] <= nsum;   hcar[g][S] <= ncar;   end
                    end
                end else begin : csa_stage  // S==1, S==3
                    wire [15:0] ss   = {hsum[g][S-1][14:0], 1'b0};
                    wire [15:0] cc   = {hcar[g][S-1][14:0], 1'b0};
                    wire [15:0] nsum, ncar;
                    csa3 #(.W(16)) u_csa (.a(ss), .b(cc), .c(sl_ext), .s(nsum), .co(ncar));
                    always @(posedge clk or negedge rst_n) begin
                        if (!rst_n) begin hsum[g][S] <= 16'sd0; hcar[g][S] <= 16'sd0; end
                        else begin hsum[g][S] <= nsum; hcar[g][S] <= ncar; end
                    end
                end
            end
        end
    endgenerate

    // ── valid chain: v_w delayed W ──
    reg [W-1:0] v_sr;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) v_sr <= 0;
        else        v_sr <= {v_sr[W-2:0], v_w};
    end
    wire v_t = v_sr[W-1];

    // ── final reduction: ONE shared stage-3 8→2 tree + hi 8→2 + 4→2 combine ──
    // Shared 8→2 tree reduces hsum[g][3]/hcar[g][3] ALWAYS (no gating):
    //   W4:     hsum[g][3] = W4 K=16 dot          -> (lo_s,lo_c) = W4 result
    //   ternary: hsum[g][3] = dot_lo (groups 4-7)  -> (lo_s,lo_c) = dot_lo
    // The old W4 tree (7 CSA) and ternary lo tree (5 CSA) read the SAME 8 registers
    // gated oppositely; this single 8→2 tree replaces both.
    wire [15:0] l1_0s, l1_0c, l1_1s, l1_1c, l1_2s, l1_2c;
    csa3 #(.W(16)) u_t0 (.a(hsum[0][3]),.b(hcar[0][3]),.c(hsum[1][3]),.s(l1_0s),.co(l1_0c));
    csa3 #(.W(16)) u_t1 (.a(hcar[1][3]),.b(hsum[2][3]),.c(hcar[2][3]),.s(l1_1s),.co(l1_1c));
    csa3 #(.W(16)) u_t2 (.a(hsum[3][3]),.b(hcar[3][3]),.c(16'd0),.s(l1_2s),.co(l1_2c));
    wire [15:0] l2_0s, l2_0c, l2_1s, l2_1c;
    csa3 #(.W(16)) u_t3 (.a(l1_0s),.b(l1_0c),.c(l1_1s),.s(l2_0s),.co(l2_0c));
    csa3 #(.W(16)) u_t4 (.a(l1_1c),.b(l1_2s),.c(l1_2c),.s(l2_1s),.co(l2_1c));
    wire [15:0] tb0s, tb0c, lo_s, lo_c;
    csa3 #(.W(16)) u_t5 (.a(l2_0s),.b(l2_0c),.c(l2_1s),.s(tb0s),.co(tb0c));
    csa3 #(.W(16)) u_t6 (.a(tb0s),.b(tb0c),.c(l2_1c),.s(lo_s),.co(lo_c));

    // Ternary hi 8→2 tree: reduces hsum[g][1]/hcar[g][1] (dot_hi), gated 0 in W4 mode
    // so the lattice stays dark when W4 is selected.
    wire [15:0] hi0 = wt_mode ? hsum[0][1] : 16'd0;
    wire [15:0] hi1 = wt_mode ? hcar[0][1] : 16'd0;
    wire [15:0] hi2 = wt_mode ? hsum[1][1] : 16'd0;
    wire [15:0] hi3 = wt_mode ? hcar[1][1] : 16'd0;
    wire [15:0] hi4 = wt_mode ? hsum[2][1] : 16'd0;
    wire [15:0] hi5 = wt_mode ? hcar[2][1] : 16'd0;
    wire [15:0] hi6 = wt_mode ? hsum[3][1] : 16'd0;
    wire [15:0] hi7 = wt_mode ? hcar[3][1] : 16'd0;
    wire [15:0] hl1_0s, hl1_0c, hl1_1s, hl1_1c, hl1_2s, hl1_2c;
    csa3 #(.W(16)) u_h0 (.a(hi0),.b(hi1),.c(hi2),.s(hl1_0s),.co(hl1_0c));
    csa3 #(.W(16)) u_h1 (.a(hi3),.b(hi4),.c(hi5),.s(hl1_1s),.co(hl1_1c));
    csa3 #(.W(16)) u_h2 (.a(hi6),.b(hi7),.c(16'd0),.s(hl1_2s),.co(hl1_2c));
    wire [15:0] hl2_0s, hl2_0c, hl2_1s, hl2_1c;
    csa3 #(.W(16)) u_h3 (.a(hl1_0s),.b(hl1_0c),.c(hl1_1s),.s(hl2_0s),.co(hl2_0c));
    csa3 #(.W(16)) u_h4 (.a(hl1_1c),.b(hl1_2s),.c(hl1_2c),.s(hl2_1s),.co(hl2_1c));
    wire [15:0] hb0s, hb0c, hi_s, hi_c;
    csa3 #(.W(16)) u_h5 (.a(hl2_0s),.b(hl2_0c),.c(hl2_1s),.s(hb0s),.co(hb0c));
    csa3 #(.W(16)) u_h6 (.a(hb0s),.b(hb0c),.c(hl2_1c),.s(hi_s),.co(hi_c));

    // Delay hi_s/hi_c 2 cycles so dot_hi aligns with dot_lo (stage3 arrives 2 cycles
    // after stage1).  Delaying 2 numbers (vs the old 4 hq numbers) halves the reg cost.
    reg signed [15:0] hi_s_d1, hi_c_d1, hi_s_d2, hi_c_d2;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin hi_s_d1<=16'sd0; hi_c_d1<=16'sd0; hi_s_d2<=16'sd0; hi_c_d2<=16'sd0; end
        else begin
            hi_s_d1 <= hi_s;    hi_c_d1 <= hi_c;
            hi_s_d2 <= hi_s_d1; hi_c_d2 <= hi_c_d1;
        end
    end

    // Final 4→2: (hi_d2, lo) -> (sum, carry).  W4: hi=0 -> redundant(lo_s+lo_c) = W4 dot.
    // No output MUX — the same register pair is the result in both modes.
    wire [15:0] f0s, f0c, tile_s_pre, tile_c_pre;
    csa3 #(.W(16)) u_f0 (.a(hi_s_d2),.b(hi_c_d2),.c(lo_s),.s(f0s),.co(f0c));
    csa3 #(.W(16)) u_f1 (.a(f0s),.b(f0c),.c(lo_c),.s(tile_s_pre),.co(tile_c_pre));

    // ── output register ──
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin tile_s <= 16'sd0; tile_c <= 16'sd0; tile_valid <= 1'b0; end
        else begin
            tile_s <= tile_s_pre; tile_c <= tile_c_pre;
            tile_valid <= v_t;
        end
    end
endmodule
