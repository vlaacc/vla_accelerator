// lut_compute_unit.v — Per-channel bit-serial shift-accumulate compute unit
// (Option A: 8-stage MSB-first Horner with single-cycle carry-select 16-bit adder)
//
// Computes the FULL signed dot of 16 w_N weights x 16 INT4 activations via a
// fixed 8-stage MSB-first Horner pipeline:
//   dot = ((( -slice[7]) << 1 + slice[6]) << 1 + slice[5]) ... + slice[0]
//   slice[b] = subset_sum[mask[b]]  (mask[b] = bit b of the 4 weights in a group)
//
// Each bit-stage does the 16-bit shift-add (acc = (acc<<1) + slice, with the
// MSB stage negating slice) in ONE cycle using a carry-select adder (critical
// path = 8-bit ripple + MUX ~0.5ns).  This replaces the earlier 2x 8-bit
// pipelined split (16 sub-stages + lo_d/sh_hi handoff), halving both the
// latency and the weight shift-register depth.
//
// For W < 8 the weight is sign-extended to 8b, so the upper sign bits cancel in
// the Horner accumulation, yielding the correct W-bit dot with a FIXED 8-stage
// pipeline (no runtime W).
//
// Latency = ST_PIPE_WEIGHT + ST_PIPE_BITS + ST_FINAL_ACC_STAGES = 1 + 8 + 4 = 13 (ST_RES_DELAY).
`include "sparse_ternary_params.vh"

(* keep_hierarchy *) module lut_compute_unit (
    input  wire                                 clk,
    input  wire                                 rst_n,
    input  wire                                 tile_start,   // unused
    input  wire                                 compute_en,   // 1 拍推 1 tile
    input  wire                                 weight_signed, // 1=有符号(-8..7)/0=无符号(0..15)，门控 Horner stage-0 取反
    input  wire [`ST_WT_PER_TILE*`ST_WT_BITS-1:0] weight,     // 16 权重 x 8b (小端序)
    input  wire [`ST_PIPE_SUB*`ST_PIPE_BITS*`ST_BUILD_TOTAL_REGS-1:0] group_partial_sums,  // W delayed subset-sums
    output wire [`ST_SUBSUM_WIDTH-1:0]          sub_sum_out,
    output wire                                 sub_sum_valid
);
    localparam W   = `ST_WT_BITS;            // 8 (fixed pipeline)
    localparam N_G = `ST_N_GROUPS_PER_TILE;  // 4 groups

    // ── S0: weight register ──
    reg [`ST_WT_PER_TILE*W-1:0] weight_r;
    reg                         v_w;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) weight_r <= 0;
        else        weight_r <= weight;
    end
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) v_w <= 1'b0;
        else        v_w <= compute_en;
    end

    // ── Weight shift register: wd[S] = weight_r delayed S (S=0..W-1) ──
    wire [`ST_WT_PER_TILE*W-1:0] wd [0:W-1];
    assign wd[0] = weight_r;
    genvar d;
    generate
        for (d = 1; d < W; d = d + 1) begin : wdel
            reg [`ST_WT_PER_TILE*W-1:0] r;
            always @(posedge clk or negedge rst_n) begin
                if (!rst_n) r <= 0;
                else        r <= wd[d-1];
            end
            assign wd[d] = r;
        end
    endgenerate

    // ── 8-stage MSB-first Horner, single-cycle 16-bit shift-add per stage ──
    //    acc[g][S] = (acc[g][S-1] << 1) + slice[S]  (S>0);  acc[g][0] = -slice[7]
    //    slice[S] = subset_sum[mask[S]]  (mask[S] = weight bit W-1-S of 4 weights)
    reg signed [15:0] acc [0:N_G-1][0:W-1];
    genvar S, g, i;
    generate
        for (S = 0; S < W; S = S + 1) begin : bitstage
            wire [`ST_WT_PER_TILE-1:0] mask_s;
            for (i = 0; i < `ST_WT_PER_TILE; i = i + 1) begin : mi
                assign mask_s[i] = wd[S][i*W + (W-1-S)];
            end
            for (g = 0; g < N_G; g = g + 1) begin : grp
                wire [3:0]  m      = mask_s[g*4 +: 4];
                wire [6:0]  sl     = group_partial_sums[S*`ST_BUILD_TOTAL_REGS + g*`ST_BUILD_REGS_PER_GROUP + m*7 +: 7];
                wire [15:0] sl_ext = {{9{sl[6]}}, sl};
                wire [15:0] a      = (S == 0) ? (weight_signed ? ~sl_ext : sl_ext) : {acc[g][S-1][14:0], 1'b0};
                wire [15:0] b      = (S == 0) ? 16'd0  : sl_ext;
                wire        cin    = (S == 0) ? (weight_signed ? 1'b1 : 1'b0) : 1'b0;
                wire [15:0] sum;
                wire        cout_unused;
                carry_select_add16 u_add (.a(a), .b(b), .cin(cin), .sum(sum), .cout(cout_unused));
                always @(posedge clk or negedge rst_n) begin
                    if (!rst_n) acc[g][S] <= 16'sd0;
                    else        acc[g][S] <= sum;
                end
            end
        end
    endgenerate

    // ── group dot = acc[g][W-1] (full 16-bit signed dot of the group) ──
    wire signed [15:0] grp_dot [0:N_G-1];
    genvar gd;
    generate
        for (gd = 0; gd < N_G; gd = gd + 1) assign grp_dot[gd] = acc[gd][W-1];
    endgenerate

    // ── valid chain: v_w delayed W (tracks acc[..][W-1]) ──
    reg [W-1:0] v_sr;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) v_sr <= 0;
        else        v_sr <= {v_sr[W-2:0], v_w};
    end
    wire v_t = v_sr[W-1];

    // ── 8-tile group accumulation, PIPELINED final stage ──
    // 4 register stages: (1) balanced-tree pair sums, (2) balanced-tree total,
    // (3) low-half accumulate, (4) high-half accumulate (carry registered).
    // Each 16-bit tree add uses a carry-select adder (~0.5ns); each half of the
    // 24-bit accumulate is a ~12-bit add (<0.5ns).

    // tile_count: free-running 0..7, increments once per tile (on v_t).
    reg [2:0] tile_count;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) tile_count <= 3'd0;
        else if (v_t) tile_count <= tile_count + 1'b1;
    end
    wire first_t = v_t && (tile_count == 3'd0);
    wire last_t  = v_t && (tile_count == `ST_TILES_PER_GROUP - 1);

    // Stage 1: balanced-tree pair sums (16b carry-select)
    wire [15:0] p01_sum, p23_sum;
    carry_select_add16 u_pair01 (.a(grp_dot[0]), .b(grp_dot[1]), .cin(1'b0), .sum(p01_sum), .cout());
    carry_select_add16 u_pair23 (.a(grp_dot[2]), .b(grp_dot[3]), .cin(1'b0), .sum(p23_sum), .cout());
    reg signed [15:0] pair01, pair23;
    reg               v1, f1, l1;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin pair01 <= 16'sd0; pair23 <= 16'sd0; v1 <= 1'b0; f1 <= 1'b0; l1 <= 1'b0; end
        else begin
            pair01 <= p01_sum;
            pair23 <= p23_sum;
            v1 <= v_t; f1 <= first_t; l1 <= last_t;
        end
    end

    // Stage 2: balanced-tree total -> tile_dot (16b carry-select)
    wire [15:0] td_sum;
    carry_select_add16 u_td (.a(pair01), .b(pair23), .cin(1'b0), .sum(td_sum), .cout());
    reg signed [15:0] tile_dot_r;
    reg               v2, f2, l2;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin tile_dot_r <= 16'sd0; v2 <= 1'b0; f2 <= 1'b0; l2 <= 1'b0; end
        else begin
            tile_dot_r <= td_sum;
            v2 <= v1; f2 <= f1; l2 <= l1;
        end
    end

    // Stage 3: low-half accumulate (12b add + carry out), first-tile overwrite
    reg [11:0] sub_lo;
    reg        sub_carry;
    reg [11:0] tile_hi_d;   // upper 12 bits of tile_dot_r, delayed for high stage
    reg        v3, f3, l3;
    wire signed [23:0] tile_se = {{8{tile_dot_r[15]}}, tile_dot_r[15:0]};
    wire [12:0] lo_sum = {1'b0, sub_lo} + {1'b0, tile_se[11:0]};
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin sub_lo <= 12'd0; sub_carry <= 1'b0; tile_hi_d <= 12'd0; v3 <= 1'b0; f3 <= 1'b0; l3 <= 1'b0; end
        else begin
            if (v2) begin
                sub_lo    <= f2 ? tile_se[11:0]  : lo_sum[11:0];
                sub_carry <= f2 ? 1'b0           : lo_sum[12];
            end
            tile_hi_d <= tile_se[23:12];
            v3 <= v2; f3 <= f2; l3 <= l2;
        end
    end

    // Stage 4: high-half accumulate (12b add + carry in); result + valid pulse.
    // sub_lo (stage 3) is written one cycle before sub_hi, so sampling {sub_hi,
    // sub_lo} directly would skew: back-to-back, the next group's tile 0
    // overwrites sub_lo before sub_hi is ready.  sub_lo_d holds sub_lo one extra
    // cycle (captured before the overwrite) so {sub_hi, sub_lo_d} is coherent.
    reg [11:0] sub_hi;
    reg [11:0] sub_lo_d;
    reg        grp_done_r;
    wire [11:0] hi_sum = sub_hi + tile_hi_d + {11'd0, sub_carry};
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin sub_hi <= 12'd0; sub_lo_d <= 12'd0; grp_done_r <= 1'b0; end
        else begin
            if (v3) sub_hi <= f3 ? tile_hi_d : hi_sum;
            sub_lo_d <= sub_lo;
            grp_done_r <= v3 && l3;
        end
    end

    assign sub_sum_out   = {sub_hi, sub_lo_d};
    assign sub_sum_valid = grp_done_r;

endmodule
