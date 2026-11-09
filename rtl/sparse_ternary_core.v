// sparse_ternary_core.v — Single core: 1 BUILD + 16 x ternary_engine (16ch each)
// tile-major dataflow, runtime W4/ternary (wt_mode).  Each ternary_engine streams a
// per-tile redundant dot product (tile_s, tile_c); the core concatenates the 16
// engines' dots into N_CH lanes and hands them to a register-block accumulator
// (rtl/acc_block_reg.v) that owns the K=128 group accumulation.
//
// wt_mode=1 → ternary: K=32 tile, 4 tiles/group.
// wt_mode=0 → W4:      K=16 tile, 8 tiles/group.
// Both modes use act_compress (activation sparsity); the compressor is sized for
// the max K=32 tile, and W4 zero-pads the upper 16 activations so it naturally
// sees only the low 16 non-zeros.
`include "sparse_ternary_params.vh"

(* keep_hierarchy *) module sparse_ternary_core #(
    parameter N_CH = `ST_N_CH_PER_CORE,
    parameter N_BATCH = 16,                      // register-block depth
    localparam BATCH_W = $clog2(N_BATCH),        // 4
    localparam TILE_W  = 16                      // per-tile redundant dot width
) (
    input  wire                                        clk,
    input  wire                                        rst_n,
    input  wire                                        wt_mode,       // 1=ternary 0=W4
    input  wire [`ST_GROUP_K * `ST_ACT_WIDTH - 1:0]    act_row,
    input  wire [N_CH * `ST_WT_WORD - 1:0]             weight_all,
    input  wire                                        build_en,
    input  wire                                        compute_en,
    input  wire                                        weight_signed,
    input  wire [2:0]                                  act_tile_idx,
    input  wire                                        group_start,
    // Live batch count of the current sub-pass (see addr_gen / acc_block_reg): the last
    // sub-pass of a block can be short, and the batch wrap must follow it.
    input  wire [4:0]                                  b_sub,
    output wire [N_CH * `ST_SUBSUM_WIDTH - 1:0]        sub_sum,
    output wire [BATCH_W-1:0]                          sub_sum_batch,
    output wire                                        sub_sum_valid
);
    localparam ENGINES = `ST_T_COPIES_PER_CORE;
    localparam CH_PER_ENGINE = N_CH / ENGINES;
    localparam ACT_TILE_MAX_BITS = `ST_TILE_K_MAX * `ST_ACT_WIDTH;   // 128 (ternary)
    localparam ACT_TILE_W4_BITS  = `ST_TILE_K_W4  * `ST_ACT_WIDTH;   // 64  (W4)

    // ── tile activation slice: K=32 (ternary, 128b) or K=16 (W4, 64b), zero-pad
    //    the upper 64b in W4 mode so the 8-group build_unit sees 32 activations ──
    wire [ACT_TILE_MAX_BITS-1:0] act_tile = wt_mode
        ? act_row[act_tile_idx * ACT_TILE_MAX_BITS +: ACT_TILE_MAX_BITS]
        : {{(ACT_TILE_MAX_BITS-ACT_TILE_W4_BITS){1'b0}},
           act_row[act_tile_idx * ACT_TILE_W4_BITS +: ACT_TILE_W4_BITS]};

    // ── act_compress: activation sparsity (both modes; compressor is K=32) ──
    wire [5:0]                  act_nz_c;
    wire [4:0]                  act_idx0_c, act_idx1_c, act_idx2_c, act_idx3_c;
    wire [`ST_ACT_WIDTH-1:0]    act_c0_c, act_c1_c, act_c2_c, act_c3_c;
    wire                        compress_en_c;

    act_compressor u_ac (
        .activations(act_tile),
        .valid(), .nz_cnt(act_nz_c),
        .idx_0(act_idx0_c), .idx_1(act_idx1_c), .idx_2(act_idx2_c), .idx_3(act_idx3_c),
        .act_c0(act_c0_c), .act_c1(act_c1_c), .act_c2(act_c2_c), .act_c3(act_c3_c),
        .compress_en(compress_en_c)
    );

    // ── stage-1 registers: pipeline the compressor off the critical path ──
    reg                        compress_en_r;
    reg [5:0]                  act_nz_r;
    reg [4:0]                  act_idx0_r, act_idx1_r, act_idx2_r, act_idx3_r;
    reg [`ST_ACT_WIDTH-1:0]    act_c0_r, act_c1_r, act_c2_r, act_c3_r;
    reg [ACT_TILE_MAX_BITS-1:0] act_tile_r;
    reg                        build_en_r;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            compress_en_r <= 1'b0;
            act_nz_r     <= 6'd0;
            act_idx0_r <= 5'd0; act_idx1_r <= 5'd0; act_idx2_r <= 5'd0; act_idx3_r <= 5'd0;
            act_c0_r   <= 4'd0; act_c1_r   <= 4'd0; act_c2_r   <= 4'd0; act_c3_r   <= 4'd0;
            act_tile_r  <= 0;
            build_en_r <= 1'b0;
        end else begin
            compress_en_r <= compress_en_c;
            act_nz_r     <= act_nz_c;
            act_idx0_r <= act_idx0_c; act_idx1_r <= act_idx1_c; act_idx2_r <= act_idx2_c; act_idx3_r <= act_idx3_c;
            act_c0_r   <= act_c0_c;   act_c1_r   <= act_c1_c;   act_c2_r   <= act_c2_c;   act_c3_r   <= act_c3_c;
            act_tile_r  <= act_tile;
            build_en_r <= build_en;
        end
    end

    // ── stage 2 (combinational): MUX build input + weight gather ──
    wire [ACT_TILE_MAX_BITS-1:0] act_compact_r =
        {{(ACT_TILE_MAX_BITS - 4*`ST_ACT_WIDTH){1'b0}}, act_c3_r, act_c2_r, act_c1_r, act_c0_r};
    wire [ACT_TILE_MAX_BITS-1:0] build_act = compress_en_r ? act_compact_r : act_tile_r;

    wire [N_CH * `ST_WT_WORD - 1:0] weight_gathered;
    weight_gather #(
        .N_CH(N_CH), .K(`ST_ACT_COMPRESS_K)
    ) u_wg (
        .wt_mode(wt_mode),
        .weight_in(weight_all),
        .idx_0(act_idx0_r), .idx_1(act_idx1_r), .idx_2(act_idx2_r), .idx_3(act_idx3_r),
        .nz_cnt(act_nz_r), .compress_en(compress_en_r),
        .weight_out(weight_gathered)
    );
    wire build_en_use   = build_en_r;
    wire compute_en_use = compute_en;

    wire [`ST_BUILD_TOTAL_REGS-1:0] build_out;
    build_unit u_build (
        .clk(clk), .rst_n(rst_n),
        .build_en(build_en_use),
        .activations(build_act),
        .group_partial_sums(build_out)
    );

    // ── subset sums fed to the engines (shifted for both W4 and ternary Horner) ──
    // subset_sum shift register: (ST_PIPE_BITS-1) deep, prepend build_out ──
    reg [(`ST_PIPE_SUB*`ST_PIPE_BITS-1)*`ST_BUILD_TOTAL_REGS-1:0] sd;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) sd <= 0;
        else        sd <= {sd[(`ST_PIPE_SUB*`ST_PIPE_BITS-2)*`ST_BUILD_TOTAL_REGS-1:0], build_out};
    end
    wire [`ST_PIPE_SUB*`ST_PIPE_BITS*`ST_BUILD_TOTAL_REGS-1:0] gps = {sd, build_out};

    // ── per-engine per-tile dots, concatenated into N_CH lanes ──
    wire [N_CH * TILE_W - 1:0] tile_s_all;
    wire [N_CH * TILE_W - 1:0] tile_c_all;
    wire                        core_tile_valid;

    genvar eng;
    generate
        for (eng = 0; eng < ENGINES; eng = eng + 1) begin : engine_inst
            localparam CH_OFF = eng * CH_PER_ENGINE;
            wire [CH_PER_ENGINE * `ST_WT_WORD - 1:0] eng_weight;
            assign eng_weight = weight_gathered[CH_OFF * `ST_WT_WORD +: CH_PER_ENGINE * `ST_WT_WORD];

            wire [CH_PER_ENGINE * TILE_W - 1:0] eng_tile_s;
            wire [CH_PER_ENGINE * TILE_W - 1:0] eng_tile_c;
            wire eng_tile_valid;

            assign tile_s_all[CH_OFF*TILE_W +: CH_PER_ENGINE*TILE_W] = eng_tile_s;
            assign tile_c_all[CH_OFF*TILE_W +: CH_PER_ENGINE*TILE_W] = eng_tile_c;

            ternary_engine #(
                .N_CH_PER_ENGINE(CH_PER_ENGINE), .CH_OFFSET(CH_OFF)
            ) u_engine (
                .clk(clk), .rst_n(rst_n), .wt_mode(wt_mode),
                .weight_segment(eng_weight),
                .build_out(gps),
                .compute_en(compute_en_use),
                .weight_signed(weight_signed),
                .tile_s(eng_tile_s),
                .tile_c(eng_tile_c),
                .tile_valid(eng_tile_valid)
            );
        end
    endgenerate

    // All engines share an identical pipeline latency, so any engine's per-tile
    // valid tracks the whole core.  Use engine[0].
    assign core_tile_valid = engine_inst[0].eng_tile_valid;

    // ── register-block accumulator: owns the K=128 group accumulation ──
    acc_block_reg #(
        .N_CH(N_CH), .N_BATCH(N_BATCH), .TILE_W(TILE_W),
        .SUM_W(`ST_SUBSUM_WIDTH), .TILES_PER_GROUP(`ST_TILES_PER_GROUP)
    ) u_acc (
        .clk(clk), .rst_n(rst_n),
        .tile_valid(core_tile_valid),
        .group_start(group_start),
        .wt_mode(wt_mode),
        .b_sub(b_sub),
        .tile_s(tile_s_all),
        .tile_c(tile_c_all),
        .sub_sum(sub_sum),
        .sub_sum_batch(sub_sum_batch),
        .sub_sum_valid(sub_sum_valid)
    );
endmodule
