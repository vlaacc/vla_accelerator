// addr_gen.v — Tile-major tiling address generator (mirrors sparse_ternary_ctrl)
//
// Mirrors the tile-major loop:
//   for k_group: for j_pass: for sub_pass: for tile(0..7): for batch(0..N_BATCH-1)
//
// The batch is the INNER loop (advances every build cycle); wt_tile_idx and
// act_tile_idx advance every N_BATCH cycles (when batch wraps) — they are the
// OUTER tile.  The weight tile lags the activation tile by one presentation
// (build/compute stagger), matching the tb_core_tile_major presentation order
// (weight[n-1] presented alongside act[n]).
//
// sub_pass / j_pass / k_group advance on ctrl_group_done (the controller's group
// boundary, once per sub-pass).
module addr_gen #(
    parameter R_CORES         = 2,
    parameter N_CH_PER_CORE   = 256,
    parameter T_PER           = 16,
    parameter GROUP_K         = 128,
    parameter TILES_PER_GROUP = 8
) (
    input  wire        clk,
    input  wire        rst_n,
    input  wire [15:0] dim_I, dim_J, dim_K,
    input  wire        ctrl_tile_start,
    input  wire        ctrl_group_done,    // group boundary pulse (once per sub-pass)
    input  wire        ctrl_build_en,
    input  wire        ctrl_compute_en,
    input  wire        ctrl_dma_wt_start,  // run-start (load dims / reset counters)
    input  wire        act_prefetch,       // Task 5: activation transpose-load prefetch (unused here)
    input  wire        wt_mode,            // 1=ternary(4 tiles/group) 0=W4(8 tiles/group)
    output wire [2:0]  wt_tile_idx,        // weight tile for WRF read (0..7)
    output wire [2:0]  act_tile_idx,       // activation tile for build_unit (0..7)
    output reg  [15:0] batch,              // sub-pass-local batch (0..N_BATCH-1)
    output reg  [15:0] batch_prefetch,     // Task 5: activation-read batch (tied 0 here)
    output reg  [15:0] sub_pass,           // current sub-pass (0..S_passes-1)
    output reg  [15:0] j_pass,             // current J pass
    output reg  [15:0] k_group,            // current K group
    // LIVE batch count of the current sub-pass, consumed by acc_block_reg (via
    // sparse_ternary_core).  The last sub-pass of a block may be SHORT -- dim_I=55 gives
    // B_batches=28, S_passes=2 and B_sub = 16 then 12 -- and acc_block_reg must wrap its
    // tile counter after B_sub batches, not after the hard-coded N_BATCH=16.  It is
    // published from here, not recomputed in the core, so the two can never disagree.
    output wire [4:0]  b_sub
);
    // N_BATCH must match sparse_ternary_ctrl / sparse_ternary_core / acc_block_reg.
    localparam N_BATCH = 16;
    localparam BATCH_W = $clog2(N_BATCH);              // 4
    localparam TILE_W  = 3;  // $clog2(8) max; runtime wrap (3 ternary / 7 W4)

    reg [15:0] B_batches, J_passes, K_groups, S_passes;
    reg [TILE_W-1:0]  tile;      // build-side tile (0..7)
    reg [TILE_W-1:0]  tile_d;    // tile delayed 1 presentation (compute-side weight tile)

    // Live batch count for the current sub-pass (the last sub-pass may be short).
    wire [15:0] B_sub = (sub_pass + 1 < S_passes)
                        ? 16'd16                       // N_BATCH
                        : (B_batches - (sub_pass << 4));
    assign b_sub = B_sub[4:0];

    // ── Per-cycle batch/tile: batch inner, tile outer ──
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            batch <= 0; tile <= 0;
        end else if (ctrl_dma_wt_start) begin
            batch <= 0; tile <= 0;
        end else if (ctrl_build_en) begin
            if (batch == B_sub - 16'd1) begin
                batch <= 0;
                tile  <= (tile == (wt_mode ? 3'd3 : 3'd7)) ? 0 : tile + 1'b1;
            end else begin
                batch <= batch + 1'b1;
            end
        end
    end

    // Weight tile lags the activation tile by one presentation (build/compute
    // stagger): at presentation n the build uses tile(n) and the compute uses
    // tile(n-1).  A free-running 1-cycle delay of `tile` realises this.
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) tile_d <= 0;
        else        tile_d <= tile;
    end

    assign act_tile_idx = tile;
    assign wt_tile_idx  = tile_d;

    // ── sub_pass / j_pass / k_group: load dims on run-start, advance on group
    //    boundary. ──
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            B_batches <= 0; J_passes <= 0; K_groups <= 0; S_passes <= 0;
            sub_pass <= 0; j_pass <= 0; k_group <= 0;
        end else begin
            if (ctrl_dma_wt_start) begin
                K_groups  <= dim_K / GROUP_K;
                J_passes  <= (dim_J + N_CH_PER_CORE - 1) / N_CH_PER_CORE;
                B_batches <= (dim_I + R_CORES - 1) / R_CORES;
                S_passes  <= ((dim_I + R_CORES - 1) / R_CORES + N_BATCH - 1) / N_BATCH;
                sub_pass <= 0; j_pass <= 0; k_group <= 0;
            end else if (ctrl_group_done) begin
                if (sub_pass + 1 < S_passes) begin
                    sub_pass <= sub_pass + 1;
                end else begin
                    sub_pass <= 0;
                    if (j_pass + 1 < J_passes) begin
                        j_pass <= j_pass + 1;
                    end else begin
                        j_pass <= 0;
                        if (k_group + 1 < K_groups)
                            k_group <= k_group + 1;
                    end
                end
            end
        end
    end

    // batch_prefetch is the batch-major activation-read prefetch pointer; it has no
    // meaning under the tile-major loop (activation read address changes every
    // cycle) and is re-wired by Task 5.  Tie it off here.
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) batch_prefetch <= 0;
        else        batch_prefetch <= 16'd0;
    end

endmodule
