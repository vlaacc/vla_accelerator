// weight_tile_cache.v — BLOCK-PING-PONG weight cache (tile-major weight-stationary)
//
// Holds ONE K-tile of weight (N_CH × 16 weights × 4 bit = 16 Kbit) in a DFF buffer and
// broadcasts it (16 Kbit/cycle) while OTHER buffers are refilled from SRAM with the tiles
// the broadcast will need later.
//
// ── What changed on 2026-09-27 (issue #143) ────────────────────────────────────────
// The refill used to run a THREE-buffer rotation: tile T went into buffer T%3.  Two
// consequences, both measured on the seconds-level harness sim/tb_wtc_refill.v:
//
//   * a tile had to be RE-READ from SRAM every time the broadcast came back round to it,
//     so the block's tiles were re-fetched once per sub-pass; and
//   * the refill could not run further than two tiles ahead (`cur_tile <= bcast_tile+2`),
//     because a third tile ahead would have landed in the buffer the broadcast was about
//     to read.
//
// Together those make the correctness of the broadcast depend on how many PRESENTATIONS
// a tile is broadcast for, i.e. on the sub-pass length B_sub.  Measured (real-top
// pattern: one full sub-pass, then a short one): ternary is correct iff every tile window
// is >= 7 cycles, W4 iff >= 8.  The greedy sub-pass blocking (16,16,...,remainder) leaves
// dim_I=33/65 with a 1-batch last sub-pass (wrong in both modes, L2 `342`/`512` errors)
// and W4 dim_I=110 with a 7-batch one (also wrong — the top-level W4 tb never ran that
// shape, so it was invisible).
//
// The fix is STRUCTURAL: one buffer PER TILE POSITION in the block, and a refill that
// walks the block's tiles 0..N-1 exactly ONCE and then stops.  The tile a buffer holds is
// now a function of its index, so
//
//   * a refill can never clobber a buffer the broadcast still needs (tile T lands in
//     buffer T, and buffer T is never written again within the block), and
//   * the refill's rate stops being on the critical path after the block's first pass:
//     N_TILES tiles are fetched over the whole block instead of once per sub-pass.
//
// The remaining constraint is the FIRST pass through a block's tiles (the broadcast must
// not reach tile T before it is fetched).  `preload_all` covers the one regime where that
// is a real limit: a block whose first sub-pass is itself short (S_passes == 1, i.e.
// B_batches <= N_BATCH), where there is no earlier full sub-pass to build the lead.
//
// Handshake (sparse_ternary_top.v's wgt_fill_handshake contract):
//
//   `start`       = the DMA tlast: block m is complete in region `blk`.  Arms the NXT
//                   channel to preload the first `preload_n` tiles into the *other* bank.
//                   This happens DURING block m-1's compute, so nothing the broadcast is
//                   reading is touched.
//   `blk_switch`  = the ctrl consumed the block (left S_LOAD_WGT).  The NXT bank becomes
//                   the broadcast bank, `bcast_tile` restarts at 0, and the CUR channel
//                   inherits the NXT counters.
//   `ready`       = the incoming bank holds the first `preload_n` tiles.  The top ANDs it
//                   with the arrived/consumed counter pair, so a prefetched block is
//                   reported ready even though the ctrl asked for it later.
//   `tile_rdy`    = the buffer the broadcast will need next is resident.  Nothing stalls
//                   on it in the top (a mid-group stall would drain the LUT pipeline); it
//                   is the contract sim/tb_wtc_refill.v pins.
module weight_tile_cache #(
    parameter N_CH      = 256,
    parameter W_CH_TILE = 64,                  // 16 weights × 4 bit
    parameter TILE_W    = N_CH * W_CH_TILE,    // 16384
    parameter SRAM_W    = 1024,                // SRAM read width (2048 in the top)
    parameter N_TILES   = 8,
    parameter CHUNKS    = TILE_W / SRAM_W      // 8 (2048 b)
) (
    input  wire clk,
    input  wire rst_n,
    input  wire wt_mode,                       // 1=ternary(4 tiles) 0=W4(8 tiles)
    // SRAM read port (1-cycle read latency)
    input  wire [SRAM_W-1:0] sram_dout,
    output wire [$clog2(N_TILES*CHUNKS):0] sram_addr,   // + 1 block (region) bit
    output reg  sram_csb,                       // active low, held selected during refill
    // ASAP7 1RW: the weight SRAM port is SHARED between this refill read and the DMA
    // fill write.  The top arbitrates and asserts sram_rdy only on cycles the read
    // wins; a de-asserted cycle stalls both refill channels and suppresses the capture.
    input  wire sram_rdy,
    // control
    input  wire start,                          // pulse: a block's DMA completed (region `blk`)
    input  wire blk,                            // SRAM region holding that block
    input  wire blk_switch,                     // pulse: ctrl consumed it -> switch banks
    input  wire tile_boundary,                  // pulse: advance broadcast tile
    // 1 = the incoming block's FIRST sub-pass is short: preload the WHOLE block before
    // `ready`, because with no earlier full sub-pass there is no time to fetch tiles 3..
    // at the broadcast's pace.  Driven by sparse_ternary_top from B_batches < 8.
    input  wire preload_all,
    output wire ready,                          // incoming bank: tiles 0..preload_n-1 resident
    output wire tile_rdy,                       // the next broadcast tile is resident
    // broadcast
    output wire [TILE_W-1:0] wt_out
);
    localparam AW = $clog2(N_TILES * CHUNKS);  // SRAM words per block (6)
    localparam CW = $clog2(CHUNKS);            // chunks per tile (3)
    localparam TW = $clog2(N_TILES);           // tiles per block (3)

    // ── ONE buffer per tile POSITION of the block, per bank ──
    // (bank 0 = A, bank 1 = B; the ping-pong lets the next block be preloaded while the
    //  current one is broadcast.)
    reg [SRAM_W-1:0] bufA [0:N_TILES*CHUNKS-1];
    reg [SRAM_W-1:0] bufB [0:N_TILES*CHUNKS-1];

    reg         bcast_bank;    // bank the broadcast reads
    reg [11:0]  bcast_tile;    // UNWRAPPED broadcast tile (0,1,2,...)

    // ── CUR channel: walks the broadcast bank's tiles 0..n_mode-1 and then stops ──
    reg         cur_en;
    reg [11:0]  cur_tile;
    reg [CW-1:0] cur_cnt;
    reg         cur_region;    // SRAM region of the block being broadcast

    // ── NXT channel: preloads the incoming block into the other bank ──
    reg         nxt_active;
    reg [11:0]  nxt_tile;
    reg [CW-1:0] nxt_cnt;
    reg         nxt_region;

    // write-side tracking: dout arrives 1 cycle after its address is issued, so capture
    // (bank, buffer, chunk) of the address issued THIS cycle and write it NEXT cycle.
    reg         wr_valid;
    reg         wr_bank;
    reg [TW-1:0] wr_buf;
    reg [CW-1:0] wr_cnt;

    integer i;

    // ── derived ──
    // Boxed by the tiles the CURRENT mode has (4 ternary / N_TILES W4).  `cur_tp` doubles
    // as the buffer index AND the SRAM tile index: that is what makes a tile readable
    // straight into the buffer it belongs to, with no rotation and no re-read.
    wire [31:0]   n_mode    = wt_mode ? 32'd4 : N_TILES;
    wire [31:0]   preload_n = preload_all ? n_mode : 32'd3;
    wire [TW-1:0] bcast_tp  = wt_mode ? bcast_tile[1:0] : bcast_tile[TW-1:0];
    wire [TW-1:0] cur_tp    = wt_mode ? cur_tile[1:0]   : cur_tile[TW-1:0];
    wire [TW-1:0] nxt_tp    = wt_mode ? nxt_tile[1:0]   : nxt_tile[TW-1:0];

    // ── port arbitration: the broadcast bank has priority.  The CUR channel free-runs to
    //    the end of the block (no rate throttle: with a buffer per tile position, running
    //    ahead is not just safe, it is what removes the refill from the critical path),
    //    and the preload uses whatever cycles are left. ──
    wire cur_want = cur_en     && (cur_tile < n_mode);
    wire nxt_want = nxt_active && (nxt_tile < preload_n);
    wire issue_cur = sram_rdy && cur_want;
    wire issue_nxt = sram_rdy && !cur_want && nxt_want;

    // Word index inside a block.  ⚠️ Sized explicitly: inside a concatenation the
    // arithmetic is SELF-determined, so `{region, tile*CHUNKS + cnt}` would be evaluated
    // at the width of `CHUNKS` (an unsized parameter) and the region bit would be shifted
    // out of the truncated assignment — the refill silently read region 0 for every block
    // (found by sim/tb_wtc_refill.v: the second block's broadcast carried the first
    // block's tags).
    wire [AW-1:0] cur_word = cur_tp * CHUNKS + cur_cnt;
    wire [AW-1:0] nxt_word = nxt_tp * CHUNKS + nxt_cnt;

    // Continuous SRAM read address (one address per issuing cycle).  When neither channel
    // issues, the NXT address is presented; the capture logic discards the dout.
    assign sram_addr = issue_cur ? {cur_region, cur_word} : {nxt_region, nxt_word};

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            bcast_bank <= 1'b0;
            bcast_tile <= 12'd0;
            cur_en     <= 1'b0;
            cur_tile   <= 12'd0;
            cur_cnt    <= {CW{1'b0}};
            cur_region <= 1'b0;
            nxt_active <= 1'b0;
            nxt_tile   <= 12'd0;
            nxt_cnt    <= {CW{1'b0}};
            nxt_region <= 1'b0;
            sram_csb   <= 1'b0;
            wr_valid   <= 1'b0;
            wr_bank    <= 1'b0;
            wr_buf     <= {TW{1'b0}};
            wr_cnt     <= {CW{1'b0}};
            for (i = 0; i < N_TILES*CHUNKS; i = i + 1) begin
                bufA[i] <= {SRAM_W{1'b0}};
                bufB[i] <= {SRAM_W{1'b0}};
            end
        end else begin
            sram_csb <= 1'b0;

            if (blk_switch) begin
                // The incoming block becomes the broadcast block.  The CUR channel takes
                // over exactly where the NXT channel stopped, and the NXT channel goes
                // idle until the next tlast.
                bcast_bank <= ~bcast_bank;
                bcast_tile <= 12'd0;
                cur_en     <= 1'b1;
                cur_tile   <= nxt_tile;
                cur_cnt    <= nxt_cnt;
                cur_region <= nxt_region;
                nxt_active <= 1'b0;
                nxt_tile   <= 12'd0;
                nxt_cnt    <= {CW{1'b0}};
            end else begin
                if (tile_boundary)
                    bcast_tile <= bcast_tile + 1'b1;

                if (start) begin
                    // A block's DMA just landed in region `blk`.  Load its first
                    // `preload_n` tiles into the bank the broadcast is NOT using.
                    nxt_tile   <= 12'd0;
                    nxt_cnt    <= {CW{1'b0}};
                    nxt_region <= blk;
                    nxt_active <= 1'b1;
                end

                // Continuous stream: issue one address/cycle while the bound allows.
                if (issue_cur) begin
                    if (cur_cnt == (CHUNKS - 1)) begin
                        cur_cnt  <= {CW{1'b0}};
                        cur_tile <= cur_tile + 1'b1;
                    end else
                        cur_cnt <= cur_cnt + 1'b1;
                end
                if (issue_nxt) begin
                    if (nxt_cnt == (CHUNKS - 1)) begin
                        nxt_cnt  <= {CW{1'b0}};
                        nxt_tile <= nxt_tile + 1'b1;
                    end else
                        nxt_cnt <= nxt_cnt + 1'b1;
                end
            end

            // Write the chunk whose dout arrived this cycle (1-cycle read latency).
            // The buffer is named by the TILE POSITION, so a write can only ever land in
            // the buffer that tile belongs to.  Since the CUR channel walks each position
            // exactly once per block, no write can alias a buffer the broadcast holds.
            if (wr_valid) begin
                if (wr_bank) bufB[wr_buf*CHUNKS + wr_cnt] <= sram_dout;
                else         bufA[wr_buf*CHUNKS + wr_cnt] <= sram_dout;
            end
            // Capture the (bank, buffer, chunk) of the address issued this cycle.  A
            // cycle that issues nothing delivers no dout, so nothing is written.
            wr_valid <= issue_cur || issue_nxt;
            wr_bank  <= issue_cur ? bcast_bank : ~bcast_bank;
            wr_buf   <= issue_cur ? cur_tp    : nxt_tp;
            wr_cnt   <= issue_cur ? cur_cnt   : nxt_cnt;
        end
    end

    // ── broadcast: mux the broadcast bank's buffers by the tile's position in the block ──
    genvar c;
    generate
        for (c = 0; c < CHUNKS; c = c + 1) begin : bc
            assign wt_out[c*SRAM_W +: SRAM_W] =
                   bcast_bank ? bufB[bcast_tp*CHUNKS + c] : bufA[bcast_tp*CHUNKS + c];
        end
    endgenerate

    // The incoming block is usable once its first `preload_n` tiles are resident.
    assign ready = nxt_active && (nxt_tile >= preload_n);

    // ── tile_rdy: "the buffer for bcast_tile+1 is resident" ──────────────────────
    // `cur_tile` is the tile the CUR channel is reading NOW, so every tile below it has
    // been loaded.  Once the whole block is loaded (`cur_tile >= n_mode`) every position
    // is resident and the flag simply stays high.
    assign tile_rdy = (cur_tile >= bcast_tile + 12'd2) || (cur_tile >= n_mode);

endmodule
