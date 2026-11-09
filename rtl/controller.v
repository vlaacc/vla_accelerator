// controller.v — Tile sequencer FSM for SparseNN with DMA overlap
//
// Double-buffered weight and activation buffers enable overlapping DMA loads
// of tile N+1 with computation of tile N:
//
//   Tile 0: LOAD_WGT → LOAD_ACT → PRELOAD → COMPUTE ──→ DRAIN → SWAP
//   Tile 1:                                    [DMA wgt+act]           PRELOAD → COMPUTE → DRAIN → SWAP
//   Tile 2:                                                              [DMA]              PRELOAD → ...
//
// Requires: weight_buffer.v and activation_buffer.v with ping-pong banks.

`include "sparsenn_params.vh"

module controller #(
    parameter N = `N_PE_ROWS,
    parameter M = `M_PE_COLS
) (
    input  wire         clk,
    input  wire         rst_n,

    // Host interface
    input  wire         start,
    output reg          done,
    output reg          busy,
    input  wire [15:0]  dim_I,
    input  wire [15:0]  dim_J,
    input  wire [15:0]  dim_K,

    // Systolic array control
    output reg          sa_weight_load,
    output reg          sa_accum_clear,
    output reg          sa_stall,
    output reg  [15:0]  sa_act_addr,
    output reg  [15:0]  sa_wt_addr,

    // Accumulator bank control
    output reg          acc_wr_en,
    output reg  [15:0]  acc_wr_addr,
    output reg          acc_rd_en,
    output reg  [15:0]  acc_rd_addr,
    output reg          acc_accum_en,

    // DMA control
    output reg          dma_act_start,
    output reg          dma_wt_start,
    input  wire         dma_act_done,
    input  wire         dma_wt_done,

    // Tile counters (for debug/logging)
    output reg  [15:0]  tile_i, tile_j, tile_k,
    output reg  [15:0]  total_tiles,

    // Double-buffer bank control
    output reg          wbuf_wr_bank,
    output reg          wbuf_rd_bank,
    output reg          abuf_wr_bank,
    output reg          abuf_rd_bank,

    // Debug: exposed FSM state for performance monitoring
    output wire [3:0]   dbg_state
);

    // ── State encoding ──
    localparam S_IDLE            = 4'd0;
    localparam S_LOAD_WGT        = 4'd1;
    localparam S_LOAD_ACT        = 4'd2;
    localparam S_PRELOAD         = 4'd3;
    localparam S_COMPUTE         = 4'd4;
    localparam S_DRAIN           = 4'd5;
    localparam S_WAIT_DMA        = 4'd6;   // wait for next-tile DMA to finish
    localparam S_SWAP            = 4'd7;   // swap buffer banks
    localparam S_NEXT_TILE       = 4'd8;
    localparam S_DONE            = 4'd9;

    reg [3:0] state, next_state;
    assign dbg_state = state;

    // ── Tile loop variables ──
    reg [15:0] i0, j0, k0;
    reg [15:0] I0, J0, K0;
    reg [15:0] tile_I, tile_J, tile_K;

    // ── Computation / drain counters ──
    reg [15:0] compute_cycles;
    reg [15:0] compute_needed;

    // ── Preload / drain tracking ──
    reg [4:0]  preload_col;   // 5 bits: supports up to M=32
    reg [15:0] drain_row;
    wire weight_preload_done = (preload_col == M);
    wire drain_done          = (drain_row == tile_J);

    // ── DMA overlap tracking ──
    reg        dma_started;          // next-tile DMA has been kicked off
    reg        dma_wt_done_latched;  // dma_done is a 1-cycle pulse; latch it
    reg        dma_act_done_latched;
    wire       dma_both_done_latched = dma_wt_done_latched && dma_act_done_latched;
    wire       has_next_tile;

    // ── Bank phase: which bank is the CURRENT tile reading from? ──
    // first_tile: rd=0, wr=0 (load + read from same bank)
    // After first swap: rd=1, wr=0 → rd=0, wr=1 → rd=1, wr=0 → ...
    reg        bank_phase;        // toggles each swap after first tile

    // ── State machine sequential ──
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            state <= S_IDLE;
        else
            state <= next_state;
    end

    // ── Next-state logic ──
    always @(*) begin
        next_state = state;
        case (state)
            S_IDLE:
                if (start)
                    next_state = S_LOAD_WGT;

            S_LOAD_WGT:
                if (dma_wt_done_latched)
                    next_state = S_LOAD_ACT;

            S_LOAD_ACT:
                if (dma_act_done_latched)
                    next_state = S_PRELOAD;

            S_PRELOAD:
                if (weight_preload_done)
                    next_state = S_COMPUTE;

            S_COMPUTE:
                if (compute_cycles >= compute_needed)
                    next_state = S_DRAIN;

            S_DRAIN:
                if (drain_done) begin
                    if (has_next_tile)
                        next_state = S_WAIT_DMA;
                    else
                        next_state = S_NEXT_TILE;  // go through NEXT_TILE to set done
                end

            S_WAIT_DMA:
                if (dma_both_done_latched)
                    next_state = S_SWAP;

            S_SWAP:
                // swap banks (1 cycle), then advance tile
                next_state = S_NEXT_TILE;

            S_NEXT_TILE:
                next_state = tile_loop_advance();

            S_DONE:
                next_state = S_IDLE;

            default:
                next_state = S_IDLE;
        endcase
    end

    // ── Tile advancement (combinational) ──
    function [3:0] tile_loop_advance;
        begin
            if (k0 + 1 < K0) begin
                tile_loop_advance = S_PRELOAD;
            end else if (j0 + 1 < J0) begin
                tile_loop_advance = S_PRELOAD;
            end else if (i0 + 1 < I0) begin
                tile_loop_advance = S_PRELOAD;
            end else begin
                tile_loop_advance = S_DONE;
            end
        end
    endfunction

    // ── Buffer bank capacity (double-buffered, half depth per bank) ──
    localparam MAX_TILE_K = `WEIGHT_BUF_DEPTH / 2;  // 2048 rows per bank

    // ── Has next tile? (combinational, evaluated BEFORE counters advance) ──
    assign has_next_tile = (k0 + 1 < K0) || (j0 + 1 < J0) || (i0 + 1 < I0);

    // ── Main execution logic ──
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            done <= 0;
            busy <= 0;
            sa_weight_load <= 0;
            sa_accum_clear <= 0;
            sa_stall <= 0;
            sa_act_addr <= 0;
            sa_wt_addr <= 0;
            acc_wr_en <= 0; acc_wr_addr <= 0;
            acc_rd_en <= 0; acc_rd_addr <= 0;
            acc_accum_en <= 0;
            dma_act_start <= 0; dma_wt_start <= 0;
            i0 <= 0; j0 <= 0; k0 <= 0;
            I0 <= 0; J0 <= 0; K0 <= 0;
            tile_I <= N; tile_J <= M; tile_K <= MAX_TILE_K;
            compute_cycles <= 0; compute_needed <= 0;
            preload_col <= 0; drain_row <= 0;
            total_tiles <= 0;
            dma_started <= 0;
            bank_phase <= 0;
            wbuf_wr_bank <= 0; wbuf_rd_bank <= 0;
            abuf_wr_bank <= 0; abuf_rd_bank <= 0;
            dma_wt_done_latched <= 0;
            dma_act_done_latched <= 0;
        end else begin
            // ── Latch DMA done pulses (dma_done is a 1-cycle strobe) ──
            if (dma_wt_done) dma_wt_done_latched <= 1;
            if (dma_act_done) dma_act_done_latched <= 1;

            // Defaults
            sa_weight_load <= 0;
            sa_accum_clear <= 0;
            sa_stall <= 0;
            acc_wr_en <= 0; acc_rd_en <= 0;

            case (state)
                S_IDLE: begin
                    busy <= 0;
                    done <= 0;
                    preload_col <= 0;
                    drain_row <= 0;
                    dma_started <= 0;
                    bank_phase <= 0;
                    // First tile: load into bank 0, read from bank 0
                    wbuf_wr_bank <= 0; wbuf_rd_bank <= 0;
                    abuf_wr_bank <= 0; abuf_rd_bank <= 0;
                    if (start) begin
                        busy <= 1;
                        tile_I <= (dim_I < N) ? dim_I : N;
                        tile_J <= (dim_J < M) ? dim_J : M;
                        tile_K <= (dim_K < MAX_TILE_K) ? dim_K : MAX_TILE_K;
                        I0 <= (dim_I + N - 1) / N;
                        J0 <= (dim_J + M - 1) / M;
                        K0 <= (dim_K <= MAX_TILE_K) ? 1 :
                              ((dim_K + MAX_TILE_K - 1) / MAX_TILE_K);
                        i0 <= 0; j0 <= 0; k0 <= 0;
                        total_tiles <= 0;
                        dma_wt_start <= 1;
                    end
                end

                S_LOAD_WGT: begin
                    if (dma_wt_done_latched) begin
                        dma_wt_start <= 0;
                        dma_wt_done_latched <= 0;
                        dma_act_start <= 1;
                    end
                end

                S_LOAD_ACT: begin
                    if (dma_act_done_latched) begin
                        dma_act_start <= 0;
                        dma_act_done_latched <= 0;
                        preload_col <= 0;
                        sa_accum_clear <= 1;
                    end
                end

                S_PRELOAD: begin
                    sa_accum_clear <= 0;
                    sa_weight_load <= 1;
                    sa_wt_addr <= {12'd0, preload_col};
                    preload_col <= preload_col + 1;

                    if (weight_preload_done) begin
                        sa_weight_load <= 0;
                        compute_cycles <= 0;
                        compute_needed <= tile_K + M + N;
                    end
                end

                S_COMPUTE: begin
                    // Stream activations from read bank
                    sa_act_addr <= compute_cycles;
                    compute_cycles <= compute_cycles + 1;

                    // ═══ DMA OVERLAP: start next tile's DMA on first COMPUTE cycle ═══
                    if (!dma_started && has_next_tile) begin
                        dma_wt_start <= 1;
                        dma_act_start <= 1;
                        dma_started <= 1;
                        // Clear stale latches from any previous DMA
                        dma_wt_done_latched <= 0;
                        dma_act_done_latched <= 0;
                        // Write to the OTHER bank (not the one we're reading from)
                        wbuf_wr_bank <= ~wbuf_rd_bank;
                        abuf_wr_bank <= ~abuf_rd_bank;
                    end
                end

                S_DRAIN: begin
                    acc_wr_en <= 1;
                    acc_wr_addr <= drain_row;
                    drain_row <= drain_row + 1;
                end

                S_WAIT_DMA: begin
                    // Wait for next-tile DMA to finish (both weight + activation)
                    if (dma_both_done_latched) begin
                        dma_wt_start <= 0;
                        dma_act_start <= 0;
                        dma_started <= 0;
                        dma_wt_done_latched <= 0;
                        dma_act_done_latched <= 0;
                    end
                end

                S_SWAP: begin
                    // Toggle banks: old write bank → new read bank, old read bank → new write bank
                    // After first tile: rd was 0, wr was 0. Swap → rd=1, wr=0.
                    // After second tile: rd=1, wr=0. Swap → rd=0, wr=1.
                    if (bank_phase) begin
                        wbuf_rd_bank <= 0; wbuf_wr_bank <= 1;
                        abuf_rd_bank <= 0; abuf_wr_bank <= 1;
                    end else begin
                        wbuf_rd_bank <= 1; wbuf_wr_bank <= 0;
                        abuf_rd_bank <= 1; abuf_wr_bank <= 0;
                    end
                    bank_phase <= ~bank_phase;
                    drain_row <= 0;
                    preload_col <= 0;
                end

                S_NEXT_TILE: begin
                    // Advance tile loop counters
                    if (k0 + 1 < K0) begin
                        k0 <= k0 + 1;
                    end else begin
                        k0 <= 0;
                        if (j0 + 1 < J0) begin
                            j0 <= j0 + 1;
                        end else begin
                            j0 <= 0;
                            if (i0 + 1 < I0) begin
                                i0 <= i0 + 1;
                            end else begin
                                done <= 1;
                                busy <= 0;
                            end
                        end
                    end
                    total_tiles <= total_tiles + 1;
                end

                S_DONE: begin
                    done <= 0;
                end

                default: ;
            endcase
        end
    end

endmodule
