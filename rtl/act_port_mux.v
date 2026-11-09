// act_port_mux.v — who owns the activation SRAM's single address port this cycle?
//
// The act store is 2 banks x 8 x SRAM1RW256x64, and ALL 16 macros share ONE address
// (`addr0`) plus one `csb0` (tied low: an access is issued every cycle).  A "bank" is
// therefore half of one 1024-b access, not an independent port.  Bank select is `web0`
// only, so read and write contend for the same port and some arbitration must exist.
//
// Extracted from sparse_ternary_top.v so that the seconds-level audit harness
// (sim/tb_act_port_audit.v) instantiates the SHIPPING expression rather than a
// re-implementation of it -- replicating glue in a harness is how this project has
// previously shipped harnesses that could not fail.
//
// HISTORY.  Phase 1 (the phase mux): the whole DMA window owned the port (`act_loading`),
// which made the load fully serial -- during S_GROUP the core read every cycle and the
// DMA got no slot at all (+4,801 cycles/block, 2026-09-28-gate-dram-recheck.md §4).
// Phase 2 (plan (b), this version): the core reads the SRAM only during TILE 0 of each
// sub_pass (one read per distinct row instead of one per presentation); for the rest of
// the sub_pass it is served by act_line_buffer, and the port belongs to the DMA.
//
// ARBITRATION.  A refill read is LOAD-BEARING — the core is stall-free, so if the read
// is not issued on the cycle the presentation needs it there is no recovery.  The DMA,
// by contrast, is a stream and its source holds the beat (AXI `tready`).  So the refill
// read WINS, and `sparse_ternary_top` drives `s_axis_act_tready = !act_sram_rd`; the
// DMA gets every other cycle of the compute, which is 3x (ternary) / 7x (W4) more than
// it needs at the measured DRAM rate.
//
// REGION BIT.  The DMA fills the NEXT k_group while the core reads the current one, so
// the two must live in different address regions.  A macro is 256 words deep and a block
// only needs ceil(dim_I/2) <= 128 of them, so the top address bit selects the region:
//   read  addr = {act_rd_region, abs_batch[6:0]}
//   write addr = {act_wr_region, act_load_cnt[7:1]}
// `act_wr_region` flips on dma_act_done (one block), `act_rd_region` follows k_group, so
// the pair is always opposite.  When the geometry does NOT fit two regions (dim_I > 256,
// i.e. ceil(dim_I/2) > 128) `act_overlap_en` is 0: the plain addresses are used and the
// ctrl falls back to the phase-1 serial load (no prefetch), so the write can never land
// on a row the core is reading.

`timescale 1ns / 1ps

module act_port_mux (
    input  wire [15:0] abs_batch,      // compute-side row index = sub_pass*N_BATCH + batch
    input  wire [10:0] act_load_cnt,   // DMA-side beat counter (one beat = one row)
    input  wire        dma_act_wr_en,  // the DMA is presenting a beat this cycle
    input  wire        act_sram_rd,  // the core needs a refill read this cycle
    input  wire        act_overlap_en, // two region halves fit (ceil(dim_I/2) <= 128)
    input  wire        act_rd_region,  // region the core is reading
    input  wire        act_wr_region,  // region the DMA is filling
    output wire [7:0]  act_scratch_addr,
    output wire        act_bank0_web,
    output wire        act_bank1_web,
    output wire        act_rd
);
    // Read beats write, always: see ARBITRATION above.
    wire wr = dma_act_wr_en && !act_sram_rd;
    assign act_rd = ~wr;

    wire [7:0] rd_addr = act_overlap_en ? {act_rd_region, abs_batch[6:0]}
                                        : abs_batch[7:0];
    wire [7:0] wr_addr = act_overlap_en ? {act_wr_region, act_load_cnt[7:1]}
                                        : act_load_cnt[8:1];

    assign act_scratch_addr = act_rd ? rd_addr : wr_addr;

    // web0 is low-active: 0 = write, 1 = read.  A beat writes one bank (row parity);
    // the other bank reads whatever the address says.
    assign act_bank0_web = ~(wr && !act_load_cnt[0]);
    assign act_bank1_web = ~(wr &&  act_load_cnt[0]);
endmodule
