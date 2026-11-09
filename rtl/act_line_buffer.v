// act_line_buffer.v — the small register file that sits between the activation SRAM
// and the two LUT cores.
//
// WHY IT EXISTS (plan (b), docs/superpowers/plans/2026-09-28-gate-dram-recheck.md §5)
// ---------------------------------------------------------------------------------
// The act store is 2 banks x 8 x SRAM1RW256x64 on ONE shared address, and `csb0` is
// tied low, so an access is issued EVERY cycle.  The core reads 1024 b/cycle out of it
// but build_unit consumes only 128 b per core per cycle, and `abs_batch` has no tile
// term — so within a sub_pass the SAME 16-row ramp is re-read once per tile.  Measured
// (sim/tb_act_port_audit.v, A1/A2): a read on every S_GROUP cycle, each row read
// T_PER*J_passes times.  That redundancy is the only thing standing between the DMA and
// the port, and it is what this buffer converts into DMA cycles.
//
// SHAPE: DEPTH rows x W bits.  At the shipping geometry that is 16 x 1024 b = one
// sub_pass's worth of activation rows (1024 b = 2 banks x 512 b = the even/odd row pair
// for one absolute batch).
//
// LATENCY: this is a SYNCHRONOUS-read register file — `addr@k -> rd_data@k+1` — which is
// EXACTLY the SRAM1RW macro's address->dout latency.  That is the whole reason the
// 2-cycle alignment (addr@k -> dout@k+1 -> act_row@k+2, see sparse_ternary_top.v) is
// preserved to the cycle when the core is switched onto this path.  Do NOT turn this
// into a combinational read and do NOT register the read address: the first makes the
// path 1 cycle short and the second makes it 1 long, and both have cost multi-round bugs
// before (see the alignment comment in sparse_ternary_top.v).
//
// The write port is independent (a plain FF array), so a row for the NEXT sub_pass can
// be written the cycle after the SRAM dout for it appears without disturbing the read.

`timescale 1ns / 1ps

module act_line_buffer #(
    parameter DEPTH = 16,
    parameter W     = 1024,
    parameter IW    = 4
) (
    input  wire          clk,
    input  wire          wr_en,
    input  wire [IW-1:0] wr_idx,
    input  wire [W-1:0]  wr_data,
    input  wire [IW-1:0] rd_idx,
    output reg  [W-1:0]  rd_data
);
    reg [W-1:0] mem [0:DEPTH-1];

    // Non-blocking on both ports: a same-cycle read/write of one index reads the OLD
    // value (Verilog NBA semantics), which is what the mux in act_read_path relies on
    // during a refill.
    always @(posedge clk) begin
        if (wr_en) mem[wr_idx] <= wr_data;
        rd_data <= mem[rd_idx];
    end
endmodule
