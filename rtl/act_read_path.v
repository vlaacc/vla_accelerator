// act_read_path.v — everything between the activation macros and the two cores' act_row
// inputs.  Extracted from sparse_ternary_top.v for the same reason act_port_mux was:
// the seconds-level audit harness (sim/tb_act_port_audit.v) must instantiate the
// SHIPPING read path — port mux, line buffer, and the register that sets the 2-cycle
// depth — with a behavioural macro model around it, instead of re-implementing it.
//
// WHAT IT DOES (plan (b))
//   * TILE 0 of every sub_pass: the SRAM read IS the core's data path (and the same
//     read fills the line buffer, so the refill costs no extra port cycle at all).
//     `act_sram_rd` is high, `act_port_mux` gives the read the port, and the selector
//     below takes the macro dout.
//   * TILES 1..T_PER-1: the core is served by the line buffer, which holds exactly what
//     tile 0 read — the presentation order is tile-outer/batch-inner, so the same
//     16-row ramp replays and the buffer is still valid.  The SRAM port is free and
//     belongs to the DMA.  Measured effect: the store is read once per distinct row per
//     sub_pass instead of once per presentation (sim/tb_act_port_audit.v A2), which is
//     the redundancy that pays for the overlap.
//
// TIMING — this is the load-bearing part (sparse_ternary_top.v's alignment comment).
// For the address/row presented at cycle k the core must see act_row at k+2:
//   SRAM : addr@k -> macro dout@k+1 (act_dout) -> act_row@k+2      [2 cycles, unchanged]
//   LB   : rd_idx@k -> rd_data@k+1 (synchronous read) -> act_row@k+2 [2 cycles]
// The two are the same depth, so the select is just `act_sram_rd` delayed by one — and
// the SRAM ADDRESS is NOT delayed (registering it would make 3, one short and register-
// ing nothing would make 1).  `act_ab_d`/`act_sram_rd_d` below are the line buffer's
// WRITE side (which consumes the dout a cycle after the address) and the source select;
// neither is in the read address path.
//
// The alignment contracts this preserves, unchanged:
//   * `act_row` is 2 cycles behind the controller's build/compute schedule (the other
//     core inputs are delayed 2 to match, compute_en 3);
//   * `act_settle` in the ctrl still covers the post-load SRAM latency, because tile 0
//     of the first sub_pass still reads the macro.

`timescale 1ns / 1ps

module act_read_path #(
    parameter ROW_IDX_W = 8            // macro address width (SRAM1RW256x64: 256 words)
) (
    input  wire           clk,
    input  wire           rst_n,

    // compute-side row index (sub_pass*N_BATCH + batch) and its two regions
    input  wire [15:0]    abs_batch,
    input  wire           act_sram_rd,      // the core needs the SRAM read this cycle
    input  wire [10:0]    act_load_cnt,     // DMA beat counter
    input  wire           dma_act_wr_en,    // the DMA is presenting a beat this cycle
    input  wire           act_overlap_en,   // two region halves fit
    input  wire           act_rd_region,    // region the core reads (k_group parity)
    input  wire           act_wr_region,    // region the DMA fills (block parity)

    // macro side
    input  wire [511:0]   act_din,          // one DMA beat = one bank-row
    input  wire [1023:0]  act_dout,         // {bank1, bank0} at `act_scratch_addr`
    output wire [ROW_IDX_W-1:0] act_scratch_addr,
    output wire           act_bank0_web,
    output wire           act_bank1_web,
    output wire           act_rd,           // the port is serving a read this cycle

    // core side (2 cycles behind the address)
    output wire [511:0]   act_row_even,
    output wire [511:0]   act_row_odd
);
    act_port_mux u_port_mux (
        .abs_batch(abs_batch), .act_load_cnt(act_load_cnt),
        .dma_act_wr_en(dma_act_wr_en), .act_sram_rd(act_sram_rd),
        .act_overlap_en(act_overlap_en),
        .act_rd_region(act_rd_region), .act_wr_region(act_wr_region),
        .act_scratch_addr(act_scratch_addr),
        .act_bank0_web(act_bank0_web), .act_bank1_web(act_bank1_web),
        .act_rd(act_rd)
    );

    // Write side of the line buffer: the macro dout for the address presented LAST
    // cycle, written to the row that address named.
    reg        act_sram_rd_d;
    reg [15:0] act_ab_d;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin act_sram_rd_d <= 1'b0; act_ab_d <= 16'd0; end
        else begin act_sram_rd_d <= act_sram_rd; act_ab_d <= abs_batch; end
    end

    wire [1023:0] act_lb_rdata;
    act_line_buffer #(.DEPTH(16), .W(1024), .IW(4)) u_lb (
        .clk(clk),
        .wr_en(act_sram_rd_d),
        .wr_idx(act_ab_d[3:0]),
        .wr_data(act_dout),
        .rd_idx(abs_batch[3:0]),
        .rd_data(act_lb_rdata)
    );

    // Source select, one cycle after the decision: `act_sram_rd` at cycle k says which
    // path the k+2 presentation takes, and both candidates are valid at k+1.
    wire [1023:0] act_row_src = act_sram_rd_d ? act_dout : act_lb_rdata;

    reg [511:0] act_row_even_r, act_row_odd_r;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin act_row_even_r <= 512'd0; act_row_odd_r <= 512'd0; end
        else begin
            act_row_even_r <= act_row_src[511:0];
            act_row_odd_r  <= act_row_src[1023:512];
        end
    end

    assign act_row_even = act_row_even_r;
    assign act_row_odd  = act_row_odd_r;
endmodule
