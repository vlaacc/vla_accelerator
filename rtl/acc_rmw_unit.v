// acc_rmw_unit.v — sub_sum -> acc SRAM INT32 read-modify-write accumulator
// (**ASAP7 1RW version, scheme C4: a row is FOUR sub-word accesses**)
//
// Consumer of sub_sum_fifo.  Each FIFO entry (2 core x 256 ch x 24 b = 12288 b)
// carries the FULL signed dot (the Horner already folded all W bit-planes), so this
// unit only sign-extends 24 b -> 32 b and accumulates (no shift/sign).
//
// ── Scheme C4: 16384 b -> 4096 b bus ──
// The acc bank is 64 x srambank_256x4x64 (4096 b wide x 1024 deep) instead of
// 256 x srambank_64x4x64 (16384 b wide x 256 deep).  Same 512 KB, but the long
// bus and the macro pin count both drop 4x.  The asap7 DRT floor is a Lef58 V3
// cut-enclosure violation at every SRAM pin (the macro's M4 pin is 0.024 tall,
// exactly the single-cut length, so a single-cut via has zero enclosure and the
// rule wants 0.02425); 4x fewer pins means ~4x fewer of those.
//
// A logical row (16384 b) is split into 4 sub-words of 4096 b:
//   sub-word k per core covers channels [k*64 .. k*64+63]
//   acc_addr = {row[7:0], sel[1:0]}
//
// Port traffic per entry is unchanged at 2 x 16384 b = 32768 b; it now moves in
// 8 accesses of 4096 b.
//
//   step 0 : rd sub0     step 4 : wr sub0   (acc_dout for sub0 is valid here --
//   step 1 : wr sub0         ...              the macro registers its read)
//   step 2 : rd sub1         ...
//   step 3 : wr sub1     step 7 : wr sub3
//   step 8 : bubble       -> group_done + prefetch the next FIFO entry
//
// No rd staging buffer is needed: read at step 2k, write at step 2k+1, and the
// macro's registered output is still valid on that write cycle.
//
// fifo_data layout: {first, row, core1, core0}: core c at [c*6144 +: 6144],
//   channel ch (24 b) at [c*6144 + ch*24 +: 24].

module acc_rmw_unit #(
    parameter N_CORE     = 2,
    parameter N_CH_CORE  = 256,
    parameter SUBSUM_W   = 24,    // input full-dot sub_sum width
    parameter ACC_W      = 32,    // acc lane width
    parameter ROW_W      = 8,     // 256 logical rows
    parameter FIRST_W    = 1,     // "first k_group" flag: overwrite vs accumulate
    parameter ACC_SUB_W  = 2      // 2 -> four 4096-b sub-words per row
) (
    input  wire                          clk,
    input  wire                          rst_n,
    input  wire                          fifo_empty,
    output reg                           fifo_rd_en,
    input  wire [FIRST_W+ROW_W+N_CORE*N_CH_CORE*SUBSUM_W-1:0] fifo_data,

    // ── Acc SRAM: one shared address, one access = one SUB-WORD ──
    input  wire [N_CORE*N_CH_CORE*ACC_W/(1<<ACC_SUB_W)-1:0]  acc_dout,
    output reg  [N_CORE*N_CH_CORE*ACC_W/(1<<ACC_SUB_W)-1:0]  acc_din,
    output reg                                               acc_web,   // active low
    output reg  [ROW_W+ACC_SUB_W-1:0]                        acc_addr,  // {row, sel}
    output reg                                               group_done
);
    localparam ACC_FULL_W  = N_CORE * N_CH_CORE * ACC_W;              // 16384
    localparam SUB_W       = ACC_FULL_W >> ACC_SUB_W;                 // 4096
    localparam SUB_SLICE_W = SUB_W / N_CORE;                          // 2048 per core
    localparam LANES_SUB   = N_CH_CORE >> ACC_SUB_W;                  // 64 lanes per core
    localparam STEPS       = (1 << (ACC_SUB_W+1)) + 1;                // 8 accesses + bubble
    localparam STEP_W      = 4;

    reg [STEP_W-1:0] step;
    reg              active;
    reg [FIRST_W+ROW_W+N_CORE*N_CH_CORE*SUBSUM_W-1:0] latched;

    wire [ROW_W-1:0] latched_row   = latched[N_CORE*N_CH_CORE*SUBSUM_W +: ROW_W];
    wire             latched_first = latched[N_CORE*N_CH_CORE*SUBSUM_W + ROW_W];

    wire [ACC_SUB_W-1:0] cur_sel = step[ACC_SUB_W:1];                 // sub-word index
    wire                 do_read  = active && !step[0];               // even steps
    wire                 do_write = active &&  step[0];               // odd steps
    // the write cycle of sub-word k is one cycle after its read, so the macro's
    // registered output is still valid -- no staging buffer required.

    // ── accumulate / overwrite, per core per channel of the CURRENT sub-word ──
    genvar c, l;
    generate
        for (c = 0; c < N_CORE; c = c + 1) begin : core
            for (l = 0; l < LANES_SUB; l = l + 1) begin : lane
                wire [SUBSUM_W-1:0] ss_raw =
                    latched[c*(N_CH_CORE*SUBSUM_W)
                            + (cur_sel*LANES_SUB + l)*SUBSUM_W +: SUBSUM_W];
                wire signed [ACC_W-1:0] ss_val =
                    {{(ACC_W-SUBSUM_W){ss_raw[SUBSUM_W-1]}}, ss_raw};
                always @(*) begin
                    if (latched_first)
                        acc_din[c*SUB_SLICE_W + l*ACC_W +: ACC_W] = ss_val;
                    else
                        acc_din[c*SUB_SLICE_W + l*ACC_W +: ACC_W] =
                            acc_dout[c*SUB_SLICE_W + l*ACC_W +: ACC_W] + ss_val;
                end
            end
        end
    endgenerate

    // ── combinational SRAM control, aligned to the registered `step` ──
    always @(*) begin
        acc_addr = {latched_row, cur_sel};
        acc_web  = do_write ? 1'b0 : 1'b1;      // active low
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            fifo_rd_en <= 0; active <= 0; step <= 0;
            group_done <= 0; latched  <= 0; acc_din <= 0;
        end else begin
            fifo_rd_en <= 0; group_done <= 0;

            if (!active) begin
                if (!fifo_empty) begin
                    fifo_rd_en <= 1;
                    active     <= 1;
                    step       <= 0;
                    latched    <= fifo_data;
                end
            end else begin
                if (step == STEP_W'(STEPS - 1)) begin
                    group_done <= 1;            // the last write was issued last cycle
                    if (!fifo_empty) begin
                        fifo_rd_en <= 1;
                        latched    <= fifo_data;
                        step       <= 0;        // active stays 1, back-to-back
                    end else begin
                        active <= 0;
                    end
                end else begin
                    step <= step + 1'b1;
                end
            end
        end
    end
endmodule
