// int8_engine.v — INT8 tiling controller: 4 x 32x32 mesh + act SRAM + acc SRAM
//
// This is the piece the ASAP7 1RW plan lists as "唯一还没设计的环节" (T16).  It owns
// the INT8-side memory hierarchy and drives one complete GEMM pass per `start`:
//
//     C[128 x 32] = A[128 x K] x W[K x 32],   K = 32
//
// ── Tiling (M-split, from the frozen spec) ──
// The 4 meshes hold the SAME 32K x 32N weight block and process 4 DIFFERENT m rows,
// so the weight is broadcast once (int8_mesh_top) and the activations are what gets
// replicated.  Mesh g owns global rows { 32g + r : r = 0..31 }.
//
// ── Memory layouts ──
// act SRAM (1024 b = MESHES x K x 8 b), address = m_tile*K + r:
//     chunk g (256 b) = A[m_tile*128 + 32g + r][0 .. K-1]        (one row, K k-values)
//     => one address yields all four meshes' next activation word at once, which is
//        exactly the 1024 b/cycle the M-split costs.
//
// acc SRAM (4096 b = MESHES x N x 32 b), address = m_tile*K + r:
//     lane group g (1024 b) = C[m_tile*128 + 32g + r][0 .. N-1]
//     so row m = m_tile*128 + 32g + r lives at acc[m_tile*K + r], lanes [32g*32 +: 1024].
//     One address therefore carries four output rows that differ by 32 in m.
//
// ── Feed / drain timing ──
// rtl/int8_act_skew.v converts the row-major stream into the anti-diagonal feed the
// array needs, so the array emits output row r exactly LAT = K+1 array-cycles after
// it consumed activation row r.  One register sits on the activation path inside the
// macro (its registered read) plus one explicit one, so the feeder's word index at
// step s is s-1:
//
//     step s :  present act address m_tile*K + s        (s = 0 .. K-1)
//               feeder shift                            (s = 1 .. 2K-1)
//               output row r = s - LAG -> acc write      (LAG = K + 2, s = LAG .. MAXS)
//
// Two details are load-bearing:
//   * the read is captured into `act_dout_r` rather than consumed straight off the
//     macro port — the behavioural macro drives dout on the falling edge, so sampling
//     it combinationally at the following rising edge races the *next* access;
//   * the feeder must keep shifting well past the last stored word while its input is
//     forced to zero.  A_i(x) only reaches zero at x = 2K-2, and anything non-zero
//     there folds into a valid output row through the partial-sum chain.
//
// ── v1 scope (deliberate, documented) ──
//   * a single k-block of K = 32, i.e. dim_K must equal `K`.  The array is
//     feed-forward (pe_int8's accumulator is written but never read — see §P2 of the
//     design doc) so a longer K needs a read-modify-write pass over the acc SRAM; that
//     costs 2 acc accesses per output word against the 1 word/cycle the array
//     produces, i.e. it halves the array duty — exactly the "INT8 acc demand =
//     2048 b/cycle = 50% of the 4096 b port" figure in the frozen spec.
//   * N is fixed at the array width (32); an n-tile loop needs a second weight buffer.
//     The weight is stationary across m, and that is the loop that matters here.
//   * no DMA arbitration: `act_fill_*` is honoured only while the engine is idle.
//
// `done` is a one-cycle pulse after the last m tile drains.

`include "sparsenn_params.vh"

module int8_engine #(
    parameter MESHES   = 4,
    parameter K        = 32,      // array rows  = reduction depth per pass
    parameter N        = 32,      // array cols  = output channels per mesh
    parameter ACC_ROWS = 256,
    parameter ACT_ROWS = 1024
) (
    input  wire                    clk,
    input  wire                    rst_n,
    input  wire                    start,
    input  wire [3:0]              m_tiles,      // 128-row m tiles to run (1..8)

    // ── activation SRAM fill (row-major, address = m_tile*K + r) ──
    input  wire                    act_fill_en,
    input  wire [9:0]              act_fill_addr,
    input  wire [MESHES*K*8-1:0]   act_fill_data,

    // ── weight load: K columns x (K row weights x 8 b), broadcast to all meshes ──
    input  wire                    wgt_ld_en,
    input  wire [4:0]              wgt_ld_col,
    input  wire [K*8-1:0]          wgt_ld_data,

    // ── result read-out (idle only) ──
    input  wire                    acc_rd_en,
    input  wire [7:0]              acc_rd_addr,
    output wire [MESHES*N*32-1:0]  acc_rd_data,

    // The act/acc SRAM banks and the vector epilogue are instantiated *inside* this
    // block (they used to live in the DSA top, which forced ~9,200 + 8,192 top-level
    // wires between the engine and the memories and made the parent unroutable).
    // Their 80 SRAM macros stay hard macros in this block's own GDS, so the final
    // merged GDS still shows every SRAM where it physically sits.

    output wire                    done,
    output wire                    busy
);
    localparam ACTW = MESHES*K*8;      // 1024
    localparam ACCW = MESHES*N*32;     // 4096

    // ── local memory / epilogue nets (were top-level ports) ──
    wire             act_csb, act_web;
    wire [9:0]       act_addr;
    wire [ACTW-1:0]  act_dout;
    wire             acc_csb, acc_web;
    wire [7:0]       acc_addr;
    wire [ACCW-1:0]  acc_din;
    wire [ACCW-1:0]  acc_dout;
    reg  [ACCW-1:0]  vec_a_r;          // FSM-captured acc word
    wire [ACCW-1:0]  vec_out;          // vector result written back
    wire signed [31:0] vec_red_sum, vec_red_max;
    wire             vec_en = 1'b1;    // epilogue always runs (parent tied it high)
    localparam LAT  = K + 1;           // array: output row r at array-cycle r + LAT
    // Two registers sit between the address and the feeder: the macro's own
    // registered read (driven on the falling edge) and act_dout_r.  So the feeder's
    // word index at step s is s-2, and output row r lands at step r + LAT + 2.
    localparam RD_LAG = 2;
    localparam LAG  = LAT + RD_LAG;
    localparam MAXS = LAG + K - 1;     // last step of an m tile

    localparam S_IDLE = 3'd0, S_CLR = 3'd1, S_PRE = 3'd2, S_FEED = 3'd3, S_DONE = 3'd4,
               S_VEC  = 3'd5;
    // Vector epilogue phase per acc row: 0 = issue read, 1 = capture (macro read
    // latency), 2..5 = vector_unit_4096 lane latency, 6 = write back.
    localparam VP_WR = 3'd6;

    reg [2:0]  state;
    reg [3:0]  tile;
    reg [7:0]  s;
    reg        done_r;
    wire       preload_done;

    reg [2:0]  vp;
    reg [7:0]  vrow;
    // acc_dout / vec_a_r are local now: the acc bank and the vector unit live in
    // this block.
    // last acc row the GEMM wrote for the final tile = tile*K + (K-1)
    wire [7:0] vec_last = {tile, 5'b0} + (K - 1);

    wire running   = (state == S_FEED);
    wire last_tile = (tile == (m_tiles - 4'd1));
    wire last_step = (s == MAXS[7:0]);

    // ── step decode ──
    // feeder word index w = s - RD_LAG: read window w = 0..K-1, flush to w = 2K-2
    wire       rd_addr_v = running && (s < K);                           // s = 0..K-1
    wire       word_sel  = running && (s >= RD_LAG) && (s < (RD_LAG + K));
    wire       feed_sh   = running && (s >= 8'd1) && (s <= (2*K));
    wire       out_valid = running && (s >= LAG) && (s < (LAG + K));
    wire [7:0] out_row   = s - LAG[7:0];
    wire [9:0] act_a     = {tile, 5'b0} + {5'b0, (rd_addr_v ? s[4:0] : 5'd0)};
    wire [7:0] acc_waddr = {tile, 5'b0} + {3'b0, out_row[4:0]};

    assign busy = (state != S_IDLE);

    // ── FSM ──
    // acc_clear is asserted combinationally for exactly one cycle (state S_CLR), so
    // the mesh's preload FSM goes busy on the following edge and preload_done is
    // still low when S_PRE is first entered.
    wire acc_clear = (state == S_CLR);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state  <= S_IDLE;
            tile   <= 4'd0;
            s      <= 8'd0;
            done_r <= 1'b0;
        end else begin
            case (state)
                S_IDLE: begin
                    done_r <= 1'b0;
                    if (start) begin
                        tile  <= 4'd0;
                        s     <= 8'd0;
                        state <= S_CLR;
                    end
                end
                S_CLR:  state <= S_PRE;
                S_PRE:  if (preload_done) begin s <= 8'd0; state <= S_FEED; end
                S_FEED: begin
                    if (last_step) begin
                        if (last_tile) begin
                            if (vec_en) begin
                                vp   <= 3'd0;
                                vrow <= 8'd0;
                                state <= S_VEC;
                            end else begin
                                done_r <= 1'b1;
                                state  <= S_DONE;
                            end
                        end else begin
                            tile <= tile + 4'd1;
                            s    <= 8'd0;
                        end
                    end else begin
                        s <= s + 8'd1;
                    end
                end
                S_VEC: begin
                    if (vp == 3'd1) vec_a_r <= acc_dout;
                    if (vp == VP_WR) begin
                        if (vrow == vec_last) begin
                            done_r <= 1'b1;
                            state  <= S_DONE;
                        end else begin
                            vrow <= vrow + 8'd1;
                            vp   <= 3'd0;
                        end
                    end else begin
                        vp <= vp + 3'd1;
                    end
                end
                S_DONE: if (!start) begin
                            done_r <= 1'b0;
                            state  <= S_IDLE;
                        end
                default: state <= S_IDLE;
            endcase
        end
    end

    assign done = done_r;

    // ── activation SRAM control: 1RW, engine read has priority, DMA fill when idle ──
    // sram_act_bank_int8 is instantiated below and connected to
    // act_csb/act_web/act_addr (out) and act_dout (in).
    assign act_csb  = running ? 1'b0      : ~act_fill_en;
    assign act_web  = running ? 1'b1      : ~act_fill_en;   // 1 = read
    assign act_addr = running ? act_a     : act_fill_addr;

    // Registered read: the macro drives dout on the falling edge, so consuming it
    // combinationally at the next rising edge would race the following access.
    reg [ACTW-1:0] act_dout_r;
    always @(posedge clk or negedge rst_n)
        if (!rst_n) act_dout_r <= {ACTW{1'b0}};
        else        act_dout_r <= act_dout;

    // ── 4 skew feeders (one per mesh) + the mesh array ──
    wire [MESHES*K*8-1:0] act_skew;
    genvar g;
    generate
        for (g = 0; g < MESHES; g = g + 1) begin : skew
            // words past the tile must read as zero, otherwise the flush tail folds
            // non-zero products into output rows that are still being drained
            wire [K*8-1:0] wsel = word_sel ? act_dout_r[g*K*8 +: K*8] : {K*8{1'b0}};
            int8_act_skew #(.K(K)) u_skew (
                .clk(clk), .rst_n(rst_n), .valid(feed_sh),
                .word_act(wsel),
                .act_skew(act_skew[g*K*8 +: K*8])
            );
        end
    endgenerate

    wire [MESHES*N*32-1:0] psum_out;

    int8_mesh_top #(.MESHES(MESHES), .K(K), .N(N)) u_mesh (
        .clk(clk), .rst_n(rst_n),
        .wgt_ld_en(wgt_ld_en), .wgt_ld_col(wgt_ld_col), .wgt_ld_data(wgt_ld_data),
        .act_in(act_skew),
        .psum_out(psum_out),
        .accum_clear(acc_clear),
        .preload_done(preload_done),
        .dbg_col()
    );

    // ── vector epilogue datapath: one acc word == one vector operation ──
    // vector_unit_4096 is instantiated below; it drives vec_out back in and
    // takes vec_a_r (the FSM-captured acc word) with the fixed dequant params.
    wire            vec_rd = (state == S_VEC) && (vp == 3'd0);
    wire            vec_wr = (state == S_VEC) && (vp == VP_WR);

    // ── accumulator SRAM control: 1RW — GEMM drain write, vector read/write, external read ──
    wire            acc_own  = out_valid | vec_rd | vec_wr;
    assign acc_csb  = acc_own ? 1'b0 : ~acc_rd_en;
    assign acc_web  = (out_valid | vec_wr) ? 1'b0 : 1'b1;   // 0 = write
    assign acc_addr = (out_valid | vec_rd | vec_wr) ? (out_valid ? acc_waddr : vrow)
                                                    : acc_rd_addr;
    assign acc_din  = out_valid ? psum_out : vec_out;

    assign acc_rd_data = acc_dout;

    // ── the two SRAM banks and the vector epilogue, now inside this block ──
    sram_act_bank_int8 u_act_sram (
        .clk (clk), .csb(act_csb), .web(act_web),
        .addr(act_addr), .din({ACTW{1'b0}}), .dout(act_dout)
    );

    sram_acc_bank_int8 u_acc_sram (
        .clk (clk), .csb(acc_csb), .web(acc_web),
        .addr(acc_addr), .din(acc_din), .dout(acc_dout)
    );

    vector_unit_4096 u_vec (
        .clk(clk), .rst_n(rst_n),
        .zp(16'sd0), .scale(16'sd256), .sh(5'd8),
        .acc_rd_data(vec_a_r),
        .b_operand({ACCW{1'b0}}),
        .acc_wr_data(vec_out),
        .op(2'd0), .is_fp(1'b0),
        .vec_out(), .red_sum(vec_red_sum), .red_max(vec_red_max)
    );

endmodule
