// st_soc_top.v — CSR + sparse_ternary_top integration top (signoff-only wrapper)
//
// Wraps config_regs (CSR) around sparse_ternary_top + int8_engine so the accelerator
// has a memory-mapped control shell.
//
// The AXI-S data path goes straight through each engine's inline fill logic; the
// Gemmini-era dma_reader/dma_writer shells are gone (they could not drive the ASAP7
// 1RW full-word SRAMs, and their weight/act/acc buffers do not exist here).
//
// ONE result port, time-multiplexed: the LUT drain (8192 b acc word, dim_I rows) has
// priority, the INT8 drain (4096 b acc word, 32 rows) takes the port the rest of the
// time.  Both acc SRAMs are written independently and are never read at once.

`include "sparsenn_params.vh"

module st_soc_top (
    input  wire          clk,
    input  wire          rst_n,

    // AXI4-Lite (simplified) CSR
    input  wire          csr_wr_en,
    input  wire [7:0]    csr_addr,
    input  wire [31:0]   csr_wr_data,
    output wire [31:0]   csr_rd_data,

    // AXI4-Stream: weight DMA (512b)
    input  wire          s_axis_wgt_tvalid,
    output wire          s_axis_wgt_tready,
    input  wire [511:0]  s_axis_wgt_tdata,
    input  wire          s_axis_wgt_tlast,

    // AXI4-Stream: activation DMA (512b)
    input  wire          s_axis_act_tvalid,
    output wire          s_axis_act_tready,
    input  wire [511:0]  s_axis_act_tdata,
    input  wire          s_axis_act_tlast,

    // AXI4-Stream: result DMA (512b)
    output wire          m_axis_out_tvalid,
    input  wire          m_axis_out_tready,
    output wire [511:0]  m_axis_out_tdata,
    output wire          m_axis_out_tlast,

    // Control
    input  wire          weight_signed,   // 1=signed / 0=unsigned (W8 two-pass)
    input  wire          wt_mode,         // 1=ternary(K=32) / 0=W4(K=16)

    // ── LUT engine DMA / error handshake (host side) ──
    // dma_wt_start: one pulse per (k_group, j_pass) weight block — the host sends one
    //   block per pulse and never earlier (see sparse_ternary_top's port comment for the
    //   weight-SRAM refill hazard behind that rule).
    // cfg_error: sticky configuration rejection, cleared by the next start.  While it is
    //   high the LUT engine did not run (dim_I x dim_J exceeds the 128 acc rows).
    output wire          engine_wt_start,
    output wire          engine_act_start,
    output wire          engine_cfg_error,

    // Debug
    output wire [3:0]    dbg_state
);

    // ── config_regs outputs ──
    wire [15:0] cfg_dim_I, cfg_dim_J, cfg_dim_K;
    wire [31:0] cfg_wgt_addr, cfg_act_addr, cfg_out_addr;
    wire        cfg_start, cfg_soft_reset, cfg_core_mode;

    // ── soft reset ──
    // CTRL.bit1 self-clears inside config_regs, so this is a clean 1-cycle pulse.  It
    // clears the two engines without disturbing the CSR contents, which is what makes
    // it "soft": after the pulse the register file still holds dims/mode.
    wire rst_n_eng = rst_n & ~cfg_soft_reset;

    // Forward declarations: these are consumed by the config_regs / sparse_ternary_top
    // instantiations above and driven by the INT8 block below (Verilog needs the net
    // declared before the instantiation that ports it).
    wire          st_busy, int8_busy;
    wire          lut_out_tvalid, lut_out_tlast, lut_drain_busy;
    wire [511:0]  lut_out_tdata;
    wire          i8_rd_en;
    wire [7:0]    i8_rd_addr;
    wire [4095:0] i8_rd_data;
    wire [255:0]  lut_wt_i8_col;
    wire [4:0]    lut_wt_i8_col_idx;
    wire          lut_wt_i8_valid;
    wire          engines_busy = st_busy | int8_busy | lut_drain_busy;

    // ── sparse_ternary_top done (single-cycle pulse) → level for CSR ──
    wire        st_done;          // pulse
    reg         busy_sticky, done_sticky;

    // The top is only finished when BOTH engines are.  `all_done` is a level that
    // holds until the next start, so the CSR can poll STATUS.done at leisure.
    wire        all_done;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            busy_sticky <= 1'b0;
            done_sticky <= 1'b0;
        end else if (cfg_soft_reset) begin
            busy_sticky <= 1'b0;
            done_sticky <= 1'b0;
        end else begin
            if (cfg_start) begin
                busy_sticky <= 1'b1;
                done_sticky <= 1'b0;
            end
            if (all_done) begin
                busy_sticky <= 1'b0;
                done_sticky <= 1'b1;
            end
        end
    end

    // ── CSR ──
    config_regs u_cfg (
        .clk          (clk),
        .rst_n        (rst_n),
        .reg_wr_en    (csr_wr_en),
        .reg_addr     (csr_addr),
        .reg_wr_data  (csr_wr_data),
        .reg_rd_data  (csr_rd_data),
        .dim_I        (cfg_dim_I),
        .dim_J        (cfg_dim_J),
        .dim_K        (cfg_dim_K),
        .wgt_base_addr(cfg_wgt_addr),
        .act_base_addr(cfg_act_addr),
        .out_base_addr(cfg_out_addr),
        .start        (cfg_start),
        .done         (done_sticky),
        .busy         (busy_sticky | engines_busy),
        .soft_reset   (cfg_soft_reset),
        .core_mode    (cfg_core_mode)
    );

    // ── Accelerator core (unchanged) ──
    sparse_ternary_top u_st (
        .clk              (clk),
        .rst_n            (rst_n_eng),
        .s_axis_wgt_tvalid(s_axis_wgt_tvalid),
        .s_axis_wgt_tready(s_axis_wgt_tready),
        .s_axis_wgt_tdata (s_axis_wgt_tdata),
        .s_axis_wgt_tlast (s_axis_wgt_tlast),
        .s_axis_act_tvalid(s_axis_act_tvalid),
        .s_axis_act_tready(s_axis_act_tready),
        .s_axis_act_tdata (s_axis_act_tdata),
        .s_axis_act_tlast (s_axis_act_tlast),
        .m_axis_out_tvalid(lut_out_tvalid),
        .m_axis_out_tready(m_axis_out_tready),
        .m_axis_out_tdata (lut_out_tdata),
        .m_axis_out_tlast (lut_out_tlast),
        .out_drain_busy   (lut_drain_busy),
        .start            (cfg_start),
        .weight_signed    (weight_signed),
        .wt_mode          (wt_mode),
        .done             (st_done),
        .busy             (st_busy),
        .dim_I            (cfg_dim_I),
        .dim_J            (cfg_dim_J),
        .dim_K            (cfg_dim_K),
        .dma_wt_start     (engine_wt_start),
        .dma_act_start    (engine_act_start),
        .cfg_error        (engine_cfg_error),
        .dbg_state        (dbg_state),
        .wt_i8_col        (lut_wt_i8_col),
        .wt_i8_col_idx    (lut_wt_i8_col_idx),
        .wt_i8_valid      (lut_wt_i8_valid)
    );


    // ── INT8 weight load: serialise the broadcast snapshot into the mesh's columns ──
    // frozen spec: the INT8 8 Kbit block rides the LUT weight broadcast, so there is no
    // INT8 weight SRAM.  int8_engine's wgt_ld port writes one 256 b column per cycle into
    // the mesh's shared weight register file, so 32 cycles load the block — and the
    // engine must NOT start until that is done, because its preload FSM walks the same
    // register file.  `i8_wgt_ready` is therefore part of the engine's start condition.
    reg [4:0] i8_wl_col;
    reg       i8_wl_run;
    reg       i8_wgt_ready;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            i8_wl_col <= 5'd0; i8_wl_run <= 1'b0; i8_wgt_ready <= 1'b0;
        end else if (cfg_soft_reset) begin
            i8_wl_col <= 5'd0; i8_wl_run <= 1'b0; i8_wgt_ready <= 1'b0;
        end else begin
            if (cfg_start) i8_wgt_ready <= 1'b0;              // re-arm per run
            if (lut_wt_i8_valid && !i8_wl_run && !i8_wgt_ready) begin
                i8_wl_run <= 1'b1;
                i8_wl_col <= 5'd0;
            end else if (i8_wl_run) begin
                i8_wl_col <= i8_wl_col + 5'd1;
                if (i8_wl_col == 5'd31) begin
                    i8_wl_run    <= 1'b0;
                    i8_wgt_ready <= 1'b1;
                end
            end
        end
    end

    // The LUT now streams one 256 b column per cycle, already in lock-step with
    // this FSM's counter, so the weight data goes straight into the engine and the
    // 8192-bit bus (plus its 32:1 mux) is gone.
    wire [255:0] i8_wgt_data = lut_wt_i8_col;

    // ── INT8 engine: 4 x 32x32 mesh + own act/acc SRAM + 4096 b vector epilogue ──
    // Weights come from the LUT broadcast snapshot above.  The engine is held off until
    // those 32 columns are loaded; `done` then waits for both engines.
    reg  i8_go_r;
    wire i8_start;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            i8_go_r <= 1'b0;
        end else if (cfg_soft_reset) begin
            i8_go_r <= 1'b0;
        end else if (i8_start) begin
            i8_go_r <= 1'b0;                                  // consume
        end else if (cfg_start) begin
            i8_go_r <= 1'b1;
        end
    end

    assign i8_start = i8_go_r && i8_wgt_ready;

    wire int8_done;

    // The INT8 act/acc SRAM banks and the vector epilogue now live INSIDE the PE
    // macro.  Keeping them at this level put ~9,200 + 8,192 wires between the engine
    // and its memories on the parent's routing layers, which no floorplan could
    // absorb (DRT stalled at ~177k violations, all hugging the macro edges).  The
    // 80 SRAM hard macros still show up in the final merged GDS -- they are part of
    // int8_engine's own GDS, placed by that block's macro_placement.tcl.

    int8_engine u_int8 (
        .clk(clk), .rst_n(rst_n_eng),
        .start(i8_start),
        .m_tiles(4'd1),                                        // one 128-row m tile
        // Activations: the INT8 side owns its M-split act macro group and has no fill
        // source yet — the AXI-S activation stream is already consumed by the LUT
        // engine and the two have different element widths (LUT 4 b, INT8 8 b), so a
        // shared tap would need a defined repack.  Left open and flagged, not guessed.
        .act_fill_en(1'b0), .act_fill_addr(10'd0),
        .wgt_ld_en(i8_wl_run), .wgt_ld_col(lut_wt_i8_col_idx),
            .wgt_ld_data(i8_wgt_data),
        .acc_rd_en(i8_rd_en), .acc_rd_addr(i8_rd_addr), .acc_rd_data(i8_rd_data),
        .done(int8_done), .busy(int8_busy)
    );

    // busy is the OR of the engine busy signals, not just a start/done latch: a CSR
    // read of STATUS.busy now reflects what the hardware is actually doing.
    reg lut_done_seen, int8_done_seen;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            lut_done_seen  <= 1'b0;
            int8_done_seen <= 1'b0;
        end else if (cfg_soft_reset) begin
            lut_done_seen  <= 1'b0;
            int8_done_seen <= 1'b0;
        end else if (cfg_start) begin
            lut_done_seen  <= 1'b0;
            int8_done_seen <= 1'b0;
        end else begin
            if (st_done)   lut_done_seen  <= 1'b1;
            if (int8_done) int8_done_seen <= 1'b1;
        end
    end
    assign all_done = lut_done_seen && int8_done_seen;

    // ── Result read-out: ONE output port, time-multiplexed between the two engines ──
    //
    // The LUT and INT8 sides own separate acc SRAMs (128 x M256 @ 8192 b and 64 x M256
    // @ 4096 b) and are written independently, but there is a single m_axis_out port, so
    // the two drains share it.  The LUT drain has priority: it is already running by the
    // time the INT8 engine finishes, and its word is twice as wide.
    localparam I8_ROWS   = 32;          // m_tiles(1) x K(32) acc rows written per run
    localparam I8_SLICES = 4096/512;    // 8 x 512 b beats per 4096 b acc word

    localparam IS_IDLE = 2'd0, IS_ADDR = 2'd1, IS_WAIT = 2'd2, IS_BEAT = 2'd3;

    reg  i8_drain_req;                  // int8_done is a pulse; latch the request
    reg [1:0]  is;
    reg [7:0]  i8_row;
    reg [3:0]  i8_cnt;
    reg [4095:0] i8_word_r;

    wire i8_grant = ~lut_out_tvalid;    // LUT drain owns the port when it is valid

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            i8_drain_req <= 1'b0; is <= IS_IDLE;
            i8_row <= 8'd0; i8_cnt <= 4'd0;
            i8_word_r <= 4096'd0;
        end else if (cfg_soft_reset) begin
            i8_drain_req <= 1'b0; is <= IS_IDLE;
        end else begin
            if (int8_done) i8_drain_req <= 1'b1;

            case (is)
                IS_IDLE: if (i8_drain_req) begin
                             i8_row <= 8'd0; i8_cnt <= 4'd0;
                             is <= IS_ADDR;
                         end
                IS_ADDR: is <= IS_WAIT;
                IS_WAIT: begin i8_word_r <= i8_rd_data; i8_cnt <= 4'd0; is <= IS_BEAT; end
                IS_BEAT: if (i8_grant && m_axis_out_tready) begin
                             if (i8_cnt == I8_SLICES-1) begin
                                 if (i8_row == I8_ROWS-1) begin
                                     is <= IS_IDLE;
                                     i8_drain_req <= 1'b0;
                                 end else begin
                                     i8_row <= i8_row + 8'd1;
                                     is <= IS_ADDR;
                                 end
                             end else begin
                                 i8_cnt <= i8_cnt + 4'd1;
                             end
                         end
                default: is <= IS_IDLE;
            endcase
        end
    end

    // The engine deselects its acc macro when acc_rd_en is low (csb=1 -> dout = X), so
    // the read must stay enabled for BOTH cycles: the address is combinational and the
    // capture at the end of IS_WAIT picks up the value the macro drove on its falling edge.
    assign i8_rd_en   = (is == IS_ADDR) || (is == IS_WAIT);
    assign i8_rd_addr = i8_row;       // combinational: see the note above
    wire i8_out_tvalid = (is == IS_BEAT) && i8_grant;

    // ── output port mux: LUT drain wins, INT8 drain takes what is left ──
    assign m_axis_out_tvalid = lut_out_tvalid | i8_out_tvalid;
    assign m_axis_out_tdata  = lut_out_tvalid ? lut_out_tdata
                                              : i8_word_r[i8_cnt*512 +: 512];
    assign m_axis_out_tlast  = lut_out_tvalid ? lut_out_tlast
                                              : (i8_cnt == I8_SLICES-1) && (i8_row == I8_ROWS-1);

endmodule
