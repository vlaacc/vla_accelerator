// sparsenn_top.v — SparseNN top-level with FP16 + FP16T3 dual-core
// Phase 1: independent cores with CSR mode-bit mutual exclusion.
// Phase 2: shared SRAM with concurrent dual-core operation.

`include "sparsenn_params.vh"

module sparsenn_top #(
    parameter N = `N_PE_ROWS,
    parameter M = `M_PE_COLS
) (
    input  wire                     clk,
    input  wire                     rst_n,

    // ═══ FP16 core AXI4-Stream ═══
    input  wire                     s_axis_wgt_tvalid,
    output wire                     s_axis_wgt_tready,
    input  wire [`AXI_DATA_WIDTH-1:0] s_axis_wgt_tdata,
    input  wire                     s_axis_wgt_tlast,

    input  wire                     s_axis_act_tvalid,
    output wire                     s_axis_act_tready,
    input  wire [`AXI_DATA_WIDTH-1:0] s_axis_act_tdata,
    input  wire                     s_axis_act_tlast,

    output wire                     m_axis_out_tvalid,
    input  wire                     m_axis_out_tready,
    output wire [`AXI_DATA_WIDTH-1:0] m_axis_out_tdata,
    output wire                     m_axis_out_tlast,

    // ═══ FP16T3 core AXI4-Stream ═══
    input  wire                     s_axis_wgt_tvalid_fp16t3,
    output wire                     s_axis_wgt_tready_fp16t3,
    input  wire [`AXI_DATA_WIDTH-1:0] s_axis_wgt_tdata_fp16t3,
    input  wire                     s_axis_wgt_tlast_fp16t3,

    input  wire                     s_axis_act_tvalid_fp16t3,
    output wire                     s_axis_act_tready_fp16t3,
    input  wire [`AXI_DATA_WIDTH-1:0] s_axis_act_tdata_fp16t3,
    input  wire                     s_axis_act_tlast_fp16t3,

    output wire                     m_axis_out_tvalid_fp16t3,
    input  wire                     m_axis_out_tready_fp16t3,
    output wire [`AXI_DATA_WIDTH-1:0] m_axis_out_tdata_fp16t3,
    output wire                     m_axis_out_tlast_fp16t3,

    // ═══ AXI4-Lite CSR ═══
    input  wire                     csr_wr_en,
    input  wire [7:0]               csr_addr,
    input  wire [31:0]              csr_wr_data,
    output wire [31:0]              csr_rd_data,

    // ═══ Status ═══
    output wire                     done,
    output wire                     busy,

    // Debug
    output wire [3:0]               dbg_state,
    output wire [3:0]               dbg_state_fp16t3
);

    // ── Config signals ──
    wire [15:0] cfg_dim_I, cfg_dim_J, cfg_dim_K;
    wire [31:0] cfg_wgt_addr, cfg_act_addr, cfg_out_addr;
    wire       cfg_start, cfg_soft_reset, cfg_core_mode;

    // ── FP16 core internal ──
    wire       f16_sa_weight_load, f16_sa_accum_clear, f16_sa_stall;
    wire [15:0] f16_sa_act_addr, f16_sa_wt_addr;
    wire       f16_acc_wr_en, f16_acc_rd_en, f16_acc_accum_en;
    wire [15:0] f16_acc_wr_addr, f16_acc_rd_addr;
    wire       f16_dma_act_start, f16_dma_wt_start, f16_dma_act_done, f16_dma_wt_done;
    wire       f16_ctrl_done, f16_ctrl_busy;
    wire       f16_wbuf_wr_bank, f16_wbuf_rd_bank;
    wire       f16_abuf_wr_bank, f16_abuf_rd_bank;

    // ── FP16T3 core internal ──
    wire       t3_start, t3_done, t3_busy;

    // ── Mutex: gate start based on core_mode ──
    assign t3_start = cfg_start && cfg_core_mode;
    wire   f16_start = cfg_start && ~cfg_core_mode;

    // ═══ Config Registers ═══
    config_regs cfg (
        .clk(clk), .rst_n(rst_n),
        .reg_wr_en(csr_wr_en), .reg_addr(csr_addr),
        .reg_wr_data(csr_wr_data), .reg_rd_data(csr_rd_data),
        .dim_I(cfg_dim_I), .dim_J(cfg_dim_J), .dim_K(cfg_dim_K),
        .wgt_base_addr(cfg_wgt_addr), .act_base_addr(cfg_act_addr),
        .out_base_addr(cfg_out_addr),
        .start(cfg_start), .done(ctrl_done_final), .busy(ctrl_busy_final),
        .soft_reset(cfg_soft_reset), .core_mode(cfg_core_mode)
    );

    // ═══ FP16T3 Core (independent buffers, Phase 1) ═══
    fp16t3_core #(.N(N), .M(M)) core_fp16t3 (
        .clk(clk), .rst_n(rst_n),

        .s_axis_wgt_tvalid(s_axis_wgt_tvalid_fp16t3),
        .s_axis_wgt_tready(s_axis_wgt_tready_fp16t3),
        .s_axis_wgt_tdata(s_axis_wgt_tdata_fp16t3),
        .s_axis_wgt_tlast(s_axis_wgt_tlast_fp16t3),

        .s_axis_act_tvalid(s_axis_act_tvalid_fp16t3),
        .s_axis_act_tready(s_axis_act_tready_fp16t3),
        .s_axis_act_tdata(s_axis_act_tdata_fp16t3),
        .s_axis_act_tlast(s_axis_act_tlast_fp16t3),

        .m_axis_out_tvalid(m_axis_out_tvalid_fp16t3),
        .m_axis_out_tready(m_axis_out_tready_fp16t3),
        .m_axis_out_tdata(m_axis_out_tdata_fp16t3),
        .m_axis_out_tlast(m_axis_out_tlast_fp16t3),

        .start(t3_start), .done(t3_done), .busy(t3_busy),
        .dim_I(cfg_dim_I), .dim_J(cfg_dim_J), .dim_K(cfg_dim_K),

        .dbg_state(dbg_state_fp16t3)
    );

    // ═══ FP16 Core (existing) ═══
    // DMA
    wire        f16_dma_wt_buf_wr_en, f16_dma_act_buf_wr_en;
    wire [`AXI_ADDR_WIDTH-1:0] f16_dma_wt_buf_wr_addr, f16_dma_act_buf_wr_addr;
    wire [`AXI_DATA_WIDTH-1:0] f16_dma_wt_buf_wr_data, f16_dma_act_buf_wr_data;

    wire [M*`FP16_WIDTH-1:0] f16_wt_buf_rd_data, f16_act_buf_rd_data;
    wire [N*`FP16_WIDTH-1:0] f16_sa_activations, f16_sa_weights;
    wire [M*`FP32_WIDTH-1:0] f16_sa_ps_in, f16_sa_ps_out, f16_acc_rd_data;
    wire f16_wt_buf_rd_en;
    wire f16_dma_out_buf_rd_en;
    wire [`AXI_ADDR_WIDTH-1:0] f16_dma_out_buf_rd_addr;

    // DMA readers (FP16)
    dma_reader #(.DATA_WIDTH(`AXI_DATA_WIDTH), .ADDR_WIDTH(`AXI_ADDR_WIDTH),
                 .BUF_DEPTH(`WEIGHT_BUF_DEPTH)) dma_wgt (
        .clk(clk), .rst_n(rst_n),
        .s_axis_tvalid(s_axis_wgt_tvalid), .s_axis_tready(s_axis_wgt_tready),
        .s_axis_tdata(s_axis_wgt_tdata), .s_axis_tlast(s_axis_wgt_tlast),
        .buf_wr_en(f16_dma_wt_buf_wr_en), .buf_wr_addr(f16_dma_wt_buf_wr_addr),
        .buf_wr_data(f16_dma_wt_buf_wr_data),
        .start(f16_dma_wt_start), .done(f16_dma_wt_done), .words_received()
    );

    dma_reader #(.DATA_WIDTH(`AXI_DATA_WIDTH), .ADDR_WIDTH(`AXI_ADDR_WIDTH),
                 .BUF_DEPTH(`ACT_BUF_DEPTH)) dma_act (
        .clk(clk), .rst_n(rst_n),
        .s_axis_tvalid(s_axis_act_tvalid), .s_axis_tready(s_axis_act_tready),
        .s_axis_tdata(s_axis_act_tdata), .s_axis_tlast(s_axis_act_tlast),
        .buf_wr_en(f16_dma_act_buf_wr_en), .buf_wr_addr(f16_dma_act_buf_wr_addr),
        .buf_wr_data(f16_dma_act_buf_wr_data),
        .start(f16_dma_act_start), .done(f16_dma_act_done), .words_received()
    );

    // ═══ Weight Buffer (DFF or SRAM) ═══
    // USE_SRAM=0: DFF-based double-buffered ping/pong (simulation)
    // USE_SRAM=1: OpenRAM SRAM macro (physical layout)
    wire [511:0] wt_buf_rd_data_sram;
    wire [255:0] wt_buf_rd_data;

    generate
        if (`USE_SRAM == 0) begin : gen_wt_buf_dff
            weight_buffer #(.DEPTH(`WEIGHT_BUF_DEPTH), .WIDTH(`AXI_DATA_WIDTH)) wt_buf (
                .clk(clk), .rst_n(rst_n),
                .wr_en(f16_dma_wt_buf_wr_en),
                .wr_addr(f16_dma_wt_buf_wr_addr[$clog2(`WEIGHT_BUF_DEPTH)-1:0]),
                .wr_data(f16_dma_wt_buf_wr_data),
                .wr_bank_sel(f16_wbuf_wr_bank),
                .rd_en(f16_wt_buf_rd_en),
                .rd_addr(f16_sa_wt_addr[$clog2(`WEIGHT_BUF_DEPTH)-1:0]),
                .rd_data(wt_buf_rd_data),
                .rd_bank_sel(f16_wbuf_rd_bank),
                .swap(1'b0), .buffer_ready()
            );
        end else begin : gen_wt_buf_sram
            // SRAM: 2×(2048×512b), WIDTH fixed at 512
            // DMA data (256b) is duplicated to fill 512b (matches fp16t3_core_sram)
            // wr_addr=latched DMA addr (registered), rd_addr=SA addr — time-multiplexed
            wire [$clog2(2048)-1:0] wt_sram_wr_addr;
            reg  [$clog2(2048)-1:0] wt_sram_wr_addr_reg;
            always @(posedge clk) begin
                if (f16_dma_wt_buf_wr_en)
                    wt_sram_wr_addr_reg <= f16_dma_wt_buf_wr_addr[$clog2(2048)-1:0];
            end
            assign wt_sram_wr_addr = wt_sram_wr_addr_reg;

            sram_weight_buffer #(.DEPTH(2048), .WIDTH(512)) wt_buf (
                .clk(clk), .rst_n(rst_n),
                .wr_en(f16_dma_wt_buf_wr_en),
                .wr_addr(wt_sram_wr_addr),
                .wr_data({f16_dma_wt_buf_wr_data, f16_dma_wt_buf_wr_data}),
                .wr_bank_sel(f16_wbuf_wr_bank),
                .rd_en(f16_wt_buf_rd_en),
                .rd_addr(f16_sa_wt_addr[$clog2(2048)-1:0]),
                .rd_data(wt_buf_rd_data_sram),
                .rd_bank_sel(f16_wbuf_rd_bank),
                .swap(1'b0), .buffer_ready()
            );
            assign wt_buf_rd_data = wt_buf_rd_data_sram[N*`FP16_WIDTH-1:0];
        end
    endgenerate

    // ═══ Activation Buffer (DFF or SRAM) ═══
    generate
        if (`USE_SRAM == 0) begin : gen_act_buf_dff
            activation_buffer #(.DEPTH(`ACT_BUF_DEPTH), .WIDTH(`AXI_DATA_WIDTH)) act_buf (
                .clk(clk), .rst_n(rst_n),
                .wr_en(f16_dma_act_buf_wr_en),
                .wr_addr(f16_dma_act_buf_wr_addr[$clog2(`ACT_BUF_DEPTH)-1:0]),
                .wr_data(f16_dma_act_buf_wr_data),
                .wr_bank_sel(f16_abuf_wr_bank),
                .rd_en(1'b1),
                .rd_addr(f16_sa_act_addr[$clog2(`ACT_BUF_DEPTH)-1:0]),
                .rd_data(f16_act_buf_rd_data),
                .rd_bank_sel(f16_abuf_rd_bank),
                .swap(1'b0), .buffer_ready()
            );
        end else begin : gen_act_buf_sram
            // SRAM: 2×(2048×256b)
            // wr_addr=latched DMA addr (registered), rd_addr=SA addr — time-multiplexed
            wire [$clog2(2048)-1:0] act_sram_wr_addr;
            reg  [$clog2(2048)-1:0] act_sram_wr_addr_reg;
            always @(posedge clk) begin
                if (f16_dma_act_buf_wr_en)
                    act_sram_wr_addr_reg <= f16_dma_act_buf_wr_addr[$clog2(2048)-1:0];
            end
            assign act_sram_wr_addr = act_sram_wr_addr_reg;

            sram_activation_buffer #(.DEPTH(2048), .WIDTH(256)) act_buf (
                .clk(clk), .rst_n(rst_n),
                .wr_en(f16_dma_act_buf_wr_en),
                .wr_addr(act_sram_wr_addr),
                .wr_data(f16_dma_act_buf_wr_data),
                .wr_bank_sel(f16_abuf_wr_bank),
                .rd_en(1'b1),
                .rd_addr(f16_sa_act_addr[$clog2(2048)-1:0]),
                .rd_data(f16_act_buf_rd_data),
                .rd_bank_sel(f16_abuf_rd_bank),
                .swap(1'b0), .buffer_ready()
            );
        end
    endgenerate

    // Systolic array (FP16)
    assign f16_sa_activations = f16_act_buf_rd_data[N*`FP16_WIDTH-1:0];
    assign f16_sa_weights     = f16_wt_buf_rd_data[N*`FP16_WIDTH-1:0];
    assign f16_wt_buf_rd_en   = f16_sa_weight_load;
    assign f16_sa_ps_in       = f16_acc_rd_data;

    systolic_array #(.N(N), .M(M)) sa (
        .clk(clk), .rst_n(rst_n),
        .weight_load(f16_sa_weight_load), .accum_clear(f16_sa_accum_clear),
        .stall(f16_sa_stall),
        .activations_in(f16_sa_activations),
        .weights_in(f16_sa_weights),
        .partial_sums_in(f16_sa_ps_in),
        .partial_sums_out(f16_sa_ps_out)
    );

    // ═══ Accumulator Bank (DFF or SRAM) ═══
    generate
        if (`USE_SRAM == 0) begin : gen_acc_bank_dff
            accumulator_bank #(.DEPTH(`ACC_BUF_DEPTH), .WIDTH(M * `FP32_WIDTH)) acc_bank (
                .clk(clk), .rst_n(rst_n),
                .wr_en(f16_acc_wr_en),
                .wr_addr(f16_acc_wr_addr[$clog2(`ACC_BUF_DEPTH)-1:0]),
                .wr_data(f16_sa_ps_out),
                .rd_en(f16_acc_rd_en),
                .rd_addr(f16_acc_rd_addr[$clog2(`ACC_BUF_DEPTH)-1:0]),
                .rd_data(f16_acc_rd_data),
                .accum_en(f16_acc_accum_en),
                .ready()
            );
        end else begin : gen_acc_bank_sram
            // SRAM: 1×(1024×512b) — exact match for ACC_BUF_DEPTH=1024
            sram_accumulator_bank #(.DEPTH(1024), .WIDTH(512)) acc_bank (
                .clk(clk), .rst_n(rst_n),
                .wr_en(f16_acc_wr_en),
                .wr_addr(f16_acc_wr_addr[$clog2(`ACC_BUF_DEPTH)-1:0]),
                .wr_data(f16_sa_ps_out),
                .rd_en(f16_acc_rd_en),
                .rd_addr(f16_acc_rd_addr[$clog2(`ACC_BUF_DEPTH)-1:0]),
                .rd_data(f16_acc_rd_data),
                .accum_en(f16_acc_accum_en),
                .ready()
            );
        end
    endgenerate

    // Controller (FP16)
    controller #(.N(N), .M(M)) ctrl (
        .clk(clk), .rst_n(rst_n),
        .start(f16_start), .done(f16_ctrl_done), .busy(f16_ctrl_busy),
        .dim_I(cfg_dim_I), .dim_J(cfg_dim_J), .dim_K(cfg_dim_K),
        .sa_weight_load(f16_sa_weight_load), .sa_accum_clear(f16_sa_accum_clear),
        .sa_stall(f16_sa_stall), .sa_act_addr(f16_sa_act_addr), .sa_wt_addr(f16_sa_wt_addr),
        .acc_wr_en(f16_acc_wr_en), .acc_wr_addr(f16_acc_wr_addr),
        .acc_rd_en(f16_acc_rd_en), .acc_rd_addr(f16_acc_rd_addr),
        .acc_accum_en(f16_acc_accum_en),
        .dma_act_start(f16_dma_act_start), .dma_wt_start(f16_dma_wt_start),
        .dma_act_done(f16_dma_act_done), .dma_wt_done(f16_dma_wt_done),
        .tile_i(), .tile_j(), .tile_k(), .total_tiles(),
        .wbuf_wr_bank(f16_wbuf_wr_bank), .wbuf_rd_bank(f16_wbuf_rd_bank),
        .abuf_wr_bank(f16_abuf_wr_bank), .abuf_rd_bank(f16_abuf_rd_bank),
        .dbg_state(dbg_state)
    );

    // DMA writer (FP16)
    wire [255:0] f16_acc_out_word;
    assign f16_acc_out_word = f16_acc_rd_data;

    dma_writer #(.DATA_WIDTH(`AXI_DATA_WIDTH), .ADDR_WIDTH(`AXI_ADDR_WIDTH),
                 .BUF_DEPTH(`ACC_BUF_DEPTH)) dma_out (
        .clk(clk), .rst_n(rst_n),
        .m_axis_tvalid(m_axis_out_tvalid), .m_axis_tready(m_axis_out_tready),
        .m_axis_tdata(m_axis_out_tdata), .m_axis_tlast(m_axis_out_tlast),
        .buf_rd_en(f16_dma_out_buf_rd_en), .buf_rd_addr(f16_dma_out_buf_rd_addr),
        .buf_rd_data(f16_acc_out_word),
        .buf_rd_valid(f16_acc_rd_en),
        .start(f16_ctrl_done), .total_words(cfg_dim_I * cfg_dim_J),
        .done()
    );

    // ── Global done/busy: OR of both cores ──
    wire ctrl_done_final = cfg_core_mode ? t3_done : f16_ctrl_done;
    wire ctrl_busy_final = cfg_core_mode ? t3_busy : f16_ctrl_busy;

    assign done = ctrl_done_final;
    assign busy = ctrl_busy_final;

endmodule
