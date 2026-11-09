// fp16t3_core_sram.v — FP16T3 compute core with OpenRAM SRAM buffers
// Replaces DFF buffers with blackboxed OpenRAM macro wrappers.
// 28 SRAM macros total: 16 weight + 8 activation + 4 accumulator.

`include "sparsenn_params.vh"

module fp16t3_core_sram #(
    parameter N = `N_PE_ROWS,
    parameter M = `M_PE_COLS
) (
    input  wire                     clk,
    input  wire                     rst_n,

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

    input  wire                     start,
    output wire                     done,
    output wire                     busy,
    input  wire [15:0]              dim_I,
    input  wire [15:0]              dim_J,
    input  wire [15:0]              dim_K,

    output wire [3:0]               dbg_state
);

    localparam WBUF_WIDTH = N * M * 2;  // 512b
    localparam SRAM_DEPTH_WGT = 2048;
    localparam SRAM_DEPTH_ACT = 2048;
    localparam SRAM_DEPTH_ACC = 1024;

    wire        sa_weight_load, sa_accum_clear, sa_stall;
    wire [15:0] sa_act_addr, sa_wt_addr;
    wire        acc_wr_en, acc_rd_en, acc_accum_en;
    wire [15:0] acc_wr_addr, acc_rd_addr;
    wire        dma_act_start, dma_wt_start, dma_act_done, dma_wt_done;
    wire        wbuf_wr_bank, wbuf_rd_bank;
    wire        abuf_wr_bank, abuf_rd_bank;

    wire        dma_wt_buf_wr_en, dma_act_buf_wr_en;
    wire [`AXI_ADDR_WIDTH-1:0] dma_wt_buf_wr_addr, dma_act_buf_wr_addr;
    wire [`AXI_DATA_WIDTH-1:0] dma_wt_buf_wr_data, dma_act_buf_wr_data;

    wire [WBUF_WIDTH-1:0]      wt_buf_rd_data;
    wire [M*`FP16_WIDTH-1:0]   act_buf_rd_data;

    wire [N*`FP16_WIDTH-1:0]   sa_activations;
    wire [M*`FP32_WIDTH-1:0]   sa_ps_in;
    wire [M*`FP32_WIDTH-1:0]   sa_ps_out;
    wire [M*`FP32_WIDTH-1:0]   acc_rd_data;

    wire                       dma_out_buf_rd_en;
    wire [`AXI_ADDR_WIDTH-1:0] dma_out_buf_rd_addr;

    // ═══ DMA Readers ═══
    dma_reader #(.DATA_WIDTH(`AXI_DATA_WIDTH), .ADDR_WIDTH(`AXI_ADDR_WIDTH),
                 .BUF_DEPTH(`WEIGHT_BUF_DEPTH)) dma_wgt (
        .clk(clk), .rst_n(rst_n),
        .s_axis_tvalid(s_axis_wgt_tvalid), .s_axis_tready(s_axis_wgt_tready),
        .s_axis_tdata(s_axis_wgt_tdata), .s_axis_tlast(s_axis_wgt_tlast),
        .buf_wr_en(dma_wt_buf_wr_en), .buf_wr_addr(dma_wt_buf_wr_addr),
        .buf_wr_data(dma_wt_buf_wr_data),
        .start(dma_wt_start), .done(dma_wt_done), .words_received()
    );

    dma_reader #(.DATA_WIDTH(`AXI_DATA_WIDTH), .ADDR_WIDTH(`AXI_ADDR_WIDTH),
                 .BUF_DEPTH(`ACT_BUF_DEPTH)) dma_act (
        .clk(clk), .rst_n(rst_n),
        .s_axis_tvalid(s_axis_act_tvalid), .s_axis_tready(s_axis_act_tready),
        .s_axis_tdata(s_axis_act_tdata), .s_axis_tlast(s_axis_act_tlast),
        .buf_wr_en(dma_act_buf_wr_en), .buf_wr_addr(dma_act_buf_wr_addr),
        .buf_wr_data(dma_act_buf_wr_data),
        .start(dma_act_start), .done(dma_act_done), .words_received()
    );

    // ═══ Weight Buffer SRAM (double-buffered, 2048×512b per bank) ═══
    // DMA writes 256b/beat; 2 beats assemble into 512b row (handled by controller addr)
    sram_weight_buffer_fp16t3 #(.DEPTH(SRAM_DEPTH_WGT), .WIDTH(WBUF_WIDTH)) wt_buf (
        .clk(clk), .rst_n(rst_n),
        .wr_en(dma_wt_buf_wr_en),
        .wr_addr(dma_wt_buf_wr_addr[$clog2(SRAM_DEPTH_WGT)-1:0]),
        .wr_data({dma_wt_buf_wr_data, dma_wt_buf_wr_data}),
        .wr_bank_sel(wbuf_wr_bank),
        .rd_en(sa_weight_load),
        .rd_addr(sa_wt_addr[$clog2(SRAM_DEPTH_WGT)-1:0]),
        .rd_data(wt_buf_rd_data),
        .rd_bank_sel(wbuf_rd_bank),
        .swap(1'b0), .buffer_ready()
    );

    // ═══ Activation Buffer SRAM (double-buffered, 2048×256b per bank) ═══
    sram_act_buffer_fp16t3 #(.DEPTH(SRAM_DEPTH_ACT), .WIDTH(`AXI_DATA_WIDTH)) act_buf (
        .clk(clk), .rst_n(rst_n),
        .wr_en(dma_act_buf_wr_en),
        .wr_addr(dma_act_buf_wr_addr[$clog2(SRAM_DEPTH_ACT)-1:0]),
        .wr_data(dma_act_buf_wr_data),
        .wr_bank_sel(abuf_wr_bank),
        .rd_en(1'b1),
        .rd_addr(sa_act_addr[$clog2(SRAM_DEPTH_ACT)-1:0]),
        .rd_data(act_buf_rd_data),
        .rd_bank_sel(abuf_rd_bank),
        .swap(1'b0), .buffer_ready()
    );

    // ═══ Systolic Array FP16T3 ═══
    assign sa_activations = act_buf_rd_data[N*`FP16_WIDTH-1:0];
    assign sa_ps_in       = acc_rd_data;

    systolic_array_fp16t3 #(.N(N), .M(M)) sa (
        .clk(clk), .rst_n(rst_n),
        .weight_load(sa_weight_load), .accum_clear(sa_accum_clear),
        .stall(sa_stall),
        .activations_in(sa_activations),
        .weights_in(wt_buf_rd_data[WBUF_WIDTH-1:0]),
        .partial_sums_in(sa_ps_in),
        .partial_sums_out(sa_ps_out)
    );

    // ═══ Accumulator Bank SRAM (1024×512b) ═══
    sram_acc_buffer_fp16t3 #(.DEPTH(SRAM_DEPTH_ACC), .WIDTH(M * `FP32_WIDTH)) acc_bank (
        .clk(clk), .rst_n(rst_n),
        .wr_en(acc_wr_en),
        .wr_addr(acc_wr_addr[$clog2(SRAM_DEPTH_ACC)-1:0]),
        .wr_data(sa_ps_out),
        .rd_en(acc_rd_en),
        .rd_addr(acc_rd_addr[$clog2(SRAM_DEPTH_ACC)-1:0]),
        .rd_data(acc_rd_data),
        .accum_en(acc_accum_en),
        .ready()
    );

    // ═══ Controller FP16T3 ═══
    controller_fp16t3 #(.N(N), .M(M)) ctrl (
        .clk(clk), .rst_n(rst_n),
        .start(start), .done(done), .busy(busy),
        .dim_I(dim_I), .dim_J(dim_J), .dim_K(dim_K),
        .sa_weight_load(sa_weight_load), .sa_accum_clear(sa_accum_clear),
        .sa_stall(sa_stall), .sa_act_addr(sa_act_addr), .sa_wt_addr(sa_wt_addr),
        .acc_wr_en(acc_wr_en), .acc_wr_addr(acc_wr_addr),
        .acc_rd_en(acc_rd_en), .acc_rd_addr(acc_rd_addr),
        .acc_accum_en(acc_accum_en),
        .dma_act_start(dma_act_start), .dma_wt_start(dma_wt_start),
        .dma_act_done(dma_act_done), .dma_wt_done(dma_wt_done),
        .wbuf_wr_bank(wbuf_wr_bank), .wbuf_rd_bank(wbuf_rd_bank),
        .abuf_wr_bank(abuf_wr_bank), .abuf_rd_bank(abuf_rd_bank),
        .dbg_state(dbg_state)
    );

    // ═══ DMA Writer ═══
    wire [255:0] acc_out_word;
    assign acc_out_word = acc_rd_data;

    dma_writer #(.DATA_WIDTH(`AXI_DATA_WIDTH), .ADDR_WIDTH(`AXI_ADDR_WIDTH),
                 .BUF_DEPTH(`ACC_BUF_DEPTH)) dma_out (
        .clk(clk), .rst_n(rst_n),
        .m_axis_tvalid(m_axis_out_tvalid), .m_axis_tready(m_axis_out_tready),
        .m_axis_tdata(m_axis_out_tdata), .m_axis_tlast(m_axis_out_tlast),
        .buf_rd_en(dma_out_buf_rd_en), .buf_rd_addr(dma_out_buf_rd_addr),
        .buf_rd_data(acc_out_word),
        .buf_rd_valid(acc_rd_en),
        .start(done), .total_words(dim_I * dim_J),
        .done()
    );

endmodule
