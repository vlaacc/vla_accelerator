// config_regs.v — Memory-mapped control/status register file
// Provides the host interface for configuring matrix dimensions,
// base addresses, and controlling the SparseNN accelerator.

module config_regs (
    input  wire         clk,
    input  wire         rst_n,

    // AXI4-Lite (simplified) register interface
    input  wire         reg_wr_en,
    input  wire [7:0]   reg_addr,
    input  wire [31:0]  reg_wr_data,
    output wire [31:0]  reg_rd_data,

    // Configuration outputs
    output reg  [15:0]  dim_I,              // Matrix A rows
    output reg  [15:0]  dim_J,              // Matrix B columns
    output reg  [15:0]  dim_K,              // Common dimension
    output reg  [31:0]  wgt_base_addr,     // Weight buffer base address
    output reg  [31:0]  act_base_addr,     // Activation buffer base address
    output reg  [31:0]  out_base_addr,     // Output buffer base address
    output reg          start,             // Start computation
    input  wire         done,              // Computation complete
    input  wire         busy,              // Accelerator busy
    output reg          soft_reset,        // Software reset
    output reg          core_mode          // 0=FP16, 1=FP16T3
);

    // ── Register map ──
    localparam REG_DIM_I        = 8'h00;   // R/W
    localparam REG_DIM_J        = 8'h04;   // R/W
    localparam REG_DIM_K        = 8'h08;   // R/W
    localparam REG_WGT_ADDR     = 8'h0C;   // R/W
    localparam REG_ACT_ADDR     = 8'h10;   // R/W
    localparam REG_OUT_ADDR     = 8'h14;   // R/W
    localparam REG_CTRL         = 8'h20;   // R/W: bit0=start, bit1=soft_reset, bit2=core_mode
    localparam REG_STATUS       = 8'h24;   // R: bit0=done, bit1=busy
    localparam REG_TILES        = 8'h28;   // R: tile count

    reg [31:0] reg_file [0:31];

    // ── Register write ──
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            dim_I <= 0; dim_J <= 0; dim_K <= 0;
            wgt_base_addr <= 0; act_base_addr <= 0; out_base_addr <= 0;
            start <= 0; soft_reset <= 0; core_mode <= 0;
        end else begin
            if (reg_wr_en) begin
                case (reg_addr[7:2])  // Word-aligned (divide by 4)
                    REG_DIM_I[7:2]:    dim_I <= reg_wr_data[15:0];
                    REG_DIM_J[7:2]:    dim_J <= reg_wr_data[15:0];
                    REG_DIM_K[7:2]:    dim_K <= reg_wr_data[15:0];
                    REG_WGT_ADDR[7:2]: wgt_base_addr <= reg_wr_data;
                    REG_ACT_ADDR[7:2]: act_base_addr <= reg_wr_data;
                    REG_OUT_ADDR[7:2]: out_base_addr <= reg_wr_data;
                    REG_CTRL[7:2]: begin
                        start <= reg_wr_data[0];
                        soft_reset <= reg_wr_data[1];
                        core_mode <= reg_wr_data[2];
                    end
                    default: ;
                endcase
            end else begin
                // Self-clearing start
                if (start && busy)
                    start <= 0;
                if (soft_reset)
                    soft_reset <= 0;
            end
        end
    end

    // ── Register read ──
    reg [31:0] rd_data_reg;
    assign reg_rd_data = rd_data_reg;

    always @(*) begin
        case (reg_addr[7:2])
            REG_DIM_I[7:2]:    rd_data_reg = {16'd0, dim_I};
            REG_DIM_J[7:2]:    rd_data_reg = {16'd0, dim_J};
            REG_DIM_K[7:2]:    rd_data_reg = {16'd0, dim_K};
            REG_WGT_ADDR[7:2]: rd_data_reg = wgt_base_addr;
            REG_ACT_ADDR[7:2]: rd_data_reg = act_base_addr;
            REG_OUT_ADDR[7:2]: rd_data_reg = out_base_addr;
            REG_CTRL[7:2]:     rd_data_reg = {29'd0, core_mode, soft_reset, start};
            REG_STATUS[7:2]:   rd_data_reg = {30'd0, busy, done};
            REG_TILES[7:2]:    rd_data_reg = 32'd0;  // Connected by top-level
            default:           rd_data_reg = 32'd0;
        endcase
    end

endmodule
