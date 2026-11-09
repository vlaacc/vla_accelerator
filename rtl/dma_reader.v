// dma_reader.v — AXI4-Stream slave to internal buffer
// Receives streaming data from host and writes into internal SRAM buffer.

`include "sparsenn_params.vh"

module dma_reader #(
    parameter DATA_WIDTH = `AXI_DATA_WIDTH,
    parameter ADDR_WIDTH = `AXI_ADDR_WIDTH,
    parameter BUF_DEPTH  = `WEIGHT_BUF_DEPTH
) (
    input  wire                     clk,
    input  wire                     rst_n,

    // AXI4-Stream slave interface
    input  wire                     s_axis_tvalid,
    output reg                      s_axis_tready,
    input  wire [DATA_WIDTH-1:0]    s_axis_tdata,
    input  wire                     s_axis_tlast,

    // Internal buffer write interface
    output reg                      buf_wr_en,
    output reg  [ADDR_WIDTH-1:0]    buf_wr_addr,
    output reg  [DATA_WIDTH-1:0]    buf_wr_data,

    // Control
    input  wire                     start,
    output reg                      done,
    output reg  [15:0]              words_received
);

    // ── State ──
    localparam S_IDLE   = 2'd0;
    localparam S_ACTIVE = 2'd1;
    localparam S_DONE   = 2'd2;

    reg [1:0] state;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE;
            s_axis_tready <= 0;
            buf_wr_en <= 0;
            buf_wr_addr <= 0;
            buf_wr_data <= 0;
            done <= 0;
            words_received <= 0;
        end else begin
            case (state)
                S_IDLE: begin
                    done <= 0;
                    words_received <= 0;
                    buf_wr_en <= 0;
                    s_axis_tready <= 0;
                    if (start)
                        state <= S_ACTIVE;
                end

                S_ACTIVE: begin
                    s_axis_tready <= 1;

                    if (s_axis_tvalid && s_axis_tready) begin
                        buf_wr_en <= 1;
                        buf_wr_data <= s_axis_tdata;
                        buf_wr_addr <= words_received;
                        words_received <= words_received + 1;

                        if (s_axis_tlast) begin
                            state <= S_DONE;
                            done <= 1;
                        end
                    end else begin
                        buf_wr_en <= 0;
                    end
                end

                S_DONE: begin
                    done <= 0;
                    s_axis_tready <= 0;
                    if (!start)
                        state <= S_IDLE;
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
