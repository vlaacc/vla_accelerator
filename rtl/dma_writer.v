// dma_writer.v — Internal buffer to AXI4-Stream master
// Reads results from accumulator bank and streams to host.

`include "sparsenn_params.vh"

module dma_writer #(
    parameter DATA_WIDTH = `AXI_DATA_WIDTH,
    parameter ADDR_WIDTH = `AXI_ADDR_WIDTH,
    parameter BUF_DEPTH  = `ACC_BUF_DEPTH
) (
    input  wire                     clk,
    input  wire                     rst_n,

    // AXI4-Stream master interface
    output wire                     m_axis_tvalid,
    input  wire                     m_axis_tready,
    output wire [DATA_WIDTH-1:0]    m_axis_tdata,
    output reg                      m_axis_tlast,

    // Internal buffer read interface
    output reg                      buf_rd_en,
    output reg  [ADDR_WIDTH-1:0]    buf_rd_addr,
    input  wire [DATA_WIDTH-1:0]    buf_rd_data,
    input  wire                     buf_rd_valid,

    // Control
    input  wire                     start,
    input  wire [15:0]              total_words,     // Number of words to transfer
    output reg                      done
);

    localparam S_IDLE   = 2'd0;
    localparam S_ACTIVE = 2'd1;
    localparam S_DONE   = 2'd2;

    reg [1:0] state;
    reg [15:0] word_cnt;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE;
            buf_rd_en <= 0;
            buf_rd_addr <= 0;
            done <= 0;
            m_axis_tlast <= 0;
            word_cnt <= 0;
        end else begin
            case (state)
                S_IDLE: begin
                    done <= 0;
                    word_cnt <= 0;
                    m_axis_tlast <= 0;
                    buf_rd_en <= 0;
                    if (start) begin
                        buf_rd_en <= 1;
                        buf_rd_addr <= 0;
                        state <= S_ACTIVE;
                    end
                end

                S_ACTIVE: begin
                    if (m_axis_tvalid && m_axis_tready) begin
                        word_cnt <= word_cnt + 1;
                        buf_rd_addr <= word_cnt + 1;
                        buf_rd_en <= 1;

                        if (word_cnt + 1 >= total_words - 1) begin
                            m_axis_tlast <= 1;
                            state <= S_DONE;
                        end
                    end else begin
                        // Hold read address if stalled
                        buf_rd_en <= 0;
                    end
                end

                S_DONE: begin
                    done <= 1;
                    buf_rd_en <= 0;
                    m_axis_tlast <= 0;
                    if (!start)
                        state <= S_IDLE;
                end

                default: state <= S_IDLE;
            endcase
        end
    end

    assign m_axis_tvalid = (state == S_ACTIVE) && buf_rd_valid;
    assign m_axis_tdata  = buf_rd_data;

endmodule
