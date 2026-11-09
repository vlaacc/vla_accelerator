// sparse_decoder.v — 2:4 structured sparsity decompressor
//
// Compressed format (per 4-element group):
//   {metadata[`SPARSITY_META_BITS-1:0], val1[`FP16_WIDTH-1:0], val2[`FP16_WIDTH-1:0]}
//   Total: SPARSITY_META_BITS + 2*FP16_WIDTH bits per group
//
// Metadata encoding (exactly 2 non-zeros out of 4):
//   3'd0: positions {0,1}  3'd1: positions {0,2}  3'd2: positions {0,3}
//   3'd3: positions {1,2}  3'd4: positions {1,3}  3'd5: positions {2,3}
//   3'd6, 3'd7: reserved (all-zero group, skip)

`include "sparsenn_params.vh"

module sparse_decoder #(
    parameter N_GROUPS = `M_PE_COLS / `SPARSITY_M   // Number of 4-element groups per row
) (
    input  wire                                              clk,
    input  wire                                              rst_n,

    // Compressed input (one group per cycle)
    input  wire [`SPARSITY_META_BITS + 2*`FP16_WIDTH - 1:0]  compressed_in,
    input  wire                                              valid_in,
    output wire                                              ready_out,

    // Decompressed output (4 FP16 values per group)
    output wire [`SPARSITY_M*`FP16_WIDTH-1:0]                decompressed_out,
    output wire                                              valid_out,
    input  wire                                              ready_in
);

    // ── Decompose compressed input ──
    wire [`SPARSITY_META_BITS-1:0] metadata;
    wire [`FP16_WIDTH-1:0]         val1, val2;

    assign metadata = compressed_in[`SPARSITY_META_BITS + 2*`FP16_WIDTH - 1 : 2*`FP16_WIDTH];
    assign val1     = compressed_in[2*`FP16_WIDTH - 1 : `FP16_WIDTH];
    assign val2     = compressed_in[`FP16_WIDTH - 1 : 0];

    // ── Decode metadata to position mask ──
    reg [`SPARSITY_M-1:0] pos_mask;

    always @(*) begin
        case (metadata)
            3'd0: pos_mask = 4'b0011;  // {0,1}
            3'd1: pos_mask = 4'b0101;  // {0,2}
            3'd2: pos_mask = 4'b1001;  // {0,3}
            3'd3: pos_mask = 4'b0110;  // {1,2}
            3'd4: pos_mask = 4'b1010;  // {1,3}
            3'd5: pos_mask = 4'b1100;  // {2,3}
            default: pos_mask = 4'b0000;  // all-zero (skip)
        endcase
    end

    // ── Place values into correct positions ──
    reg [`FP16_WIDTH-1:0] decompressed [`SPARSITY_M-1:0];

    always @(*) begin
        // Zero-fill all positions
        decompressed[0] = `FP16_WIDTH'd0;
        decompressed[1] = `FP16_WIDTH'd0;
        decompressed[2] = `FP16_WIDTH'd0;
        decompressed[3] = `FP16_WIDTH'd0;

        // Place val1 and val2 according to metadata
        case (metadata)
            3'd0: begin decompressed[0] = val1; decompressed[1] = val2; end
            3'd1: begin decompressed[0] = val1; decompressed[2] = val2; end
            3'd2: begin decompressed[0] = val1; decompressed[3] = val2; end
            3'd3: begin decompressed[1] = val1; decompressed[2] = val2; end
            3'd4: begin decompressed[1] = val1; decompressed[3] = val2; end
            3'd5: begin decompressed[2] = val1; decompressed[3] = val2; end
            default: ;  // All zeros
        endcase
    end

    // ── Assemble output ──
    assign decompressed_out = {decompressed[3], decompressed[2], decompressed[1], decompressed[0]};

    // ── Handshake ──
    assign ready_out = ready_in;
    assign valid_out = valid_in;

endmodule
