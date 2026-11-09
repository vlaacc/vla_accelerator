// kogge_stone_adder.v — 并行前缀 (Kogge-Stone) 加法器
// 16-bit 仅需 log2(16)=4 级组合门深度，关键路径 O(log N) 而非 ripple-carry O(N)
// 支持减法: a - b = a + (~b) + cin(=1)

module kogge_stone_adder #(
    parameter WIDTH = 16
) (
    input  wire [WIDTH-1:0] a,
    input  wire [WIDTH-1:0] b,
    input  wire             cin,
    output wire [WIDTH-1:0] sum,
    output wire             cout
);
    localparam LEVELS = $clog2(WIDTH);  // 16 → 4

    genvar l, i;

    // ── Stage 0: 每 bit 的 generate / propagate ──
    // G[i] = a[i] & b[i], P[i] = a[i] ^ b[i]
    wire [WIDTH-1:0] G [0:LEVELS];
    wire [WIDTH-1:0] P [0:LEVELS];

    assign G[0] = a & b;
    assign P[0] = a ^ b;

    // ── Kogge-Stone 前缀计算 ──
    // 第 l 级: 每 bit i 与 bit (i - 2^l) 合并
    // (G,P) ∘ (G',P') = (G | (P & G'), P & P')
    generate
        for (l = 0; l < LEVELS; l = l + 1) begin : ks_stage
            for (i = 0; i < WIDTH; i = i + 1) begin : ks_bit
                if (i >= (1 << l)) begin
                    assign G[l+1][i] = G[l][i] | (P[l][i] & G[l][i - (1 << l)]);
                    assign P[l+1][i] = P[l][i] & P[l][i - (1 << l)];
                end else begin
                    assign G[l+1][i] = G[l][i];
                    assign P[l+1][i] = P[l][i];
                end
            end
        end
    endgenerate

    // ── 进位链 ──
    // carry[0] = cin
    // carry[i] = G_final[i-1] | (P_final[i-1] & cin), i >= 1
    wire [WIDTH-1:0] carry;
    assign carry[0] = cin;
    generate
        for (i = 1; i < WIDTH; i = i + 1) begin : carry_gen
            assign carry[i] = G[LEVELS][i-1] | (P[LEVELS][i-1] & cin);
        end
    endgenerate

    // ── 和 ──
    assign sum = P[0] ^ carry;
    assign cout = G[LEVELS][WIDTH-1] | (P[LEVELS][WIDTH-1] & cin);

endmodule
