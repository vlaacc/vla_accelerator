// bit_parallel_accumulator.v — 1-bit subset mux (bit-plane engine)
//
// bit-plane 引擎里每个 group（4 权重）只算 1 个 weight bit-plane 的贡献：
//   slice = subset_sum[mask]，mask = 4 个权重的第 b bit。
// 16:1 MUX 从 build_unit 的 16 个 subset sum 里选一个（7-bit 有符号）。
// 无加法树、无移位（移位 << b 和符号在 acc_rmw 里做）。
//
// subset_sum[7*i +: 7] = i-th subset sum (mask i), signed 7-bit
// mask[3:0] = {w3[b], w2[b], w1[b], w0[b]}

module bit_parallel_accumulator (
    input  wire [3:0]   mask,         // 4-bit bit-plane mask
    input  wire [111:0] subset_sum,   // 16 signed subset sums (build_unit)
    output wire [6:0]   slice         // subset_sum[mask] (signed 7-bit)
);
    // 16:1 mux
    assign slice = subset_sum[mask*7 +: 7];

endmodule
