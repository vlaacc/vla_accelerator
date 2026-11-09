// csa3.v — 3:2 carry-save compressor (full-adder array, NO carry propagation)
//
// Reduces three N-bit numbers to two N-bit numbers in constant gate depth
// (~1 full adder ≈ 0.1ns in Nangate45), deferring all carry propagation to a
// single final carry-propagate adder.  This is the primitive that lets the
// bit-serial Horner and the final accumulation tree run without a ripple /
// carry-select adder on the per-stage critical path.
//
// Identity:  s + co = a + b + c  (mod 2^W)
//   s[i]   = a[i] ^ b[i] ^ c[i]              (parity)
//   co[i+1]= majority(a[i], b[i], c[i])      (carry, stored ALREADY shifted left 1)
// so co = {maj[W-2:0], 1'b0}.  The top carry maj[W-1] is dropped (mod 2^W);
// values here are bounded well inside W bits so it is always 0.
module csa3 #(parameter W = 16) (
    input  wire [W-1:0] a,
    input  wire [W-1:0] b,
    input  wire [W-1:0] c,
    output wire [W-1:0] s,
    output wire [W-1:0] co
);
    wire [W-1:0] maj = (a & b) | (a & c) | (b & c);
    assign s  = a ^ b ^ c;
    assign co = {maj[W-2:0], 1'b0};
endmodule
