// carry_select_add16.v — 16-bit carry-select adder (2x 8-bit groups)
//
// Replaces a 16-bit ripple-carry (~1.5ns in Nangate45) with a single-cycle
// carry-select: the low 8-bit ripple produces the carry, and the high 8 bits are
// computed BOTH ways (carry-in 0 and 1) in parallel, then MUX-selected.
// Critical path = 8-bit ripple (low) + MUX ~0.5ns, so a 16-bit shift-add fits in
// one cycle without the 2x 8-bit pipelined split (and its handoff registers).
//
// Option A of the final-accumulation/timing work: this lets the bit-serial
// Horner run as 8 bit-stages (1 cycle each) instead of 16 sub-stages.
module carry_select_add16 (
    input  wire [15:0] a,
    input  wire [15:0] b,
    input  wire        cin,
    output wire [15:0] sum,
    output wire        cout
);
    // low 8-bit group (carry-in = cin)
    wire [8:0] lo = {1'b0, a[7:0]} + {1'b0, b[7:0]} + {8'd0, cin};
    // high 8-bit group, both carry-in cases in parallel
    wire [8:0] hi0 = {1'b0, a[15:8]} + {1'b0, b[15:8]};
    wire [8:0] hi1 = {1'b0, a[15:8]} + {1'b0, b[15:8]} + 9'd1;

    assign sum[7:0]  = lo[7:0];
    assign sum[15:8] = lo[8] ? hi1[7:0] : hi0[7:0];
    assign cout      = lo[8] ? hi1[8]  : hi0[8];
endmodule
