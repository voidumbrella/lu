const std = @import("std");
const mem = std.mem;

const assert = std.debug.assert;
const Allocator = mem.Allocator;

/// Matrix stored in column-major order.
fn Matrix(comptime T: type) type {
    return struct {
        rows: usize,
        cols: usize,
        data: []T,

        const Self = @This();

        pub fn init(allocator: *Allocator, rows: usize, cols: usize) !Self {
            return Self{
                .rows = rows,
                .cols = cols,
                .data = try allocator.alloc(T, rows * cols),
            };
        }

        pub fn deinit(self: *Self, allocator: *Allocator) void {
            allocator.free(self.data);
        }

        inline fn index(self: Self, i: usize, j: usize) usize {
            assert(i < self.rows);
            assert(j < self.cols);
            return i * self.cols + j;
        }

        pub inline fn get(self: *const Self, i: usize, j: usize) T {
            return self.data[self.index(i, j)];
        }

        pub inline fn get_mut(self: *Self, i: usize, j: usize) *T {
            return &self.data[self.index(i, j)];
        }

        pub inline fn set(self: *Self, i: usize, j: usize, n: T) void {
            self.data[self.index(i, j)] = n;
        }

        pub fn format(self: Self, comptime fmt: []const u8, options: std.fmt.FormatOptions, writer: anytype) !void {
            // Suboptimal cache use (forced to iterate by rows)
            // but doesn't really matter if we're printing a matrix
            var i: usize = 0;
            while (i < self.rows) : (i += 1) {
                var j: usize = 0;
                while (j < self.cols) : (j += 1) {
                    try writer.print("{d:>12.7} ", .{self.get(i, j)});
                }
                try writer.writeAll("\n");
            }
        }

        // Zeroes a matrix.
        pub fn zero(self: *Self) *Self {
            mem.set(T, self.data, 0);
            return self;
        }

        // Zeroes a matrix, except for the main diagonal elements which are set to 1.
        pub fn ident(self: *Self) *Self {
            self.zero();

            const limit = std.math.min(self.rows, self.cols);
            var i: usize = 0;
            while (i < limit) : (i += 1) {
                self.set(i, i, 1);
            }
            return self;
        }

        // Using the provided random number generator, fills the matrix with random coefficients.
        // Supports the following types:
        // - integer types: [0, maxInt(T)]
        // - f32, f64: [0, 1)
        pub fn rand(self: *Self, rng: *std.rand.Random) *Self {
            var i: usize = 0;
            while (i < self.rows) : (i += 1) {
                var j: usize = 0;
                while (j < self.cols) : (j += 1) {
                    switch (@typeInfo(T)) {
                        .Int => self.set(i, j, rng.int(T)),
                        .Float => self.set(i, j, rng.float(T)),
                        else => @compileError("Unsupported type for rand()"),
                    }
                }
            }
            return self;
        }

        // Using the provided random number generator, fills the matrix with random coefficients
        // in the range [min, max). Asserts that `min` < `max`.
        // Supports the following types:
        // - integer types
        // - f32, f64
        pub fn randRange(self: *Self, rng: *std.rand.Random, min: T, max: T) *Self {
            assert(min < max);
            var i: usize = 0;
            while (i < self.rows) : (i += 1) {
                var j: usize = 0;
                while (j < self.cols) : (j += 1) {
                    switch (@typeInfo(T)) {
                        .Int => self.set(i, j, rng.intRangeLessThan(T, min, max)),
                        .Float => self.set(i, j, (max - min) * rng.float(T) + min),
                        else => @compileError("Unsupported type for randRange()"),
                    }
                }
            }
            return self;
        }

        // Multiplies every element by the given scalar `s`.
        pub fn mulScalar(self: *Self, s: T) *Self {
            var i: usize = 0;
            while (i < self.rows) : (i += 1) {
                var j: usize = 0;
                while (j < self.cols) : (j += 1) {
                    self.get_mut(i, j).* *= s;
                }
            }
            return self;
        }

        /// General matrix multiply (GEMM) adds the matrix product AB to self.
        /// Asserts that the sizes of each matrix are compatible;
        /// if A is a nxm matrix, B should be a mxp matrix and C be a nxp matrix.
        ///
        /// If you just need the matrix product AB, initialize C to be the zero matrix using `zero`.
        pub fn gemm(noalias self: *Self, a: Self, b: Self) *Self {
            assert(a.cols == b.rows);
            assert(a.rows == self.rows);
            assert(b.cols == self.cols);

            var i: usize = 0;
            while (i < a.rows) : (i += 1) {
                var k: usize = 0;
                while (k < a.cols) : (k += 1) {
                    var j: usize = 0;
                    while (j < b.cols) : (j += 1) {
                        self.get_mut(i, j).* += a.get(i, k) * b.get(k, j);
                    }
                }
            }
            return self;
        }
    };
}

pub fn main() anyerror!void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = &gpa.allocator;

    const stdout_raw = std.io.getStdOut().writer();
    var stdout_buffered = std.io.bufferedWriter(stdout_raw);
    const stdout = stdout_buffered.writer();

    var seedbuf: [4]u8 = undefined;
    try std.os.getrandom(seedbuf[0..]);
    const seed = std.mem.readIntNative(u32, seedbuf[0..]);
    var rng = &std.rand.DefaultPrng.init(seed).random;
    {
        const N = 4096;
        const M = 4096;

        var A = try Matrix(f32).init(allocator, N, M);
        defer A.deinit(allocator);

        var B = try Matrix(f32).init(allocator, M, N);
        defer B.deinit(allocator);

        _ = A.rand(rng);
        _ = B.rand(rng);

        var C = try Matrix(f32).init(allocator, N, N);
        defer C.deinit(allocator);
        _ = C.zero();
        _ = C.gemm(A, B);
    }
    try stdout_buffered.flush();
}
