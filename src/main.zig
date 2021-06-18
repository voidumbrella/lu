const std = @import("std");
const mem = std.mem;
const time = std.time;

const assert = std.debug.assert;
const Allocator = mem.Allocator;

/// Matrix stored in row-major order.
fn Matrix(comptime T: type) type {
    return struct {
        rows: usize,
        cols: usize,
        data: []T,

        const Self = @This();

        pub fn init(allocator: *Allocator, rows: usize, cols: usize) !Self {
            assert(rows > 0 and cols > 0);
            switch (@typeInfo(T)) {
                .Float => {},
                else => @compileError("Unsupported type for Matrix"),
            }
            return Self{
                .rows = rows,
                .cols = cols,
                .data = try allocator.alloc(T, rows * cols),
            };
        }

        pub fn deinit(A: *Self, allocator: *Allocator) void {
            allocator.free(A.data);
        }

        pub fn copy(noalias dest: *Self, src: Self) void {
            mem.copy(T, dest.data, src.data);
        }

        inline fn index(A: Self, i: usize, j: usize) usize {
            assert(i < A.rows);
            assert(j < A.cols);
            return i * A.cols + j;
        }

        pub inline fn get(A: *const Self, i: usize, j: usize) T {
            return A.data[A.index(i, j)];
        }

        pub inline fn get_mut(A: *Self, i: usize, j: usize) *T {
            return &A.data[A.index(i, j)];
        }

        pub inline fn set(A: *Self, i: usize, j: usize, n: T) void {
            A.data[A.index(i, j)] = n;
        }

        pub fn format(A: Self, comptime fmt: []const u8, options: std.fmt.FormatOptions, writer: anytype) !void {
            _ = fmt;
            _ = options;

            var i: usize = 0;
            while (i < A.rows) : (i += 1) {
                var j: usize = 0;
                while (j < A.cols) : (j += 1) {
                    try writer.print("{d:>12.7} ", .{A.get(i, j)});
                }
                try writer.writeAll("\n");
            }
        }

        // Zeroes a matrix.
        pub fn zero(A: *Self) void {
            mem.set(T, A.data, 0);
        }

        // Zeroes a matrix, except for the main diagonal elements which are set to 1.
        pub fn ident(A: *Self) void {
            A.zero();

            const limit = std.math.min(A.rows, A.cols);
            var i: usize = 0;
            while (i < limit) : (i += 1) {
                A.set(i, i, 1);
            }
        }

        // Using the provided random number generator, fills the matrix with random coefficients.
        pub fn rand(A: *Self, rng: *std.rand.Random) void {
            var i: usize = 0;
            while (i < A.rows) : (i += 1) {
                var j: usize = 0;
                while (j < A.cols) : (j += 1) {
                    A.set(i, j, rng.float(T));
                }
            }
        }

        // Using the provided random number generator, fills the matrix with random coefficients
        // in the range [min, max). Asserts that `min` < `max`.
        pub fn randRange(A: *Self, rng: *std.rand.Random, min: T, max: T) void {
            assert(min < max);
            var i: usize = 0;
            while (i < A.rows) : (i += 1) {
                var j: usize = 0;
                while (j < A.cols) : (j += 1) {
                    A.set(i, j, (max - min) * rng.float(T) + min);
                }
            }
        }

        /// General matrix multiply (GEMM) adds the matrix product AB to A.
        /// Asserts that the sizes of each matrix are compatible;
        /// if A is a nxm matrix, B should be a mxp matrix and C be a nxp matrix.
        ///
        /// If you just need the matrix product AB, initialize C to be the zero matrix using `zero`.
        pub fn gemm(noalias C: *Self, A: Self, B: Self) void {
            assert(A.cols == B.rows);
            assert(A.rows == C.rows);
            assert(B.cols == C.cols);

            var i: usize = 0;
            while (i < A.rows) : (i += 1) {
                var k: usize = 0;
                while (k < A.cols) : (k += 1) {
                    var j: usize = 0;
                    while (j < B.cols) : (j += 1) {
                        C.get_mut(i, j).* += A.get(i, k) * B.get(k, j);
                    }
                }
            }
        }

        /// Solves a linear system described by Ax = b using LU decomposition.
        /// where `A` is a n x n matrix and `x` and `b` are n x 1 matrices.
        /// `x` does not need to be initialized.
        ///
        /// The solution is stored in `x`, and `A` is overwritten with its LU decomposition.
        pub fn luSolve(noalias x: *Self, noalias A: *Self, b: Self) void {
            assert(A.rows == A.cols);
            assert(A.rows == x.rows);
            assert(A.rows == b.rows);

            const N = A.rows;

            // Ax = L(Ux) = b
            A.luDecompose();

            // Solve Lw = b first
            {
                mem.copy(T, x.data, b.data);
                var i: usize = 0;
                while (i < N) : (i += 1) {
                    var j: usize = i + 1;
                    const a = x.get(i, 0);
                    while (j < N) : (j += 1) {
                        x.get_mut(j, 0).* -= A.get(j, i) * a;
                    }
                }
            }

            // Then solve Ux = w
            {
                var i: usize = N - 1;
                while (true) : (i -= 1) {
                    var res: T = x.get(i, 0);

                    var j: usize = i + 1;
                    while (j < N) : (j += 1) {
                        res -= A.get(i, j) * x.get(j, 0);
                    }

                    x.set(i, 0, res / A.get(i, i));

                    if (i == 0) break;
                }
            }
        }

        /// Performs an in-place LU decomposition with partial pivoting.
        /// The elements of L are stored in the lower triangular elements, (diagonal assumed to be 1)
        /// and elements of U are stored in the diagonal and upper triangular elements.
        ///
        /// TODO: Improve this to LUP decomposition
        pub fn luDecompose(A: *Self) void {
            assert(A.rows == A.cols);

            // Gaussian elimination recursively acting on k x k submatrices
            var k: usize = 0;
            while (k < A.rows) : (k += 1) {
                var i: usize = k + 1;
                while (i < A.rows) : (i += 1) {
                    // Find multiplier to eliminate next row
                    const mult = A.get(i, k) / A.get(k, k);
                    A.set(i, k, mult);

                    // Eliminate the next row
                    var j: usize = k + 1;
                    while (j < A.rows) : (j += 1) {
                        A.get_mut(i, j).* -= mult * A.get(k, j);
                    }
                }
            }
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

    const N = 4;
    var A = try Matrix(f64).init(allocator, N, N);
    defer A.deinit(allocator);
    var A0 = try Matrix(f64).init(allocator, N, N);
    defer A0.deinit(allocator);

    A.randRange(rng, 0, 100);
    A0.copy(A);

    var x = try Matrix(f64).init(allocator, N, 1);
    defer x.deinit(allocator);
    var b = try Matrix(f64).init(allocator, N, 1);
    defer b.deinit(allocator);
    b.randRange(rng, 0, 100);

    try stdout.print("A:\n{}\n", .{A});
    x.luSolve(&A, b);
    try stdout.print("x:\n{}\n", .{x});
    try stdout.print("b:\n{}\n", .{b});

    b.zero();
    b.gemm(A0, x);
    try stdout.print("Ax:\n{}\n", .{b});

    try stdout_buffered.flush();
}
