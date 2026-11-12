// PROBLEM: BLOCKED (tiled) matrix multiplication
// IDEA: cut the matrices into small BS x BS blocks that fit in the CPU cache.
//   C block (ii, jj) += A block (ii, kk) * B block (kk, jj), for every kk
//   Working on small blocks re-uses data from the cache -> faster than plain i-j-k for big N
// PATTERN: each C block (ii, jj) is written by ONE thread -> parallel for collapse(2) over ii, jj
//          kk, i, k, j loops stay inside (kk adds into the same C block -> never split kk)
// N must be a multiple of BS here (or use min(ii + BS, n) as the loop end)
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <omp.h>
using namespace std;

int main() {
    int n = 768, BS = 64;
    vector<double> A(n * n), B(n * n), C1(n * n, 0), C2(n * n);
    for (int k = 0; k < n * n; k++) { A[k] = k % 10; B[k] = k % 7; }

    // plain serial ijk
    double start = omp_get_wtime();
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            double sum = 0;
            for (int k = 0; k < n; k++) sum += A[i * n + k] * B[k * n + j];
            C1[i * n + j] = sum;
        }
    double ts = omp_get_wtime() - start;
    cout << "N = " << n << ", block = " << BS << "  plain serial = " << ts << "\n";

    for (int t = 1; t <= 8; t = t * 2) {
        omp_set_num_threads(t);
        for (int k = 0; k < n * n; k++) C2[k] = 0;
        start = omp_get_wtime();
        #pragma omp parallel for collapse(2)
        for (int ii = 0; ii < n; ii += BS)
            for (int jj = 0; jj < n; jj += BS)
                for (int kk = 0; kk < n; kk += BS)
                    for (int i = ii; i < min(ii + BS, n); i++)
                        for (int k = kk; k < min(kk + BS, n); k++) {
                            double aik = A[i * n + k];
                            for (int j = jj; j < min(jj + BS, n); j++)
                                C2[i * n + j] += aik * B[k * n + j];
                        }
        double tp = omp_get_wtime() - start;
        bool ok = true;
        for (int k = 0; k < n * n; k++) if (fabs(C1[k] - C2[k]) > 1e-6) ok = false;
        cout << "threads = " << t << "  blocked = " << tp << "  speedup vs plain serial = " << ts / tp
             << "  " << (ok ? "MATCH" : "MISMATCH") << "\n";
    }
}
