// PROBLEM: prefix sum (running total) out[i] = a[0] + ... + a[i]
// DEPENDENCY: out[i] needs out[i-1] -> plain parallel for is WRONG
// FIX (block method): one block per thread
//   pass 1 (parallel): running total inside my block
//   middle (single):   running total of the block totals
//   pass 2 (parallel): add the total of all earlier blocks
#include <iostream>
#include <vector>
#include <omp.h>
using namespace std;

int main() {
    long n = 20000000;
    vector<long> a(n), s(n), p(n);
    for (long i = 0; i < n; i++) a[i] = i % 7;

    double start = omp_get_wtime();
    s[0] = a[0];
    for (long i = 1; i < n; i++) s[i] = s[i - 1] + a[i];
    double ts = omp_get_wtime() - start;

    for (int t = 1; t <= 8; t = t * 2) {
        omp_set_num_threads(t);
        vector<long> block(t + 1, 0);
        start = omp_get_wtime();
        #pragma omp parallel
        {
            int id = omp_get_thread_num(), nt = omp_get_num_threads();
            long lo = n * id / nt, hi = n * (id + 1) / nt;
            long run = 0;
            for (long i = lo; i < hi; i++) { run += a[i]; p[i] = run; }
            block[id + 1] = run;
            #pragma omp barrier
            #pragma omp single
            for (int k = 1; k <= nt; k++) block[k] += block[k - 1];
            long off = block[id];
            for (long i = lo; i < hi; i++) p[i] += off;
        }
        double tp = omp_get_wtime() - start;
        cout << "threads = " << t << "  time = " << tp << "  speedup = " << ts / tp
             << "  " << (p == s ? "MATCH" : "MISMATCH") << "\n";
    }
}
