// TEMPLATE: copy this for ANY new problem. Fill in the 3 marked places.
// 1) write the NORMAL loop first   2) look at the LEFT side of '='   3) pick the pragma
#include <iostream>
#include <vector>
#include <cmath>
#include <climits>
#include <cstdlib>
#include <algorithm>
#include <omp.h>
using namespace std;

int main() {
    long n = 10000000;
    vector<long> a(n);
    for (long i = 0; i < n; i++) a[i] = i % 100;          // <-- 1. YOUR DATA

    // SERIAL
    double start = omp_get_wtime();
    long r1 = 0;
    for (long i = 0; i < n; i++)
        r1 += a[i];                                       // <-- 2. YOUR NORMAL LOOP BODY
    double ts = omp_get_wtime() - start;
    cout << "N = " << n << "  serial answer = " << r1 << "  time = " << ts << "\n";

    // PARALLEL, 1 2 4 8 threads
    for (int t = 1; t <= 8; t = t * 2) {
        omp_set_num_threads(t);
        start = omp_get_wtime();
        long r2 = 0;
        #pragma omp parallel for reduction(+:r2)           // <-- 3. YOUR PRAGMA
        for (long i = 0; i < n; i++)
            r2 += a[i];
        double tp = omp_get_wtime() - start;
        cout << "threads = " << t << "  time = " << tp << "  speedup = " << ts / tp
             << "  efficiency = " << ts / tp / t << "  " << (r1 == r2 ? "MATCH" : "MISMATCH") << "\n";
    }
}
