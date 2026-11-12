// PROBLEM: y = A x; row sums; column sums; trace; total
// y[i] / rowSum[i]: parallel for over ROWS only, local sum (NO collapse: all j add into the same y[i])
// colSum[j]: parallel for over COLUMNS, i loop inside
// trace / total: one number -> reduction(+)
#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>
using namespace std;

int main() {
    int n = 4000;
    vector<double> A((long)n * n), x(n), y1(n), y2(n), cs1(n), cs2(n);
    for (long k = 0; k < (long)n * n; k++) A[k] = k % 10;
    for (int j = 0; j < n; j++) x[j] = 1;

    double start = omp_get_wtime();
    for (int i = 0; i < n; i++) {
        double s = 0;
        for (int j = 0; j < n; j++) s += A[(long)i * n + j] * x[j];
        y1[i] = s;
    }
    double ts = omp_get_wtime() - start;
    for (int j = 0; j < n; j++) {
        double s = 0;
        for (int i = 0; i < n; i++) s += A[(long)i * n + j];
        cs1[j] = s;
    }

    for (int t = 1; t <= 8; t = t * 2) {
        omp_set_num_threads(t);
        start = omp_get_wtime();
        #pragma omp parallel for
        for (int i = 0; i < n; i++) {
            double s = 0;
            for (int j = 0; j < n; j++) s += A[(long)i * n + j] * x[j];
            y2[i] = s;
        }
        double tp = omp_get_wtime() - start;
        bool ok = true;
        for (int i = 0; i < n; i++) if (fabs(y1[i] - y2[i]) > 1e-6) ok = false;
        cout << "A*x threads = " << t << "  time = " << tp << "  speedup = " << ts / tp << "  " << (ok ? "MATCH" : "MISMATCH") << "\n";
    }

    #pragma omp parallel for
    for (int j = 0; j < n; j++) {
        double s = 0;
        for (int i = 0; i < n; i++) s += A[(long)i * n + j];
        cs2[j] = s;
    }
    double tr = 0, total = 0;
    #pragma omp parallel for reduction(+:tr)
    for (int i = 0; i < n; i++) tr += A[(long)i * n + i];
    #pragma omp parallel for collapse(2) reduction(+:total)
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) total += A[(long)i * n + j];
    cout << "column sums " << (cs1 == cs2 ? "MATCH" : "MISMATCH") << "  trace = " << tr << "  total = " << total << "\n";
}
