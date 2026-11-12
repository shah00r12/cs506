// PROBLEM: static vs dynamic vs guided scheduling on uneven work
//   static:  equal split at the start (best for equal work, no overhead)
//   dynamic: chunks on request (balances uneven work, overhead per chunk)
//   guided:  chunks start big and shrink
// work A = (i % 1000) * 100 (sir's, repeating -> static already fair)
// work B = i / 2 (growing -> static gives the last thread all heavy turns -> dynamic wins)
// work C = triangle loop: for j < i (same as B in shape)
#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>
using namespace std;

int workOf(long i, int mode) { return mode == 0 ? (i % 1000) * 100 : i / 2; }

double body(double x, int w) { for (int j = 0; j < w; j++) x = x * 1.0000001 + 0.0000001; return x; }

int main() {
    long n = 40000;
    const char *names[] = {"repeating work", "growing work"};
    for (int mode = 0; mode < 2; mode++) {
        vector<double> s(n), p1(n), p2(n), p3(n), p4(n);
        double st = omp_get_wtime();
        for (long i = 0; i < n; i++) s[i] = body(i * 0.001, workOf(i, mode));
        double ts = omp_get_wtime() - st;

        st = omp_get_wtime();
        #pragma omp parallel for schedule(static)
        for (long i = 0; i < n; i++) p1[i] = body(i * 0.001, workOf(i, mode));
        double t1 = omp_get_wtime() - st;

        st = omp_get_wtime();
        #pragma omp parallel for schedule(dynamic, 100)
        for (long i = 0; i < n; i++) p2[i] = body(i * 0.001, workOf(i, mode));
        double t2 = omp_get_wtime() - st;

        st = omp_get_wtime();
        #pragma omp parallel for schedule(dynamic, 1)
        for (long i = 0; i < n; i++) p3[i] = body(i * 0.001, workOf(i, mode));
        double t3 = omp_get_wtime() - st;

        st = omp_get_wtime();
        #pragma omp parallel for schedule(guided)
        for (long i = 0; i < n; i++) p4[i] = body(i * 0.001, workOf(i, mode));
        double t4 = omp_get_wtime() - st;

        bool ok = (s == p1 && s == p2 && s == p3 && s == p4);
        cout << names[mode] << ": serial " << ts << " | static " << t1 << " | dynamic,100 " << t2
             << " | dynamic,1 " << t3 << " | guided " << t4 << "  " << (ok ? "MATCH" : "MISMATCH") << "\n";
    }
}
