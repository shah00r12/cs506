// PROBLEM: C = A x B,  C(i,j) = sum over k of A(i,k) * B(k,j)
// PATTERN: collapse(2) on i, j (each writes its own C box); local sum; k loop INSIDE
// NEVER split k: all k steps add into the SAME C(i,j) -> race
// i-k-j order: inner loop walks rows of B and C (cache friendly) -> much faster; parallel on i only
#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>
using namespace std;

int main() {
    int n = 600;
    vector<double> A(n * n), B(n * n), C1(n * n), C2(n * n), C3(n * n);
    for (int k = 0; k < n * n; k++) { A[k] = k % 10; B[k] = k % 7; }

    double start = omp_get_wtime();
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            double sum = 0;
            for (int k = 0; k < n; k++) sum += A[i * n + k] * B[k * n + j];
            C1[i * n + j] = sum;
        }
    double ts = omp_get_wtime() - start;
    cout << "N = " << n << "  serial (ijk) = " << ts << "\n";

    for (int t = 1; t <= 8; t = t * 2) {
        omp_set_num_threads(t);
        start = omp_get_wtime();
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++) {
                double sum = 0;
                for (int k = 0; k < n; k++) sum += A[i * n + k] * B[k * n + j];
                C2[i * n + j] = sum;
            }
        double tp = omp_get_wtime() - start;

        for (int k = 0; k < n * n; k++) C3[k] = 0;
        double s2 = omp_get_wtime();
        #pragma omp parallel for
        for (int i = 0; i < n; i++)
            for (int k = 0; k < n; k++) {
                double aik = A[i * n + k];
                for (int j = 0; j < n; j++) C3[i * n + j] += aik * B[k * n + j];
            }
        double tk = omp_get_wtime() - s2;

        bool ok = true;
        for (int k = 0; k < n * n; k++) if (fabs(C1[k] - C2[k]) > 1e-6 || fabs(C1[k] - C3[k]) > 1e-6) ok = false;
        cout << "threads = " << t << "  ijk = " << tp << " (speedup " << ts / tp << ", eff " << ts / tp / t
             << ")  ikj = " << tk << "  " << (ok ? "MATCH" : "MISMATCH") << "\n";
    }
}
