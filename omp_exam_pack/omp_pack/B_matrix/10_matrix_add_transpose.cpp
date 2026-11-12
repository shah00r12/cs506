// PROBLEM: matrix add C = A + B, transpose T = A^T (rectangular), scale by k
// STORAGE: 1D vector, row by row: box (i, j) of an R x C matrix is M[i * C + j]
// PATTERN: each (i, j) writes its own box -> parallel for collapse(2)
// TRANSPOSE: T is C x R -> T[j * R + i] = A[i * C + j]  (each matrix uses its own column count)
#include <iostream>
#include <vector>
#include <omp.h>
using namespace std;

int main() {
    int R = 3000, C = 4000;
    long sz = (long)R * C;
    vector<double> A(sz), B(sz), S1(sz), S2(sz), T1(sz), T2(sz);
    for (long k = 0; k < sz; k++) { A[k] = k % 100; B[k] = 2; }

    double start = omp_get_wtime();
    for (int i = 0; i < R; i++)
        for (int j = 0; j < C; j++) {
            S1[(long)i * C + j] = A[(long)i * C + j] + B[(long)i * C + j];
            T1[(long)j * R + i] = A[(long)i * C + j];
        }
    double ts = omp_get_wtime() - start;
    cout << R << " x " << C << "  serial time = " << ts << "\n";

    for (int t = 1; t <= 8; t = t * 2) {
        omp_set_num_threads(t);
        start = omp_get_wtime();
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < R; i++)
            for (int j = 0; j < C; j++) {
                S2[(long)i * C + j] = A[(long)i * C + j] + B[(long)i * C + j];
                T2[(long)j * R + i] = A[(long)i * C + j];
            }
        double tp = omp_get_wtime() - start;
        cout << "threads = " << t << "  time = " << tp << "  speedup = " << ts / tp
             << "  efficiency = " << ts / tp / t << "  " << ((S1 == S2 && T1 == T2) ? "MATCH" : "MISMATCH") << "\n";
    }
}
