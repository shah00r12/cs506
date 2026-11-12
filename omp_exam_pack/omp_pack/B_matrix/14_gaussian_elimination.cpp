// PROBLEM: solve A x = b with Gaussian elimination
// STEP k (SERIAL, k = 0..n-1): use row k to make column k zero in every row below it
//   rows i > k are independent of each other in step k -> parallel for over i (own row)
//   step k+1 needs step k finished -> outer k loop serial (barrier at end of omp for)
// BACK SUBSTITUTION: x[i] needs x[i+1..n-1] -> serial outer loop; the inner sum is a reduction
// Matrix made diagonally dominant so no pivoting is needed.
#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>
using namespace std;

void solve(vector<double> A, vector<double> b, vector<double> &x, int n, bool par) {
    for (int k = 0; k < n; k++) {
        #pragma omp parallel for if(par)
        for (int i = k + 1; i < n; i++) {
            double f = A[(long)i * n + k] / A[(long)k * n + k];
            for (int j = k; j < n; j++) A[(long)i * n + j] -= f * A[(long)k * n + j];
            b[i] -= f * b[k];
        }
    }
    for (int i = n - 1; i >= 0; i--) {
        double s = 0;
        #pragma omp parallel for reduction(+:s) if(par)
        for (int j = i + 1; j < n; j++) s += A[(long)i * n + j] * x[j];
        x[i] = (b[i] - s) / A[(long)i * n + i];
    }
}

int main() {
    int n = 1200;
    vector<double> A((long)n * n), b(n), xs(n), xp(n), xtrue(n);
    for (int i = 0; i < n; i++) {
        xtrue[i] = i % 5 + 1;
        for (int j = 0; j < n; j++) A[(long)i * n + j] = (i == j) ? n : (double)((i + j) % 7) / 10;
    }
    for (int i = 0; i < n; i++) {                   // b = A * xtrue, so the answer must be xtrue
        double s = 0;
        for (int j = 0; j < n; j++) s += A[(long)i * n + j] * xtrue[j];
        b[i] = s;
    }

    double start = omp_get_wtime();
    solve(A, b, xs, n, false);
    double ts = omp_get_wtime() - start;
    cout << "N = " << n << "  serial time = " << ts << "\n";

    for (int t = 1; t <= 8; t = t * 2) {
        omp_set_num_threads(t);
        start = omp_get_wtime();
        solve(A, b, xp, n, true);
        double tp = omp_get_wtime() - start;
        double err = 0;
        for (int i = 0; i < n; i++) err = max(err, fabs(xp[i] - xtrue[i]));
        cout << "threads = " << t << "  time = " << tp << "  speedup = " << ts / tp
             << "  max error = " << err << "  " << (err < 1e-6 ? "MATCH" : "MISMATCH") << "\n";
    }
}
