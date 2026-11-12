// PROBLEM: heat diffusion / Jacobi iteration on a 2D plate. Top edge is hot (100), others 0.
//   new(i, j) = average of the 4 neighbours (up, down, left, right) of OLD
//   repeat until the biggest change < tolerance (convergence)
// PATTERN: old -> new arrays (like Game of Life), collapse(2) for the cells,
//          biggest change = one number -> reduction(max:diff). Iterations are SERIAL.
#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>
using namespace std;

int run(int N, double tol, bool par, vector<double> &result) {
    vector<double> u(N * N, 0), un(N * N, 0);
    for (int j = 0; j < N; j++) { u[j] = 100; un[j] = 100; }      // hot top edge
    int iter = 0;
    double diff = tol + 1;
    while (diff > tol && iter < 5000) {
        diff = 0;
        #pragma omp parallel for collapse(2) reduction(max:diff) if(par)
        for (int i = 1; i < N - 1; i++)
            for (int j = 1; j < N - 1; j++) {
                double v = 0.25 * (u[(i - 1) * N + j] + u[(i + 1) * N + j] + u[i * N + j - 1] + u[i * N + j + 1]);
                un[i * N + j] = v;
                double d = fabs(v - u[i * N + j]);
                if (d > diff) diff = d;
            }
        swap(u, un);
        iter++;
    }
    result = u;
    return iter;
}

int main() {
    int N = 300;
    double tol = 1e-3;
    vector<double> r1, r2;
    double st = omp_get_wtime();
    int it1 = run(N, tol, false, r1);
    double ts = omp_get_wtime() - st;
    cout << "N = " << N << "  serial: " << it1 << " iterations, time = " << ts << "\n";

    for (int t = 1; t <= 8; t = t * 2) {
        omp_set_num_threads(t);
        st = omp_get_wtime();
        int it2 = run(N, tol, true, r2);
        double tp = omp_get_wtime() - st;
        bool ok = (it1 == it2);
        for (int k = 0; k < N * N; k++) if (fabs(r1[k] - r2[k]) > 1e-9) ok = false;
        cout << "threads = " << t << "  iterations = " << it2 << "  time = " << tp << "  speedup = " << ts / tp
             << "  " << (ok ? "MATCH" : "MISMATCH") << "\n";
    }
    cout << "temperature at centre = " << r1[(N / 2) * N + N / 2] << "\n";
}
