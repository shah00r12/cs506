// PROBLEM: (a) N-body: force on each body from all other bodies (1D simplified)
//          (b) count pairs (i < j) whose sum equals a target; closest pair distance
// (a) each body i writes its own force[i] -> parallel for (inner loop over all j)
// (b) pairs: triangle loop (j from i+1) -> uneven work -> schedule(dynamic); count -> reduction(+);
//     closest -> reduction(min)
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <omp.h>
using namespace std;

int main() {
    int n = 20000;
    vector<double> pos(n), mass(n), f1(n), f2(n);
    for (int i = 0; i < n; i++) { pos[i] = i * 1.5 + (rand() % 100) / 100.0; mass[i] = rand() % 10 + 1; }

    double st = omp_get_wtime();
    for (int i = 0; i < n; i++) {
        double f = 0;
        for (int j = 0; j < n; j++) {
            if (i == j) continue;
            double d = pos[j] - pos[i];
            f += mass[i] * mass[j] / (d * fabs(d));
        }
        f1[i] = f;
    }
    double ts = omp_get_wtime() - st;
    st = omp_get_wtime();
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        double f = 0;
        for (int j = 0; j < n; j++) {
            if (i == j) continue;
            double d = pos[j] - pos[i];
            f += mass[i] * mass[j] / (d * fabs(d));
        }
        f2[i] = f;
    }
    double tp = omp_get_wtime() - st;
    bool ok = true;
    for (int i = 0; i < n; i++) if (fabs(f1[i] - f2[i]) > 1e-9 * (1 + fabs(f1[i]))) ok = false;
    cout << "(a) N-body: serial " << ts << "  parallel " << tp << "  speedup " << ts / tp << "  " << (ok ? "MATCH" : "MISMATCH") << "\n";

    vector<int> a(n);
    for (int i = 0; i < n; i++) a[i] = rand() % 1000;
    int target = 1000;
    long pairs1 = 0;
    for (int i = 0; i < n; i++) for (int j = i + 1; j < n; j++) if (a[i] + a[j] == target) pairs1++;
    long pairs2 = 0;
    double closest = 1e300;
    #pragma omp parallel for schedule(dynamic, 64) reduction(+:pairs2) reduction(min:closest)
    for (int i = 0; i < n; i++)
        for (int j = i + 1; j < n; j++) {
            if (a[i] + a[j] == target) pairs2++;
            double d = fabs(pos[i] - pos[j]);
            if (d < closest) closest = d;
        }
    cout << "(b) pairs summing to " << target << " = " << pairs2 << "  closest distance = " << closest
         << "  " << (pairs1 == pairs2 ? "MATCH" : "MISMATCH") << "\n";
}
