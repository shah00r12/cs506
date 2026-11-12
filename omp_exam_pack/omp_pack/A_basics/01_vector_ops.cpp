// PROBLEM: element-wise vector work: c = a + b, saxpy c = k*a + b, square, reverse into new array
// PATTERN: each i writes its OWN box -> #pragma omp parallel for
// TRAPS: separate output array; pragma line directly followed by for
#include <iostream>
#include <vector>
#include <omp.h>
using namespace std;

int main() {
    long n = 10000000;
    double k = 3.0;
    vector<double> a(n), b(n), c1(n), c2(n), r1(n), r2(n);
    for (long i = 0; i < n; i++) { a[i] = i % 100; b[i] = 1; }

    double start = omp_get_wtime();
    for (long i = 0; i < n; i++) {
        c1[i] = k * a[i] + b[i];
        r1[n - 1 - i] = a[i];
    }
    double ts = omp_get_wtime() - start;
    cout << "N = " << n << "  serial time = " << ts << "\n";

    for (int t = 1; t <= 8; t = t * 2) {
        omp_set_num_threads(t);
        start = omp_get_wtime();
        #pragma omp parallel for
        for (long i = 0; i < n; i++) {
            c2[i] = k * a[i] + b[i];
            r2[n - 1 - i] = a[i];
        }
        double tp = omp_get_wtime() - start;
        cout << "threads = " << t << "  time = " << tp << "  speedup = " << ts / tp
             << "  efficiency = " << ts / tp / t << "  " << ((c1 == c2 && r1 == r2) ? "MATCH" : "MISMATCH") << "\n";
    }
}
