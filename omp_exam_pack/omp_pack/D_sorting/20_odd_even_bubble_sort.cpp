// PROBLEM: parallel bubble sort = odd-even transposition sort
// normal bubble: compare (j, j+1) then (j+1, j+2) share a[j+1] -> can't run together
// even round: (0,1)(2,3)...  odd round: (1,2)(3,4)...  -> pairs never overlap -> parallel
// SYNC: barrier between rounds (omp for gives it). NO nowait. parallel OUTSIDE the round loop.
// DESCENDING: swap when a[j] < a[j+1]
#include <iostream>
#include <vector>
#include <cstdlib>
#include <omp.h>
using namespace std;

int main() {
    int n = 20000;
    vector<int> a(n);
    for (int i = 0; i < n; i++) a[i] = rand() % 100000;

    vector<int> s = a;
    double st = omp_get_wtime();
    for (int round = 0; round < n; round++)
        for (int j = round % 2; j < n - 1; j += 2)
            if (s[j] > s[j + 1]) swap(s[j], s[j + 1]);
    double ts = omp_get_wtime() - st;
    cout << "N = " << n << "  serial = " << ts << "\n";

    for (int t = 1; t <= 8; t = t * 2) {
        omp_set_num_threads(t);
        vector<int> p = a;
        st = omp_get_wtime();
        #pragma omp parallel
        for (int round = 0; round < n; round++) {
            #pragma omp for
            for (int j = round % 2; j < n - 1; j += 2)
                if (p[j] > p[j + 1]) swap(p[j], p[j + 1]);
        }
        double tp = omp_get_wtime() - st;
        cout << "threads = " << t << "  time = " << tp << "  speedup = " << ts / tp << "  " << (p == s ? "MATCH" : "MISMATCH") << "\n";
    }
}
