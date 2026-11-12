// PROBLEM: histogram of values 0..9; grade groups (mark/10); letter count (c - 'a')
// RACE: all threads hit the same few boxes with ++
// PATTERN: own local array per thread -> omp for -> critical merge once (fast)
//          also: atomic (correct, slow), array reduction reduction(+:h[:10]) (plain array only)
#include <iostream>
#include <vector>
#include <string>
#include <cstdlib>
#include <omp.h>
using namespace std;

int main() {
    long n = 10000000;
    vector<int> a(n);
    for (long i = 0; i < n; i++) a[i] = rand() % 100;      // marks 0..99

    long g1[10] = {0};
    double start = omp_get_wtime();
    for (long i = 0; i < n; i++) g1[a[i] / 10]++;
    double ts = omp_get_wtime() - start;

    for (int t = 1; t <= 8; t = t * 2) {
        omp_set_num_threads(t);

        long g2[10] = {0};
        start = omp_get_wtime();
        #pragma omp parallel
        {
            long local[10] = {0};
            #pragma omp for
            for (long i = 0; i < n; i++) local[a[i] / 10]++;
            #pragma omp critical
            {
                for (int k = 0; k < 10; k++) g2[k] += local[k];
            }
        }
        double tl = omp_get_wtime() - start;

        long g3[10] = {0};
        start = omp_get_wtime();
        #pragma omp parallel for
        for (long i = 0; i < n; i++) {
            #pragma omp atomic
            g3[a[i] / 10]++;
        }
        double ta = omp_get_wtime() - start;

        long g4[10] = {0};
        #pragma omp parallel for reduction(+:g4[:10])
        for (long i = 0; i < n; i++) g4[a[i] / 10]++;

        bool ok = true;
        for (int k = 0; k < 10; k++) if (g2[k] != g1[k] || g3[k] != g1[k] || g4[k] != g1[k]) ok = false;
        cout << "threads = " << t << "  local = " << tl << " (speedup " << ts / tl << ")  atomic = " << ta
             << "  " << (ok ? "MATCH" : "MISMATCH") << "\n";
    }
    for (int k = 0; k < 10; k++) cout << k * 10 << "-" << k * 10 + 9 << ": " << g1[k] << "  ";
    cout << "\n";

    // letter count
    string text;
    for (int r = 0; r < 100000; r++) text += "parallel programming with openmp ";
    long L = text.size(), c1[26] = {0}, c2[26] = {0};
    for (long i = 0; i < L; i++) if (text[i] >= 'a' && text[i] <= 'z') c1[text[i] - 'a']++;
    #pragma omp parallel
    {
        long local[26] = {0};
        #pragma omp for
        for (long i = 0; i < L; i++) if (text[i] >= 'a' && text[i] <= 'z') local[text[i] - 'a']++;
        #pragma omp critical
        {
            for (int k = 0; k < 26; k++) c2[k] += local[k];
        }
    }
    bool ok = true;
    for (int k = 0; k < 26; k++) if (c1[k] != c2[k]) ok = false;
    cout << "letters: p = " << c2['p' - 'a'] << ", r = " << c2['r' - 'a'] << "  " << (ok ? "MATCH" : "MISMATCH") << "\n";
}
