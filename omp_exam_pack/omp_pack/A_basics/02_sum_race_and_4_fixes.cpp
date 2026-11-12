// PROBLEM: parallel sum - show the race, then fix it 4 ways
// RACE: sum += a[i] is read-add-write; two threads read the same old value -> one update lost
// FIXES: reduction (best) | critical (correct, very slow) | atomic (correct, slow) | own copy + merge once (fast)
#include <iostream>
#include <vector>
#include <omp.h>
using namespace std;

int main() {
    long n = 10000000;
    vector<long> a(n);
    for (long i = 0; i < n; i++) a[i] = i % 100;
    omp_set_num_threads(4);

    long s0 = 0;
    for (long i = 0; i < n; i++) s0 += a[i];
    cout << "serial = " << s0 << "\n";

    long w = 0;
    #pragma omp parallel for
    for (long i = 0; i < n; i++) w += a[i];                 // RACE
    cout << "no protection = " << w << (w == s0 ? " (lucky)" : "  <-- WRONG") << "\n";

    double st = omp_get_wtime();
    long s1 = 0;
    #pragma omp parallel for reduction(+:s1)
    for (long i = 0; i < n; i++) s1 += a[i];
    double t1 = omp_get_wtime() - st;

    st = omp_get_wtime();
    long s2 = 0;
    #pragma omp parallel for
    for (long i = 0; i < n; i++) {
        #pragma omp critical
        s2 += a[i];
    }
    double t2 = omp_get_wtime() - st;

    st = omp_get_wtime();
    long s3 = 0;
    #pragma omp parallel for
    for (long i = 0; i < n; i++) {
        #pragma omp atomic
        s3 += a[i];
    }
    double t3 = omp_get_wtime() - st;

    st = omp_get_wtime();
    long s4 = 0;
    #pragma omp parallel
    {
        long my = 0;
        #pragma omp for
        for (long i = 0; i < n; i++) my += a[i];
        #pragma omp critical
        s4 += my;
    }
    double t4 = omp_get_wtime() - st;

    cout << "reduction = " << s1 << "  time " << t1 << "\n";
    cout << "critical  = " << s2 << "  time " << t2 << "\n";
    cout << "atomic    = " << s3 << "  time " << t3 << "\n";
    cout << "own copy  = " << s4 << "  time " << t4 << "\n";
    cout << ((s1 == s0 && s2 == s0 && s3 == s0 && s4 == s0) ? "ALL MATCH" : "MISMATCH") << "\n";
}
