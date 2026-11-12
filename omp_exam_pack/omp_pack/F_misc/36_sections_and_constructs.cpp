// PROBLEM: run DIFFERENT independent jobs at the same time -> #pragma omp sections
//   + demo of single, master, barrier, nowait, firstprivate, lastprivate, thread ids
// sections: each section is one job, given to one thread. Use when the jobs are different functions.
#include <iostream>
#include <vector>
#include <climits>
#include <omp.h>
using namespace std;

int main() {
    long n = 20000000;
    vector<long> a(n);
    for (long i = 0; i < n; i++) a[i] = (i * 13) % 1000;

    long sum = 0, mx = LONG_MIN, evens = 0;
    double st = omp_get_wtime();
    #pragma omp parallel sections
    {
        #pragma omp section
        { for (long i = 0; i < n; i++) sum += a[i]; }
        #pragma omp section
        { for (long i = 0; i < n; i++) if (a[i] > mx) mx = a[i]; }
        #pragma omp section
        { for (long i = 0; i < n; i++) if (a[i] % 2 == 0) evens++; }
    }
    cout << "sections: sum = " << sum << " max = " << mx << " evens = " << evens
         << "  time = " << omp_get_wtime() - st << "\n";
    // each section writes its OWN variable -> no race

    omp_set_num_threads(4);
    int x = 10, last = -1;
    #pragma omp parallel firstprivate(x)
    {
        x += omp_get_thread_num();                  // own copy that STARTED at 10
        #pragma omp master
        cout << "master (thread 0) only, no waiting\n";
        #pragma omp barrier
        #pragma omp single
        cout << "single: one thread, others wait at the end\n";
        #pragma omp critical
        cout << "thread " << omp_get_thread_num() << " has x = " << x << "\n";
    }
    #pragma omp parallel for lastprivate(last)
    for (int i = 0; i < 100; i++) last = i;         // lastprivate: value from the LAST iteration (99)
    cout << "lastprivate gives " << last << " (99)\n";
}
