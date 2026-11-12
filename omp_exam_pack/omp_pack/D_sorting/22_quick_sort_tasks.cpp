// PROBLEM: parallel quick sort with tasks
// partition FIRST (serial, i = low - 1 = end of the 'small' area), then task(left side), task(right side), taskwait
// merge sort: work AFTER split (merge). quick sort: work BEFORE split (partition).
#include <iostream>
#include <vector>
#include <cstdlib>
#include <omp.h>
using namespace std;

int partitionArr(vector<int> &a, int low, int high) {
    int pivot = a[high], i = low - 1;
    for (int j = low; j < high; j++)
        if (a[j] < pivot) { i++; swap(a[i], a[j]); }
    swap(a[i + 1], a[high]);
    return i + 1;
}
void qsSerial(vector<int> &a, int lo, int hi) {
    if (lo >= hi) return;
    int p = partitionArr(a, lo, hi);
    qsSerial(a, lo, p - 1);
    qsSerial(a, p + 1, hi);
}
void qsParallel(vector<int> &a, int lo, int hi) {
    if (lo >= hi) return;
    if (hi - lo < 2048) { qsSerial(a, lo, hi); return; }
    int p = partitionArr(a, lo, hi);
    #pragma omp task shared(a) firstprivate(lo, p)
    qsParallel(a, lo, p - 1);
    #pragma omp task shared(a) firstprivate(p, hi)
    qsParallel(a, p + 1, hi);
    #pragma omp taskwait
}

int main() {
    int sizes[] = {100000, 1000000, 10000000};
    for (int s = 0; s < 3; s++) {
        int n = sizes[s];
        vector<int> a(n);
        for (int i = 0; i < n; i++) a[i] = rand();
        vector<int> ser = a;
        double st = omp_get_wtime();
        qsSerial(ser, 0, n - 1);
        double ts = omp_get_wtime() - st;
        cout << "\nN = " << n << "  serial = " << ts << "\n";
        for (int t = 1; t <= 8; t = t * 2) {
            omp_set_num_threads(t);
            vector<int> p = a;
            st = omp_get_wtime();
            #pragma omp parallel
            {
                #pragma omp single
                qsParallel(p, 0, n - 1);
            }
            double tp = omp_get_wtime() - st;
            cout << "threads = " << t << "  time = " << tp << "  speedup = " << ts / tp << "  " << (p == ser ? "MATCH" : "MISMATCH") << "\n";
        }
    }
}
