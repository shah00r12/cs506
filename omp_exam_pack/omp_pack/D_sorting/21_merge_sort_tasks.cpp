// PROBLEM: parallel merge sort with tasks
// base case -> cutoff (small: serial) -> task(left) -> task(right) -> taskwait -> merge
// start: parallel + single.  shared(a, tmp), firstprivate(left, mid / mid, right)
#include <iostream>
#include <vector>
#include <cstdlib>
#include <omp.h>
using namespace std;

void merge(vector<int> &a, vector<int> &tmp, int l, int m, int r) {
    int i = l, j = m + 1, k = l;
    while (i <= m && j <= r) tmp[k++] = (a[i] <= a[j]) ? a[i++] : a[j++];
    while (i <= m) tmp[k++] = a[i++];
    while (j <= r) tmp[k++] = a[j++];
    for (int t = l; t <= r; t++) a[t] = tmp[t];
}
void msSerial(vector<int> &a, vector<int> &tmp, int l, int r) {
    if (l >= r) return;
    int m = l + (r - l) / 2;
    msSerial(a, tmp, l, m);
    msSerial(a, tmp, m + 1, r);
    merge(a, tmp, l, m, r);
}
void msParallel(vector<int> &a, vector<int> &tmp, int l, int r) {
    if (l >= r) return;
    if (r - l < 2048) { msSerial(a, tmp, l, r); return; }
    int m = l + (r - l) / 2;
    #pragma omp task shared(a, tmp) firstprivate(l, m)
    msParallel(a, tmp, l, m);
    #pragma omp task shared(a, tmp) firstprivate(m, r)
    msParallel(a, tmp, m + 1, r);
    #pragma omp taskwait
    merge(a, tmp, l, m, r);
}

int main() {
    int sizes[] = {100000, 1000000, 10000000};
    for (int s = 0; s < 3; s++) {
        int n = sizes[s];
        vector<int> a(n), tmp(n);
        for (int i = 0; i < n; i++) a[i] = rand();
        vector<int> ser = a;
        double st = omp_get_wtime();
        msSerial(ser, tmp, 0, n - 1);
        double ts = omp_get_wtime() - st;
        cout << "\nN = " << n << "  serial = " << ts << "\n";
        for (int t = 1; t <= 8; t = t * 2) {
            omp_set_num_threads(t);
            vector<int> p = a;
            st = omp_get_wtime();
            #pragma omp parallel
            {
                #pragma omp single
                msParallel(p, tmp, 0, n - 1);
            }
            double tp = omp_get_wtime() - st;
            cout << "threads = " << t << "  time = " << tp << "  speedup = " << ts / tp
                 << "  efficiency = " << ts / tp / t << "  " << (p == ser ? "MATCH" : "MISMATCH") << "\n";
        }
    }
}
