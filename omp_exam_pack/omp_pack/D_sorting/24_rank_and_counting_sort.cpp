// PROBLEM 1: RANK (enumeration) sort - fully parallel, no dependency
//   rank[i] = how many elements are smaller than a[i] (ties: count equal ones with smaller index)
//   out[rank[i]] = a[i]  -> every i writes its own box. O(n^2) work, perfectly parallel.
// PROBLEM 2: COUNTING sort for small values 0..K-1
//   count: histogram (local array + critical)   start: prefix sum of counts (serial, K is small)
//   fill: for each value v (parallel): write v into out[start[v] .. start[v]+count[v]-1] (own range)
#include <iostream>
#include <vector>
#include <cstdlib>
#include <algorithm>
#include <omp.h>
using namespace std;

int main() {
    int n = 20000;
    vector<int> a(n);
    for (int i = 0; i < n; i++) a[i] = rand() % 1000;
    vector<int> ref = a;
    sort(ref.begin(), ref.end());

    vector<int> out(n);
    double st = omp_get_wtime();
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        int r = 0;
        for (int j = 0; j < n; j++)
            if (a[j] < a[i] || (a[j] == a[i] && j < i)) r++;
        out[r] = a[i];
    }
    cout << "rank sort: time = " << omp_get_wtime() - st << "  " << (out == ref ? "MATCH" : "MISMATCH") << "\n";

    long m = 20000000;
    int K = 1000;
    vector<int> b(m), out2(m);
    for (long i = 0; i < m; i++) b[i] = rand() % K;
    vector<long> cnt(K, 0), startPos(K, 0);
    st = omp_get_wtime();
    #pragma omp parallel
    {
        vector<long> local(K, 0);
        #pragma omp for
        for (long i = 0; i < m; i++) local[b[i]]++;
        #pragma omp critical
        {
            for (int v = 0; v < K; v++) cnt[v] += local[v];
        }
    }
    for (int v = 1; v < K; v++) startPos[v] = startPos[v - 1] + cnt[v - 1];
    #pragma omp parallel for schedule(dynamic, 16)
    for (int v = 0; v < K; v++)
        for (long k = 0; k < cnt[v]; k++) out2[startPos[v] + k] = v;
    double tc = omp_get_wtime() - st;
    vector<int> r2 = b;
    sort(r2.begin(), r2.end());
    cout << "counting sort of " << m << " values: time = " << tc << "  " << (out2 == r2 ? "MATCH" : "MISMATCH") << "\n";
}
