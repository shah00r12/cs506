// PROBLEM: parallel selection sort and parallel insertion sort
// SELECTION: rounds are serial; finding the min of a[i..n-1] + its position is parallel
//            (own copy + critical, every round)
// INSERTION: NOT parallel directly (inserting a[i] needs a[0..i-1] sorted).
//            workaround: insertion-sort CHUNKS in parallel, then merge chunks pairwise (parallel per pair)
#include <iostream>
#include <vector>
#include <climits>
#include <cstdlib>
#include <algorithm>
#include <omp.h>
using namespace std;

void insertionSort(vector<int> &a, int lo, int hi) {
    for (int i = lo + 1; i < hi; i++) {
        int key = a[i], j = i - 1;
        while (j >= lo && a[j] > key) { a[j + 1] = a[j]; j--; }
        a[j + 1] = key;
    }
}
void mergeRange(vector<int> &a, vector<int> &tmp, int lo, int mid, int hi) {
    int i = lo, j = mid, k = lo;
    while (i < mid && j < hi) tmp[k++] = (a[i] <= a[j]) ? a[i++] : a[j++];
    while (i < mid) tmp[k++] = a[i++];
    while (j < hi) tmp[k++] = a[j++];
    for (int t = lo; t < hi; t++) a[t] = tmp[t];
}

int main() {
    int n = 15000;
    vector<int> a(n);
    for (int i = 0; i < n; i++) a[i] = rand() % 100000;
    vector<int> ref = a;
    sort(ref.begin(), ref.end());

    // selection sort
    vector<int> p = a;
    double st = omp_get_wtime();
    for (int i = 0; i < n - 1; i++) {
        int best = INT_MAX, pos = -1;
        #pragma omp parallel
        {
            int myBest = INT_MAX, myPos = -1;
            #pragma omp for
            for (int j = i; j < n; j++)
                if (p[j] < myBest) { myBest = p[j]; myPos = j; }
            #pragma omp critical
            {
                if (myBest < best || (myBest == best && myPos < pos)) { best = myBest; pos = myPos; }
            }
        }
        swap(p[i], p[pos]);
    }
    cout << "selection sort: time = " << omp_get_wtime() - st << "  " << (p == ref ? "MATCH" : "MISMATCH") << "\n";

    // insertion sort with chunks
    int chunks = 16;
    vector<int> q = a, tmp(n);
    st = omp_get_wtime();
    #pragma omp parallel for
    for (int c = 0; c < chunks; c++)
        insertionSort(q, (long)n * c / chunks, (long)n * (c + 1) / chunks);
    for (int w = 1; w < chunks; w *= 2) {
        #pragma omp parallel for
        for (int c = 0; c < chunks; c += 2 * w) {
            int lo = (long)n * c / chunks;
            int mid = (long)n * min(c + w, chunks) / chunks;
            int hi = (long)n * min(c + 2 * w, chunks) / chunks;
            mergeRange(q, tmp, lo, mid, hi);
        }
    }
    cout << "insertion (chunks + merge): time = " << omp_get_wtime() - st << "  " << (q == ref ? "MATCH" : "MISMATCH") << "\n";
}
