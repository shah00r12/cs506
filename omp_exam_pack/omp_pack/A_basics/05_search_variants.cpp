// PROBLEM: (a) linear search: exists? first, last, count   (no break in parallel for!)
//          (b) search in a 2D matrix: first (row, col) in row-major order
//          (c) many binary searches on a sorted array: one query per turn -> own box
// PATTERNS: first -> reduction(min:pos) start n; last -> reduction(max) start -1; count -> reduction(+)
#include <iostream>
#include <vector>
#include <algorithm>
#include <omp.h>
using namespace std;

int binarySearch(vector<int> &s, int key) {
    int lo = 0, hi = (int)s.size() - 1;
    while (lo <= hi) {
        int mid = lo + (hi - lo) / 2;
        if (s[mid] == key) return mid;
        if (s[mid] < key) lo = mid + 1;
        else hi = mid - 1;
    }
    return -1;
}

int main() {
    // (a) linear search
    long n = 10000000;
    vector<long> a(n);
    for (long i = 0; i < n; i++) a[i] = i % 100;
    a[2000000] = 777; a[9000000] = 777;
    long key = 777;

    long first = n, last = -1, count = 0;
    #pragma omp parallel for reduction(min:first) reduction(max:last) reduction(+:count)
    for (long i = 0; i < n; i++) {
        if (a[i] == key) {
            count++;
            if (i < first) first = i;
            if (i > last) last = i;
        }
    }
    if (first == n) first = -1;
    cout << "(a) key " << key << ": first = " << first << " last = " << last << " count = " << count
         << ((first == 2000000 && last == 9000000 && count == 2) ? "  MATCH" : "  MISMATCH") << "\n";

    // (b) 2D search: flatten the position to one number i*C+j and take the min
    int R = 3000, C = 3000;
    vector<int> M((long)R * C);
    for (long k = 0; k < (long)R * C; k++) M[k] = k % 50;
    M[1500L * C + 77] = -9; M[2900L * C + 5] = -9;
    long best = (long)R * C;
    #pragma omp parallel for collapse(2) reduction(min:best)
    for (int i = 0; i < R; i++)
        for (int j = 0; j < C; j++)
            if (M[(long)i * C + j] == -9 && (long)i * C + j < best) best = (long)i * C + j;
    cout << "(b) 2D: found at row " << best / C << ", col " << best % C
         << ((best / C == 1500 && best % C == 77) ? "  MATCH" : "  MISMATCH") << "\n";

    // (c) many binary searches
    int m = 1000000;
    vector<int> sorted(m), q(m), r1(m), r2(m);
    for (int i = 0; i < m; i++) { sorted[i] = 2 * i; q[i] = (i * 37) % (2 * m); }
    for (int i = 0; i < m; i++) r1[i] = binarySearch(sorted, q[i]);
    #pragma omp parallel for
    for (int i = 0; i < m; i++) r2[i] = binarySearch(sorted, q[i]);       // own box
    cout << "(c) " << m << " binary searches: " << (r1 == r2 ? "MATCH" : "MISMATCH") << "\n";
}
