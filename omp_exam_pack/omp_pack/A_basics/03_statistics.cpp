// PROBLEM: min, max, sum, mean, standard deviation, count above mean
// PATTERN: independent values -> several reductions on one line
//   reduction(+:sum) reduction(max:mx) reduction(min:mn)
//   std dev needs the mean first -> second loop. (double)sum/n for decimals.
#include <iostream>
#include <vector>
#include <cmath>
#include <climits>
#include <omp.h>
using namespace std;

int main() {
    long n = 10000000;
    vector<long> a(n);
    for (long i = 0; i < n; i++) a[i] = (i * 7) % 1000;

    // serial
    long s1 = 0, mx1 = LONG_MIN, mn1 = LONG_MAX;
    for (long i = 0; i < n; i++) {
        s1 += a[i];
        if (a[i] > mx1) mx1 = a[i];
        if (a[i] < mn1) mn1 = a[i];
    }
    double mean1 = (double)s1 / n, sq1 = 0;
    for (long i = 0; i < n; i++) sq1 += (a[i] - mean1) * (a[i] - mean1);

    // parallel
    long s2 = 0, mx2 = LONG_MIN, mn2 = LONG_MAX;
    #pragma omp parallel for reduction(+:s2) reduction(max:mx2) reduction(min:mn2)
    for (long i = 0; i < n; i++) {
        s2 += a[i];
        if (a[i] > mx2) mx2 = a[i];
        if (a[i] < mn2) mn2 = a[i];
    }
    double mean2 = (double)s2 / n, sq2 = 0;
    long above = 0;
    #pragma omp parallel for reduction(+:sq2, above)
    for (long i = 0; i < n; i++) {
        sq2 += (a[i] - mean2) * (a[i] - mean2);
        if (a[i] > mean2) above++;
    }
    double sd2 = sqrt(sq2 / n);

    cout << "sum = " << s2 << "  min = " << mn2 << "  max = " << mx2 << "  mean = " << mean2
         << "  stddev = " << sd2 << "  above mean = " << above << "\n";
    bool ok = s1 == s2 && mx1 == mx2 && mn1 == mn2 && fabs(sqrt(sq1 / n) - sd2) < 1e-6;
    cout << (ok ? "MATCH" : "MISMATCH") << "\n";
}
