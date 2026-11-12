// PROBLEM: largest value AND its position (first occurrence); smallest value AND position
// PATTERN: value + position are DEPENDENT -> own copy per thread + omp for + critical merge once
// TIE-BREAK: equal value -> keep the smaller index (same as serial)
// TRAPS: { } on the if (2 lines), { } after critical
#include <iostream>
#include <vector>
#include <climits>
#include <omp.h>
using namespace std;

int main() {
    long n = 10000000;
    vector<long> a(n);
    for (long i = 0; i < n; i++) a[i] = i % 100 + 10;
    a[5555555] = 500; a[8888888] = 500;      // max twice -> answer 5555555
    a[1234567] = 1;                          // min

    long big1 = LONG_MIN, bp1 = -1, sm1 = LONG_MAX, sp1 = -1;
    for (long i = 0; i < n; i++) {
        if (a[i] > big1) { big1 = a[i]; bp1 = i; }
        if (a[i] < sm1)  { sm1 = a[i];  sp1 = i; }
    }

    for (int t = 1; t <= 8; t = t * 2) {
        omp_set_num_threads(t);
        long big = LONG_MIN, bp = -1, sm = LONG_MAX, sp = -1;
        double start = omp_get_wtime();
        #pragma omp parallel
        {
            long myBig = LONG_MIN, myBp = -1, mySm = LONG_MAX, mySp = -1;
            #pragma omp for
            for (long i = 0; i < n; i++) {
                if (a[i] > myBig) { myBig = a[i]; myBp = i; }
                if (a[i] < mySm)  { mySm = a[i];  mySp = i; }
            }
            #pragma omp critical
            {
                if (myBig > big || (myBig == big && myBp < bp)) { big = myBig; bp = myBp; }
                if (mySm < sm || (mySm == sm && mySp < sp))     { sm = mySm;  sp = mySp; }
            }
        }
        double tp = omp_get_wtime() - start;
        bool ok = big == big1 && bp == bp1 && sm == sm1 && sp == sp1;
        cout << "threads = " << t << "  max " << big << " at " << bp << "  min " << sm << " at " << sp
             << "  time = " << tp << "  " << (ok ? "MATCH" : "MISMATCH") << "\n";
    }
}
