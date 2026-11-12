// PROBLEM: (a) count primes up to N by testing each number (uneven work -> schedule(dynamic))
//          (b) sieve of Eratosthenes: for each prime p, cross out multiples (inner crossing-out is parallel)
//          (c) twin primes count (p and p+2 both prime) -> reduction
// TRAP: in the sieve the outer p loop must stay SERIAL (you need to know p is prime first)
#include <iostream>
#include <vector>
#include <omp.h>
using namespace std;

bool isPrime(long x) {
    if (x < 2) return false;
    for (long d = 2; d * d <= x; d++) if (x % d == 0) return false;
    return true;
}

int main() {
    long N = 2000000;
    long c1 = 0;
    double start = omp_get_wtime();
    for (long x = 2; x <= N; x++) if (isPrime(x)) c1++;
    double ts = omp_get_wtime() - start;

    for (int t = 1; t <= 8; t = t * 2) {
        omp_set_num_threads(t);
        long c2 = 0;
        start = omp_get_wtime();
        #pragma omp parallel for reduction(+:c2) schedule(dynamic, 100)
        for (long x = 2; x <= N; x++) if (isPrime(x)) c2++;
        double tp = omp_get_wtime() - start;
        cout << "(a) threads = " << t << "  primes = " << c2 << "  time = " << tp << "  speedup = " << ts / tp
             << "  " << (c1 == c2 ? "MATCH" : "MISMATCH") << "\n";
    }

    // (b) sieve. vector<char>, NOT vector<bool> (bool packs bits -> race between threads)
    vector<char> comp(N + 1, 0);
    for (long p = 2; p * p <= N; p++) {                  // serial
        if (!comp[p]) {
            #pragma omp parallel for
            for (long m = p * p; m <= N; m += p)          // own boxes
                comp[m] = 1;
        }
    }
    long c3 = 0, twins = 0;
    #pragma omp parallel for reduction(+:c3, twins)
    for (long x = 2; x <= N; x++) {
        if (!comp[x]) c3++;
        if (x + 2 <= N && !comp[x] && !comp[x + 2]) twins++;
    }
    cout << "(b) sieve primes = " << c3 << "  " << (c3 == c1 ? "MATCH" : "MISMATCH") << "\n";
    cout << "(c) twin prime pairs = " << twins << "\n";
}
