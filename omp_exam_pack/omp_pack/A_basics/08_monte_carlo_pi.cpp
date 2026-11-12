// PROBLEM: estimate pi by throwing random points in a 1x1 square; pi ~ 4 * (inside circle) / total
// PATTERN: one counter -> reduction(+:inside)
// TRAP: rand() is NOT thread-safe (shared hidden state -> slow and not truly random per thread).
//       give EACH thread its own random generator with its own seed (simple LCG below).
// CHECK: random -> answers differ a little; check |pi - 3.14159| < 0.01
// ALSO: pi by integration of 4/(1+x^2) (deterministic)
#include <iostream>
#include <cmath>
#include <omp.h>
using namespace std;

double myRand(unsigned long &seed) {              // simple random number generator, 0..1
    seed = seed * 6364136223846793005UL + 1442695040888963407UL;
    return (seed >> 11) * (1.0 / 9007199254740992.0);
}

int main() {
    long N = 50000000;

    unsigned long sd = 12345;
    long in1 = 0;
    double start = omp_get_wtime();
    for (long i = 0; i < N; i++) {
        double x = myRand(sd), y = myRand(sd);
        if (x * x + y * y <= 1.0) in1++;
    }
    double ts = omp_get_wtime() - start;
    cout << "serial pi = " << 4.0 * in1 / N << "  time = " << ts << "\n";

    for (int t = 1; t <= 8; t = t * 2) {
        omp_set_num_threads(t);
        long inside = 0;
        start = omp_get_wtime();
        #pragma omp parallel reduction(+:inside)
        {
            unsigned long seed = 12345 + 1000 * omp_get_thread_num();   // own seed
            #pragma omp for
            for (long i = 0; i < N; i++) {
                double x = myRand(seed), y = myRand(seed);
                if (x * x + y * y <= 1.0) inside++;
            }
        }
        double tp = omp_get_wtime() - start;
        double pi = 4.0 * inside / N;
        cout << "threads = " << t << "  pi = " << pi << "  time = " << tp << "  speedup = " << ts / tp
             << "  " << (fabs(pi - 3.14159265) < 0.01 ? "MATCH" : "MISMATCH") << "\n";
    }

    long steps = 100000000;
    double h = 1.0 / steps, s = 0;
    #pragma omp parallel for reduction(+:s)
    for (long i = 0; i < steps; i++) {
        double x = (i + 0.5) * h;
        s += 4.0 / (1.0 + x * x);
    }
    cout << "integration pi = " << s * h << "\n";
}
