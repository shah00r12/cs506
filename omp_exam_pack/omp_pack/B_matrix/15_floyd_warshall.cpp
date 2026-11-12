// PROBLEM: all-pairs shortest paths (Floyd-Warshall): dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
// k loop: SERIAL (step k needs step k-1)
// i, j loops: PARALLEL collapse(2). Safe in place: in step k, row k and column k do NOT change
//   (dist[i][k] + dist[k][k] = dist[i][k] since dist[k][k] = 0), so nobody reads a value being changed.
#include <iostream>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <omp.h>
using namespace std;

const int INF = 1000000000;

int main() {
    int n = 800;
    vector<int> g(n * n);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            if (i == j) g[i * n + j] = 0;
            else if (rand() % 10 == 0) g[i * n + j] = rand() % 100 + 1;   // ~10% edges
            else g[i * n + j] = INF;
        }

    vector<int> d1 = g;
    double start = omp_get_wtime();
    for (int k = 0; k < n; k++)
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                if (d1[i * n + k] < INF && d1[k * n + j] < INF)
                    d1[i * n + j] = min(d1[i * n + j], d1[i * n + k] + d1[k * n + j]);
    double ts = omp_get_wtime() - start;
    cout << "N = " << n << "  serial time = " << ts << "\n";

    for (int t = 1; t <= 8; t = t * 2) {
        omp_set_num_threads(t);
        vector<int> d2 = g;
        start = omp_get_wtime();
        for (int k = 0; k < n; k++) {
            #pragma omp parallel for collapse(2)
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                    if (d2[i * n + k] < INF && d2[k * n + j] < INF)
                        d2[i * n + j] = min(d2[i * n + j], d2[i * n + k] + d2[k * n + j]);
        }
        double tp = omp_get_wtime() - start;
        cout << "threads = " << t << "  time = " << tp << "  speedup = " << ts / tp
             << "  efficiency = " << ts / tp / t << "  " << (d1 == d2 ? "MATCH" : "MISMATCH") << "\n";
    }
    cout << "dist(0, 5) = " << d1[5] << "\n";
}
