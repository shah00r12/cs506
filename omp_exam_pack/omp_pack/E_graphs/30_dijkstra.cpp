// PROBLEM: Dijkstra shortest paths from node 0 (dense graph, adjacency matrix, INF = no edge)
// Same shape as Prim:
//   1. pick unvisited node u with the smallest dist      -> own copy + critical (min + position)
//   2. mark it visited
//   3. relax: dist[v] = min(dist[v], dist[u] + w(u,v))   -> each v own box -> parallel for
// steps SERIAL, inside PARALLEL.
#include <iostream>
#include <vector>
#include <climits>
#include <cstdlib>
#include <omp.h>
using namespace std;

const int INF = INT_MAX;

vector<long> dijkstra(int n, vector<int> &w, bool par) {
    vector<long> dist(n, LONG_MAX);
    vector<int> done(n, 0);
    dist[0] = 0;
    for (int step = 0; step < n; step++) {
        long best = LONG_MAX;
        int u = -1;
        #pragma omp parallel if(par)
        {
            long myBest = LONG_MAX;
            int myU = -1;
            #pragma omp for
            for (int v = 0; v < n; v++)
                if (!done[v] && dist[v] < myBest) { myBest = dist[v]; myU = v; }
            #pragma omp critical
            {
                if (myU != -1 && (myBest < best || (myBest == best && myU < u))) { best = myBest; u = myU; }
            }
        }
        if (u == -1) break;                       // the rest can't be reached
        done[u] = 1;
        #pragma omp parallel for if(par)
        for (int v = 0; v < n; v++)
            if (!done[v] && w[u * n + v] != INF && dist[u] + w[u * n + v] < dist[v])
                dist[v] = dist[u] + w[u * n + v];
    }
    return dist;
}

int main() {
    int n = 4000;
    vector<int> w(n * n);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            w[i * n + j] = (i == j) ? 0 : (rand() % 5 == 0 ? rand() % 100 + 1 : INF);   // ~20% edges

    double st = omp_get_wtime();
    vector<long> d1 = dijkstra(n, w, false);
    double ts = omp_get_wtime() - st;
    cout << "N = " << n << "  serial time = " << ts << "  dist(0, 10) = " << d1[10] << "\n";
    for (int t = 1; t <= 8; t = t * 2) {
        omp_set_num_threads(t);
        st = omp_get_wtime();
        vector<long> d2 = dijkstra(n, w, true);
        double tp = omp_get_wtime() - st;
        cout << "threads = " << t << "  time = " << tp << "  speedup = " << ts / tp << "  " << (d1 == d2 ? "MATCH" : "MISMATCH") << "\n";
    }
}
