// PROBLEM: Prim's minimum spanning tree (dense graph, adjacency matrix)
// Prim: start at node 0. Repeat n times:
//   1. pick the node NOT in the tree with the smallest key   -> value + position = own copy + critical
//   2. add it to the tree
//   3. update key[v] = min(key[v], w(u, v)) for nodes not in the tree -> each v own box -> parallel for
// The n steps are SERIAL (each needs the previous); steps 1 and 3 inside are PARALLEL.
#include <iostream>
#include <vector>
#include <climits>
#include <cstdlib>
#include <omp.h>
using namespace std;

const int INF = INT_MAX;

long prim(int n, vector<int> &w, bool par) {
    vector<int> key(n, INF), inTree(n, 0);
    key[0] = 0;
    long total = 0;
    for (int step = 0; step < n; step++) {
        int best = INF, u = -1;
        #pragma omp parallel if(par)
        {
            int myBest = INF, myU = -1;
            #pragma omp for
            for (int v = 0; v < n; v++)
                if (!inTree[v] && key[v] < myBest) { myBest = key[v]; myU = v; }
            #pragma omp critical
            {
                if (myBest < best || (myBest == best && myU < u && myU != -1)) { best = myBest; u = myU; }
            }
        }
        if (u == -1) break;                         // graph not connected
        inTree[u] = 1;
        total += key[u];
        #pragma omp parallel for if(par)
        for (int v = 0; v < n; v++)
            if (!inTree[v] && w[u * n + v] < key[v]) key[v] = w[u * n + v];
    }
    return total;
}

int main() {
    int n = 4000;
    vector<int> w(n * n);
    for (int i = 0; i < n; i++)
        for (int j = i; j < n; j++) {
            int x = (i == j) ? 0 : rand() % 1000 + 1;
            w[i * n + j] = x; w[j * n + i] = x;          // undirected
        }

    double st = omp_get_wtime();
    long t1 = prim(n, w, false);
    double ts = omp_get_wtime() - st;
    cout << "N = " << n << "  serial MST weight = " << t1 << "  time = " << ts << "\n";
    for (int t = 1; t <= 8; t = t * 2) {
        omp_set_num_threads(t);
        st = omp_get_wtime();
        long t2 = prim(n, w, true);
        double tp = omp_get_wtime() - st;
        cout << "threads = " << t << "  weight = " << t2 << "  time = " << tp << "  speedup = " << ts / tp
             << "  " << (t1 == t2 ? "MATCH" : "MISMATCH") << "\n";
    }
}
