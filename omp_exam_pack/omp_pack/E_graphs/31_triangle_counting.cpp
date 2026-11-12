// PROBLEM: count triangles in a graph (CSR with SORTED neighbour lists)
// For every edge u < v, count common neighbours w > v (each triangle u < v < w counted once)
//   common neighbours of sorted lists = two-pointer merge
// PATTERN: one total -> reduction(+:count) over nodes u, schedule(dynamic) (very uneven work)
#include <iostream>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <omp.h>
using namespace std;

int main() {
    int n = 100000, m = 1000000;
    vector<vector<int>> adj(n);
    for (int e = 0; e < m; e++) {
        int a = rand() % n, b = rand() % n;
        if (a != b) { adj[a].push_back(b); adj[b].push_back(a); }
    }
    vector<int> rowPtr(n + 1, 0), colIdx;
    for (int u = 0; u < n; u++) {
        sort(adj[u].begin(), adj[u].end());
        adj[u].erase(unique(adj[u].begin(), adj[u].end()), adj[u].end());   // remove duplicate edges
        rowPtr[u + 1] = rowPtr[u] + adj[u].size();
        for (int v : adj[u]) colIdx.push_back(v);
    }

    long c1 = 0;
    double st = omp_get_wtime();
    for (int u = 0; u < n; u++)
        for (int k = rowPtr[u]; k < rowPtr[u + 1]; k++) {
            int v = colIdx[k];
            if (v <= u) continue;
            int i = rowPtr[u], j = rowPtr[v];
            while (i < rowPtr[u + 1] && j < rowPtr[v + 1]) {
                if (colIdx[i] < colIdx[j]) i++;
                else if (colIdx[i] > colIdx[j]) j++;
                else { if (colIdx[i] > v) c1++; i++; j++; }
            }
        }
    double ts = omp_get_wtime() - st;
    cout << "serial triangles = " << c1 << "  time = " << ts << "\n";

    for (int t = 1; t <= 8; t = t * 2) {
        omp_set_num_threads(t);
        long c2 = 0;
        st = omp_get_wtime();
        #pragma omp parallel for reduction(+:c2) schedule(dynamic, 64)
        for (int u = 0; u < n; u++)
            for (int k = rowPtr[u]; k < rowPtr[u + 1]; k++) {
                int v = colIdx[k];
                if (v <= u) continue;
                int i = rowPtr[u], j = rowPtr[v];
                while (i < rowPtr[u + 1] && j < rowPtr[v + 1]) {
                    if (colIdx[i] < colIdx[j]) i++;
                    else if (colIdx[i] > colIdx[j]) j++;
                    else { if (colIdx[i] > v) c2++; i++; j++; }
                }
            }
        double tp = omp_get_wtime() - st;
        cout << "threads = " << t << "  triangles = " << c2 << "  time = " << tp << "  speedup = " << ts / tp
             << "  " << (c1 == c2 ? "MATCH" : "MISMATCH") << "\n";
    }
}
