// PROBLEM: BFS on a CSR graph: distance (hops) from source node 0 to every node
// SERIAL: queue
// PARALLEL: level by level (levels are SERIAL). In level L, every UNVISITED node v checks:
//   "is any neighbour at distance L?" -> then v is at L+1.
//   Read OLD dist, write NEW dist (own box nd[v]) -> no race. changed -> reduction(||)
//   break is OK in the INNER loop (only the outer loop is split)
#include <iostream>
#include <vector>
#include <cstdlib>
#include <omp.h>
using namespace std;

void buildCSR(int n, vector<int> &eu, vector<int> &ev, vector<int> &rowPtr, vector<int> &colIdx) {
    vector<int> deg(n, 0);
    for (size_t e = 0; e < eu.size(); e++) { deg[eu[e]]++; deg[ev[e]]++; }
    rowPtr.assign(n + 1, 0);
    for (int u = 0; u < n; u++) rowPtr[u + 1] = rowPtr[u] + deg[u];
    colIdx.assign(rowPtr[n], 0);
    vector<int> pos(rowPtr.begin(), rowPtr.end() - 1);
    for (size_t e = 0; e < eu.size(); e++) { colIdx[pos[eu[e]]++] = ev[e]; colIdx[pos[ev[e]]++] = eu[e]; }
}

int main() {
    int n = 1000000, m = 4000000;
    vector<int> eu(m), ev(m), rowPtr, colIdx;
    for (int e = 0; e < m; e++) { eu[e] = rand() % n; ev[e] = rand() % n; }
    buildCSR(n, eu, ev, rowPtr, colIdx);

    vector<int> d1(n, -1), q(n);
    int head = 0, tail = 0;
    d1[0] = 0; q[tail++] = 0;
    double st = omp_get_wtime();
    while (head < tail) {
        int u = q[head++];
        for (int k = rowPtr[u]; k < rowPtr[u + 1]; k++) {
            int v = colIdx[k];
            if (d1[v] == -1) { d1[v] = d1[u] + 1; q[tail++] = v; }
        }
    }
    double ts = omp_get_wtime() - st;
    cout << "serial BFS time = " << ts << "\n";

    for (int t = 1; t <= 8; t = t * 2) {
        omp_set_num_threads(t);
        vector<int> d(n, -1), nd(n);
        d[0] = 0;
        int level = 0;
        bool changed = true;
        st = omp_get_wtime();
        while (changed) {
            changed = false;
            #pragma omp parallel for schedule(dynamic, 256) reduction(||:changed)
            for (int v = 0; v < n; v++) {
                nd[v] = d[v];
                if (d[v] == -1) {
                    for (int k = rowPtr[v]; k < rowPtr[v + 1]; k++)
                        if (d[colIdx[k]] == level) { nd[v] = level + 1; changed = true; break; }
                }
            }
            swap(d, nd);
            level++;
        }
        double tp = omp_get_wtime() - st;
        int reached = 0;
        for (int v = 0; v < n; v++) if (d[v] != -1) reached++;
        cout << "threads = " << t << "  levels = " << level - 1 << "  reached = " << reached << "  time = " << tp
             << "  " << (d == d1 ? "MATCH" : "MISMATCH") << "\n";
    }
}
