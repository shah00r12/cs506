// PROBLEM: store a graph in CSR, then: degree of each node, max degree, total edges,
//          sum of neighbour values (SpMV y = A x), weighted SpMV
// CSR:  rowPtr[n+1], colIdx[m] (+ val[m] for weights)
//       neighbours of u are colIdx[rowPtr[u]] .. colIdx[rowPtr[u+1]-1]
//   example 0-1, 0-2, 1-2, 2-3:  rowPtr = 0 2 4 7 8   colIdx = 1 2 0 2 0 1 3 2
// THE LOOP:  for u (nodes) { for k = rowPtr[u] .. rowPtr[u+1]-1 { v = colIdx[k]; ... } }
// PARALLEL: split the NODE loop -> each u writes its own box. schedule(dynamic) (uneven degrees).
//           NEVER collapse (inner length depends on u).
// BUILD: count degrees -> prefix sum gives rowPtr -> place each edge (serial; easy & safe)
#include <iostream>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <omp.h>
using namespace std;

void buildCSR(int n, vector<int> &eu, vector<int> &ev, vector<int> &rowPtr, vector<int> &colIdx) {
    vector<int> deg(n, 0);
    for (size_t e = 0; e < eu.size(); e++) { deg[eu[e]]++; deg[ev[e]]++; }
    rowPtr.assign(n + 1, 0);
    for (int u = 0; u < n; u++) rowPtr[u + 1] = rowPtr[u] + deg[u];
    colIdx.assign(rowPtr[n], 0);
    vector<int> pos(rowPtr.begin(), rowPtr.end() - 1);
    for (size_t e = 0; e < eu.size(); e++) {
        colIdx[pos[eu[e]]++] = ev[e];
        colIdx[pos[ev[e]]++] = eu[e];
    }
}

int main() {
    int n = 4;
    vector<int> eu = {0, 0, 1, 2}, ev = {1, 2, 2, 3}, rowPtr, colIdx;
    buildCSR(n, eu, ev, rowPtr, colIdx);
    cout << "rowPtr: "; for (int x : rowPtr) cout << x << " ";
    cout << "\ncolIdx: "; for (int x : colIdx) cout << x << " ";
    cout << "\n";

    n = 1000000;
    int m = 5000000;
    eu.resize(m); ev.resize(m);
    for (int e = 0; e < m; e++) { eu[e] = rand() % n; ev[e] = rand() % n; }
    buildCSR(n, eu, ev, rowPtr, colIdx);
    vector<double> val(colIdx.size());
    for (size_t k = 0; k < val.size(); k++) val[k] = (k % 5) + 1;      // edge weights

    vector<int> deg(n);
    int maxDeg = 0;
    long total = 0;
    #pragma omp parallel for reduction(max:maxDeg) reduction(+:total)
    for (int u = 0; u < n; u++) {
        deg[u] = rowPtr[u + 1] - rowPtr[u];
        if (deg[u] > maxDeg) maxDeg = deg[u];
        total += deg[u];
    }
    cout << "nodes = " << n << " edges = " << m << " max degree = " << maxDeg << " total degree = " << total
         << (total == 2L * m ? "  MATCH" : "  MISMATCH") << "\n";

    vector<double> x(n), y1(n), y2(n);
    for (int u = 0; u < n; u++) x[u] = u % 10;
    double st = omp_get_wtime();
    for (int u = 0; u < n; u++) {
        double s = 0;
        for (int k = rowPtr[u]; k < rowPtr[u + 1]; k++) s += val[k] * x[colIdx[k]];
        y1[u] = s;
    }
    double ts = omp_get_wtime() - st;
    for (int t = 1; t <= 8; t = t * 2) {
        omp_set_num_threads(t);
        st = omp_get_wtime();
        #pragma omp parallel for schedule(dynamic, 64)
        for (int u = 0; u < n; u++) {
            double s = 0;
            for (int k = rowPtr[u]; k < rowPtr[u + 1]; k++) s += val[k] * x[colIdx[k]];
            y2[u] = s;
        }
        double tp = omp_get_wtime() - st;
        bool ok = true;
        for (int u = 0; u < n; u++) if (fabs(y1[u] - y2[u]) > 1e-9) ok = false;
        cout << "SpMV threads = " << t << "  time = " << tp << "  speedup = " << ts / tp << "  " << (ok ? "MATCH" : "MISMATCH") << "\n";
    }
}
