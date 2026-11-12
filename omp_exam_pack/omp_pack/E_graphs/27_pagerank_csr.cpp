// PROBLEM: PageRank on a CSR graph (undirected here: in-neighbours = neighbours)
//   pr_new[v] = (1 - d) / n + d * sum over neighbours u of pr[u] / degree(u)
//   repeat until the total change < tolerance
// PATTERN: each v writes its own pr_new[v] (old -> new arrays); change = reduction(+:diff)
//   iterations are SERIAL; schedule(dynamic) for uneven degrees
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
    for (size_t e = 0; e < eu.size(); e++) { colIdx[pos[eu[e]]++] = ev[e]; colIdx[pos[ev[e]]++] = eu[e]; }
}

int pagerank(int n, vector<int> &rowPtr, vector<int> &colIdx, vector<double> &pr, bool par) {
    double d = 0.85, tol = 1e-8;
    pr.assign(n, 1.0 / n);
    vector<double> np(n);
    int it = 0;
    double diff = 1;
    while (diff > tol && it < 100) {
        diff = 0;
        #pragma omp parallel for schedule(dynamic, 256) reduction(+:diff) if(par)
        for (int v = 0; v < n; v++) {
            double s = 0;
            for (int k = rowPtr[v]; k < rowPtr[v + 1]; k++) {
                int u = colIdx[k];
                s += pr[u] / (rowPtr[u + 1] - rowPtr[u]);
            }
            np[v] = (1 - d) / n + d * s;
            diff += fabs(np[v] - pr[v]);
        }
        swap(pr, np);
        it++;
    }
    return it;
}

int main() {
    int n = 500000, m = 3000000;
    vector<int> eu(m), ev(m), rowPtr, colIdx;
    for (int e = 0; e < m; e++) { eu[e] = rand() % n; ev[e] = rand() % n; }
    buildCSR(n, eu, ev, rowPtr, colIdx);

    vector<double> p1, p2;
    double st = omp_get_wtime();
    int it1 = pagerank(n, rowPtr, colIdx, p1, false);
    double ts = omp_get_wtime() - st;
    int top = 0;
    for (int v = 1; v < n; v++) if (p1[v] > p1[top]) top = v;
    cout << "serial: " << it1 << " iterations, time = " << ts << ", top node = " << top << "\n";

    for (int t = 1; t <= 8; t = t * 2) {
        omp_set_num_threads(t);
        st = omp_get_wtime();
        int it2 = pagerank(n, rowPtr, colIdx, p2, true);
        double tp = omp_get_wtime() - st;
        double err = 0;
        for (int v = 0; v < n; v++) err = max(err, fabs(p1[v] - p2[v]));
        cout << "threads = " << t << "  iterations = " << it2 << "  time = " << tp << "  speedup = " << ts / tp
             << "  " << (err < 1e-9 ? "MATCH" : "MISMATCH") << "\n";
    }
}
