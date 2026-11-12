// PROBLEM: connected components of a graph (CSR) - how many separate pieces?
// LABEL PROPAGATION: label[v] = v. Repeat: new label[v] = min(own label, labels of neighbours)
//   until nothing changes. Nodes in one component end up with the same (smallest) label.
// PATTERN: old -> new labels (own box), changed = reduction(||), rounds serial
// CHECK: number of components vs serial BFS count
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
    int n = 500000, m = 300000;                  // few edges -> many components
    vector<int> eu(m), ev(m), rowPtr, colIdx;
    for (int e = 0; e < m; e++) { eu[e] = rand() % n; ev[e] = rand() % n; }
    buildCSR(n, eu, ev, rowPtr, colIdx);

    // serial: BFS from every unvisited node
    vector<int> seen(n, 0), q(n);
    int comps1 = 0;
    for (int s = 0; s < n; s++) {
        if (seen[s]) continue;
        comps1++;
        int h = 0, tl = 0;
        q[tl++] = s; seen[s] = 1;
        while (h < tl) {
            int u = q[h++];
            for (int k = rowPtr[u]; k < rowPtr[u + 1]; k++)
                if (!seen[colIdx[k]]) { seen[colIdx[k]] = 1; q[tl++] = colIdx[k]; }
        }
    }

    vector<int> label(n), nl(n);
    for (int v = 0; v < n; v++) label[v] = v;
    bool changed = true;
    int rounds = 0;
    double st = omp_get_wtime();
    while (changed) {
        changed = false;
        #pragma omp parallel for schedule(dynamic, 256) reduction(||:changed)
        for (int v = 0; v < n; v++) {
            int best = label[v];
            for (int k = rowPtr[v]; k < rowPtr[v + 1]; k++)
                if (label[colIdx[k]] < best) best = label[colIdx[k]];
            nl[v] = best;
            if (best != label[v]) changed = true;
        }
        swap(label, nl);
        rounds++;
    }
    double tp = omp_get_wtime() - st;
    int comps2 = 0;
    #pragma omp parallel for reduction(+:comps2)
    for (int v = 0; v < n; v++) if (label[v] == v) comps2++;
    cout << "components: serial BFS = " << comps1 << ", label propagation = " << comps2
         << " (" << rounds << " rounds, " << tp << " s)  " << (comps1 == comps2 ? "MATCH" : "MISMATCH") << "\n";
}
