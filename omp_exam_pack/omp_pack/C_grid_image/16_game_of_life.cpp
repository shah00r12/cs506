// PROBLEM: Game of Life. 2D grid of random 0 (dead) / 1 (alive). Count the 8 neighbours of each cell
//   (up, down, left, right, 4 diagonals). Standard rule:
//     alive + (2 or 3 alive neighbours) -> stays alive, else dies
//     dead  + exactly 3 alive neighbours -> becomes alive
//   Rule 2 (sir's style): cell is alive next time if AT LEAST 3 neighbours are 1, otherwise dead
// PATTERN: like convolution: read OLD grid, write NEW grid (never in place!) -> collapse(2)
//          generations are SERIAL (each needs the previous); swap(old, new) after each one
// EDGES: outside the grid counts as dead (bounds check)
#include <iostream>
#include <vector>
#include <cstdlib>
#include <omp.h>
using namespace std;

int neighbours(vector<int> &g, int N, int i, int j) {
    int c = 0;
    for (int dx = -1; dx <= 1; dx++)
        for (int dy = -1; dy <= 1; dy++) {
            if (dx == 0 && dy == 0) continue;
            int r = i + dx, col = j + dy;
            if (r >= 0 && r < N && col >= 0 && col < N) c += g[r * N + col];
        }
    return c;
}

int nextState(int cell, int nb, int rule) {
    if (rule == 1) {
        if (cell == 1 && (nb == 2 || nb == 3)) return 1;
        if (cell == 0 && nb == 3) return 1;
        return 0;
    }
    return (nb >= 3) ? 1 : 0;            // rule 2: alive if at least 3 alive neighbours
}

void step(vector<int> &g, vector<int> &ng, int N, int rule, bool par) {
    #pragma omp parallel for collapse(2) if(par)
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            ng[i * N + j] = nextState(g[i * N + j], neighbours(g, N, i, j), rule);
}

int main() {
    // small check: blinker (horizontal line becomes vertical)
    int N = 5;
    vector<int> g(N * N, 0), ng(N * N);
    g[11] = g[12] = g[13] = 1;
    step(g, ng, N, 1, true);
    for (int i = 0; i < N; i++) { for (int j = 0; j < N; j++) cout << ng[i * N + j] << " "; cout << "\n"; }

    N = 1000;
    int gens = 30;
    vector<int> start(N * N);
    for (int k = 0; k < N * N; k++) start[k] = rand() % 2;

    for (int rule = 1; rule <= 2; rule++) {
        vector<int> a = start, b(N * N);
        double st = omp_get_wtime();
        for (int gen = 0; gen < gens; gen++) { step(a, b, N, rule, false); swap(a, b); }
        double ts = omp_get_wtime() - st;
        cout << "\nrule " << rule << ": N = " << N << ", generations = " << gens << ", serial time = " << ts << "\n";

        for (int t = 1; t <= 8; t = t * 2) {
            omp_set_num_threads(t);
            vector<int> p = start, q(N * N);
            st = omp_get_wtime();
            for (int gen = 0; gen < gens; gen++) { step(p, q, N, rule, true); swap(p, q); }
            double tp = omp_get_wtime() - st;
            long alive = 0;
            #pragma omp parallel for reduction(+:alive)
            for (int k = 0; k < N * N; k++) alive += p[k];
            cout << "threads = " << t << "  time = " << tp << "  speedup = " << ts / tp << "  alive = " << alive
                 << "  " << (p == a ? "MATCH" : "MISMATCH") << "\n";
        }
    }
}
