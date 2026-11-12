// PROBLEM: k-means clustering of 2D points into K groups
// each iteration (iterations are SERIAL):
//   1. ASSIGN: each point -> nearest centre. Each point writes its own label[i] -> parallel for
//              "did any label change?" -> reduction(||:changed)
//   2. UPDATE: new centre = average of its points. Sums per cluster = a few shared boxes
//              -> own local sums per thread + critical merge (histogram pattern)
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <omp.h>
using namespace std;

int kmeans(int n, int K, vector<double> &px, vector<double> &py, vector<double> &cx, vector<double> &cy,
           vector<int> &label, bool par) {
    for (int c = 0; c < K; c++) { cx[c] = px[c]; cy[c] = py[c]; }      // first K points as start
    label.assign(n, -1);
    int it = 0;
    bool changed = true;
    while (changed && it < 100) {
        changed = false;
        #pragma omp parallel for reduction(||:changed) if(par)
        for (int i = 0; i < n; i++) {
            int best = 0;
            double bd = 1e300;
            for (int c = 0; c < K; c++) {
                double d = (px[i] - cx[c]) * (px[i] - cx[c]) + (py[i] - cy[c]) * (py[i] - cy[c]);
                if (d < bd) { bd = d; best = c; }
            }
            if (label[i] != best) { label[i] = best; changed = true; }
        }

        vector<double> sx(K, 0), sy(K, 0);
        vector<long> cnt(K, 0);
        #pragma omp parallel if(par)
        {
            vector<double> lx(K, 0), ly(K, 0);
            vector<long> lc(K, 0);
            #pragma omp for
            for (int i = 0; i < n; i++) { lx[label[i]] += px[i]; ly[label[i]] += py[i]; lc[label[i]]++; }
            #pragma omp critical
            {
                for (int c = 0; c < K; c++) { sx[c] += lx[c]; sy[c] += ly[c]; cnt[c] += lc[c]; }
            }
        }
        for (int c = 0; c < K; c++)
            if (cnt[c] > 0) { cx[c] = sx[c] / cnt[c]; cy[c] = sy[c] / cnt[c]; }
        it++;
    }
    return it;
}

int main() {
    int n = 2000000, K = 4;
    vector<double> px(n), py(n);
    double gx[4] = {0, 10, 0, 10}, gy[4] = {0, 0, 10, 10};
    for (int i = 0; i < n; i++) {                       // 4 blobs around (0,0) (10,0) (0,10) (10,10)
        int g = i % 4;
        px[i] = gx[g] + (rand() % 1000) / 500.0 - 1;
        py[i] = gy[g] + (rand() % 1000) / 500.0 - 1;
    }
    vector<double> cx1(K), cy1(K), cx2(K), cy2(K);
    vector<int> l1, l2;
    double st = omp_get_wtime();
    int it1 = kmeans(n, K, px, py, cx1, cy1, l1, false);
    double ts = omp_get_wtime() - st;
    cout << "serial: " << it1 << " iterations, time = " << ts << "\n";
    for (int c = 0; c < K; c++) cout << "  centre " << c << " = (" << cx1[c] << ", " << cy1[c] << ")\n";

    for (int t = 1; t <= 8; t = t * 2) {
        omp_set_num_threads(t);
        st = omp_get_wtime();
        int it2 = kmeans(n, K, px, py, cx2, cy2, l2, true);
        double tp = omp_get_wtime() - st;
        bool ok = (l1 == l2) && (it1 == it2);
        for (int c = 0; c < K; c++) if (fabs(cx1[c] - cx2[c]) > 1e-6 || fabs(cy1[c] - cy2[c]) > 1e-6) ok = false;
        cout << "threads = " << t << "  iterations = " << it2 << "  time = " << tp << "  speedup = " << ts / tp
             << "  " << (ok ? "MATCH" : "MISMATCH") << "\n";
    }
}
