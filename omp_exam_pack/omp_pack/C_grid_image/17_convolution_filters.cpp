// PROBLEM: 2D convolution with a 3x3 filter: blur (avg / 9), edge (divide 1), sharpen (divide 1)
// out(i, j) = sum of img(i+x, j+y) * K[x][y] for x, y = 0..2   -> output (R-2) x (C-2)
// PATTERN: each output pixel = own box -> collapse(2), local sum, SEPARATE output array
#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>
using namespace std;

void conv(vector<double> &img, vector<double> &out, int R, int C, double K[3][3], double div, bool par) {
    int oR = R - 2, oC = C - 2;
    #pragma omp parallel for collapse(2) if(par)
    for (int i = 0; i < oR; i++)
        for (int j = 0; j < oC; j++) {
            double s = 0;
            for (int x = 0; x < 3; x++)
                for (int y = 0; y < 3; y++) s += img[(long)(i + x) * C + (j + y)] * K[x][y];
            out[(long)i * oC + j] = s / div;
        }
}

int main() {
    double avg[3][3] = {{1, 1, 1}, {1, 1, 1}, {1, 1, 1}};
    double edge[3][3] = {{0, 1, 0}, {1, -4, 1}, {0, 1, 0}};
    vector<double> img = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150,
                          160, 170, 180, 190, 200, 210, 220, 230, 240, 250}, out(9);
    conv(img, out, 5, 5, avg, 9, true);
    cout << "blur 5x5 example: ";
    for (int k = 0; k < 9; k++) cout << out[k] << " ";
    conv(img, out, 5, 5, edge, 1, true);
    cout << "\nedge 5x5 example: ";
    for (int k = 0; k < 9; k++) cout << out[k] << " ";
    cout << "\n";

    int n = 3000;
    vector<double> big((long)n * n), o1((long)(n - 2) * (n - 2)), o2((long)(n - 2) * (n - 2));
    for (long k = 0; k < (long)n * n; k++) big[k] = k % 256;
    double st = omp_get_wtime();
    conv(big, o1, n, n, avg, 9, false);
    double ts = omp_get_wtime() - st;
    for (int t = 1; t <= 8; t = t * 2) {
        omp_set_num_threads(t);
        st = omp_get_wtime();
        conv(big, o2, n, n, avg, 9, true);
        double tp = omp_get_wtime() - st;
        bool ok = true;
        for (long k = 0; k < (long)o1.size(); k++) if (fabs(o1[k] - o2[k]) > 1e-9) ok = false;
        cout << "threads = " << t << "  time = " << tp << "  speedup = " << ts / tp << "  " << (ok ? "MATCH" : "MISMATCH") << "\n";
    }
}
