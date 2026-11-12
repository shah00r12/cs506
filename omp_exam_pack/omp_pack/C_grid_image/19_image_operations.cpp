// PROBLEM: image operations on an N x N grayscale image (0..255)
//   brighten (+50, cap 255, count capped) | invert (255 - p) | threshold (p > 128 ? 255 : 0)
//   image histogram (256 bins) | average brightness | count pixels above 200
// PATTERNS: per-pixel output -> collapse(2); counters/sums -> reduction; 256 bins -> local array + critical
#include <iostream>
#include <vector>
#include <cstdlib>
#include <omp.h>
using namespace std;

int main() {
    int N = 3000;
    long sz = (long)N * N;
    vector<int> img(sz), br(sz), inv(sz), th(sz);
    for (long k = 0; k < sz; k++) img[k] = rand() % 256;

    long capped = 0, bright = 0, total = 0;
    #pragma omp parallel for collapse(2) reduction(+:capped, bright, total)
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            long p = (long)i * N + j;
            int v = img[p] + 50;
            if (v > 255) { v = 255; capped++; }
            br[p] = v;
            inv[p] = 255 - img[p];
            th[p] = (img[p] > 128) ? 255 : 0;
            if (img[p] > 200) bright++;
            total += img[p];
        }

    long hist[256] = {0};
    #pragma omp parallel
    {
        long local[256] = {0};
        #pragma omp for
        for (long k = 0; k < sz; k++) local[img[k]]++;
        #pragma omp critical
        {
            for (int b = 0; b < 256; b++) hist[b] += local[b];
        }
    }

    // serial check
    long c2 = 0, b2 = 0, t2 = 0, h2[256] = {0};
    bool ok = true;
    for (long k = 0; k < sz; k++) {
        int v = img[k] + 50;
        if (v > 255) { v = 255; c2++; }
        if (br[k] != v || inv[k] != 255 - img[k] || th[k] != ((img[k] > 128) ? 255 : 0)) ok = false;
        if (img[k] > 200) b2++;
        t2 += img[k];
        h2[img[k]]++;
    }
    for (int b = 0; b < 256; b++) if (hist[b] != h2[b]) ok = false;
    ok = ok && capped == c2 && bright == b2 && total == t2;
    cout << "capped = " << capped << "  above 200 = " << bright << "  average = " << (double)total / sz
         << "  pixels with value 0 = " << hist[0] << "  " << (ok ? "MATCH" : "MISMATCH") << "\n";
}
