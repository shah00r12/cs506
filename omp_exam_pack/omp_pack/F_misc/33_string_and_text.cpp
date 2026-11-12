// PROBLEM: text processing: count a pattern (overlapping), count vowels/digits/spaces, count words,
//          capitalise every letter into a new string
// PATTERNS: counts -> reduction(+); per-character output -> parallel for (own box)
// WORD COUNT trick: a word starts at i if text[i] is a letter AND (i == 0 or text[i-1] is a space)
//   -> each i checks only itself and its left neighbour -> independent
#include <iostream>
#include <string>
#include <cstdlib>
#include <omp.h>
using namespace std;

int main() {
    string text;
    const char *words[] = {"abba ", "open ", "mp ", "parallel ", "abbabba ", "2026 "};
    for (int r = 0; r < 2000000; r++) text += words[rand() % 6];
    long n = text.size();
    string pat = "abba";
    long m = pat.size();

    long p1 = 0, v1 = 0, w1 = 0;
    for (long i = 0; i + m <= n; i++) if (text.compare(i, m, pat) == 0) p1++;
    for (long i = 0; i < n; i++) {
        char c = text[i];
        if (c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u') v1++;
        if (c != ' ' && (i == 0 || text[i - 1] == ' ')) w1++;
    }

    long pc = 0, vow = 0, dig = 0, spaces = 0, wordsCount = 0;
    double st = omp_get_wtime();
    #pragma omp parallel for reduction(+:pc)
    for (long i = 0; i <= n - m; i++) {
        bool ok = true;
        for (long k = 0; k < m; k++) if (text[i + k] != pat[k]) { ok = false; break; }   // inner break is fine
        if (ok) pc++;
    }
    #pragma omp parallel for reduction(+:vow, dig, spaces, wordsCount)
    for (long i = 0; i < n; i++) {
        char c = text[i];
        if (c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u') vow++;
        if (c >= '0' && c <= '9') dig++;
        if (c == ' ') spaces++;
        if (c != ' ' && (i == 0 || text[i - 1] == ' ')) wordsCount++;
    }
    string upper(n, ' ');
    #pragma omp parallel for
    for (long i = 0; i < n; i++) upper[i] = (text[i] >= 'a' && text[i] <= 'z') ? text[i] - 32 : text[i];
    double tp = omp_get_wtime() - st;

    cout << "length = " << n << "  'abba' found = " << pc << "  vowels = " << vow << "  digits = " << dig
         << "  words = " << wordsCount << "  time = " << tp << "\n";
    cout << "first 20 chars uppercase: " << upper.substr(0, 20) << "\n";
    cout << ((pc == p1 && vow == v1 && wordsCount == w1) ? "MATCH" : "MISMATCH") << "\n";
}
