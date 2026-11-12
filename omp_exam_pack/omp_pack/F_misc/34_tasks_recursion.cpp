// PROBLEM: recursion with tasks: Fibonacci, recursive array sum, binary tree sum
// TEMPLATE: base case -> cutoff -> task 1 -> task 2 -> taskwait -> combine
//   start with: #pragma omp parallel + #pragma omp single
//   shared(x): the task stores its answer in x (needed!)
// CUTOFF: too small -> too many tiny tasks -> slower than serial
#include <iostream>
#include <vector>
#include <omp.h>
using namespace std;

long fibS(int n) { return n < 2 ? n : fibS(n - 1) + fibS(n - 2); }
long fibP(int n) {
    if (n < 25) return fibS(n);
    long x, y;
    #pragma omp task shared(x)
    x = fibP(n - 1);
    #pragma omp task shared(y)
    y = fibP(n - 2);
    #pragma omp taskwait
    return x + y;
}

long sumS(vector<long> &a, long lo, long hi) { long s = 0; for (long i = lo; i < hi; i++) s += a[i]; return s; }
long sumP(vector<long> &a, long lo, long hi) {
    if (hi - lo < 100000) return sumS(a, lo, hi);
    long mid = (lo + hi) / 2, l, r;
    #pragma omp task shared(a, l)
    l = sumP(a, lo, mid);
    #pragma omp task shared(a, r)
    r = sumP(a, mid, hi);
    #pragma omp taskwait
    return l + r;
}

// complete binary tree stored in an array: children of node i are 2i+1 and 2i+2
long treeS(vector<long> &t, long i) { if (i >= (long)t.size()) return 0; return t[i] + treeS(t, 2 * i + 1) + treeS(t, 2 * i + 2); }
long treeP(vector<long> &t, long i, int depth) {
    if (i >= (long)t.size()) return 0;
    if (depth > 10) return treeS(t, i);
    long l, r;
    #pragma omp task shared(t, l)
    l = treeP(t, 2 * i + 1, depth + 1);
    #pragma omp task shared(t, r)
    r = treeP(t, 2 * i + 2, depth + 1);
    #pragma omp taskwait
    return t[i] + l + r;
}

int main() {
    double st = omp_get_wtime();
    long f1 = fibS(40);
    double ts = omp_get_wtime() - st;
    long f2 = 0;
    st = omp_get_wtime();
    #pragma omp parallel
    {
        #pragma omp single
        f2 = fibP(40);
    }
    double tp = omp_get_wtime() - st;
    cout << "fib(40) = " << f2 << "  serial " << ts << "  parallel " << tp << "  speedup " << ts / tp
         << "  " << (f1 == f2 ? "MATCH" : "MISMATCH") << "\n";

    long n = 20000000;
    vector<long> a(n);
    for (long i = 0; i < n; i++) a[i] = i % 100;
    long s1 = sumS(a, 0, n), s2 = 0;
    #pragma omp parallel
    {
        #pragma omp single
        s2 = sumP(a, 0, n);
    }
    cout << "recursive sum = " << s2 << "  " << (s1 == s2 ? "MATCH" : "MISMATCH") << "\n";

    vector<long> tree((1 << 22) - 1);
    for (long i = 0; i < (long)tree.size(); i++) tree[i] = i % 10;
    long t1 = treeS(tree, 0), t2 = 0;
    #pragma omp parallel
    {
        #pragma omp single
        t2 = treeP(tree, 0, 0);
    }
    cout << "tree sum = " << t2 << "  " << (t1 == t2 ? "MATCH" : "MISMATCH") << "\n";
}
