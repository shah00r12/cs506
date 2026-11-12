# OpenMP Exam Pack: 38 Solved Problems + How to Handle ANY Problem

Every `.cpp` here is complete and tested: data → serial → parallel (1, 2, 4, 8 threads) → check (MATCH) → time, speedup, efficiency.
The comment at the top of each file explains the idea, the pattern and the traps.

```
g++ -fopenmp -Wall file.cpp -o file        then   .\file  (Windows)   ./file  (Linux)
```

---

## PART 1: How to solve a problem you have NEVER seen (5 steps)

**Step 1. Write the normal (serial) version first.** Do a tiny example by hand → write it in English → turn it into code.
A correct serial version already earns marks, and you need it anyway to check the parallel one.

**Step 2. Find the loops. For each loop ask: "Does one turn need the result of an earlier turn?"**
- **No** → this loop can be split among threads.
- **Yes** → this loop stays **serial**. Look one level deeper: the work *inside* one step is usually parallel.
  (Game of Life: generations serial, cells parallel. Dijkstra/Prim: steps serial, finding the min + updating parallel.
  Gaussian elimination: k serial, rows parallel. Floyd–Warshall: k serial, i/j parallel. BFS: levels serial, nodes parallel.)

**Step 3. Look at the LEFT side of `=` inside the parallel loop:**

| Each turn writes… | Use |
|---|---|
| its own box `out[i]` | `#pragma omp parallel for` |
| its own box in a 2D grid `out[i*C+j]` | `#pragma omp parallel for collapse(2)` |
| one number (sum, count, max, min, true/false) | `reduction(+ / max / min / && / \|\| : var)` |
| several independent numbers | `reduction(+:a, b) reduction(max:m)` |
| value + its position (they depend on each other) | own copy + `omp for` + `critical` merge once |
| a few shared boxes (histogram, cluster sums) | own local array + `omp for` + `critical` merge once |
| it reads neighbours that are also being updated | **two arrays**: read OLD, write NEW, then `swap` (Game of Life, Jacobi, BFS, PageRank, labels) |
| recursion (no loop) | `task` + `taskwait`, started with `parallel` + `single` |

**Step 4. Extras.**
- Work per turn is uneven (primes, graphs, triangle loops `j < i`, growing work) → `schedule(dynamic, 64)`
- A different job for each thread → `#pragma omp parallel sections`
- Random numbers → give each thread its own seed (file 08); `rand()` is not thread-safe
- `bool` flags: use `vector<char>`/`vector<int>`, never `vector<bool>`

**Step 5. Wrap it in the skeleton** (file 38): serial + time → loop t = 1, 2, 4, 8 → parallel + time → check → print speedup and efficiency.

**If you are stuck:** write the serial version, then add `#pragma omp parallel for` to the safest independent loop and add a comment
explaining why it is safe (each turn writes its own box / the outer loop must stay serial because …). Partial, explained work still scores.

---

## PART 2: Five ways to deal with a dependency (memorise these)

| Technique | When | Example files |
|---|---|---|
| Outer loop serial, inner loop parallel | step k needs step k−1 | 14 Gauss, 15 Floyd, 29 Prim, 30 Dijkstra, 09 sieve |
| Old/new arrays + swap | new value needs neighbours' values | 16 Life, 18 Jacobi, 26 BFS, 27 PageRank, 28 components |
| Phases (even/odd rounds), barrier between | neighbouring pairs | 20 odd-even sort |
| Block method (local pass, combine, fix-up pass) | running totals | 07 prefix sum |
| Tasks | divide and conquer recursion | 21 merge, 22 quick, 34 fib/sum/tree |

---

## PART 3: File index

### A. Basics
| # | Problem | Key idea |
|---|---|---|
| 01 | vector ops: add, saxpy, reverse | parallel for, separate output |
| 02 | sum: the race + 4 fixes | wrong / reduction / critical / atomic / own copy |
| 03 | statistics: sum, min, max, mean, stddev, count above mean | several reductions on one line |
| 04 | max and min WITH position | own copy + critical, tie-break on index |
| 05 | search: first/last/count, 2D matrix search, many binary searches | reduction(min/max/+); flatten (i,j) to i*C+j |
| 06 | histogram, grade groups, letter count | local array + critical; atomic; reduction(+:h[:10]) |
| 07 | prefix sum | block method |
| 08 | Monte Carlo π (random) + π by integration | per-thread random seed, reduction |
| 09 | primes count, sieve of Eratosthenes, twin primes | schedule(dynamic); sieve: p serial, crossing out parallel |

### B. Matrices
| # | Problem | Key idea |
|---|---|---|
| 10 | add + transpose (rectangular) | collapse(2), T[j*R+i] = A[i*C+j] |
| 11 | matrix multiply (ijk + ikj) | collapse(2) on i,j, local sum, never split k |
| 12 | **blocked (tiled) matrix multiply** | parallel over C blocks (ii, jj); kk, i, k, j inside; cache reuse |
| 13 | matrix × vector, row/column sums, trace, total | rows only (no collapse); trace = reduction |
| 14 | **Gaussian elimination** (solve Ax = b) | k serial, rows below k parallel; back-substitution sum = reduction |
| 15 | **Floyd–Warshall** all-pairs shortest paths | k serial, i/j collapse(2) in place (row/column k don't change) |

### C. Grids and images
| # | Problem | Key idea |
|---|---|---|
| 16 | **Game of Life** (standard rule + "≥3 neighbours alive" rule) | 8-neighbour count, old/new grids, generations serial |
| 17 | convolution: blur, edge | (R−2)×(C−2) output, collapse(2) |
| 18 | **heat diffusion / Jacobi stencil** until convergence | old/new, collapse(2), reduction(max:diff) |
| 19 | image: brighten+cap, invert, threshold, 256-bin histogram, average | collapse(2) + reductions + local histogram |

### D. Sorting
| # | Problem | Key idea |
|---|---|---|
| 20 | bubble sort → odd-even sort | rounds, barrier, no nowait |
| 21 | merge sort | tasks, cutoff, taskwait before merge |
| 22 | quick sort | partition first, then 2 tasks |
| 23 | selection sort; insertion sort | selection: min+index each round; insertion: chunks in parallel + merge |
| 24 | **rank (enumeration) sort**; **counting sort** | rank: out[rank[i]] = a[i] (own box); counting: histogram → prefix → fill per value |

### E. Graphs (CSR and adjacency matrix)
| # | Problem | Key idea |
|---|---|---|
| 25 | **CSR**: build, degrees, max degree, SpMV (weighted) | parallel over nodes, schedule(dynamic), never collapse |
| 26 | **BFS** on CSR | levels serial; unvisited node checks neighbours at level L; old/new |
| 27 | **PageRank** on CSR | pr_new[v] from neighbours' pr/degree; reduction(+:diff) to converge |
| 28 | **connected components** | label propagation: min label of neighbours; old/new; reduction(\|\|) |
| 29 | **Prim's MST** | n steps serial; pick min key (own copy + critical); update keys (parallel for) |
| 30 | **Dijkstra** | same shape as Prim: pick min dist, relax in parallel |
| 31 | **triangle counting** | sorted CSR, merge-intersection, reduction(+), dynamic |

**CSR in one picture** (graph 0-1, 0-2, 1-2, 2-3):
```
rowPtr = 0 2 4 7 8          neighbours of u = colIdx[rowPtr[u]] .. colIdx[rowPtr[u+1]-1]
colIdx = 1 2 0 2 0 1 3 2    node 2 -> colIdx[4..6] = 0 1 3

for (int u = 0; u < n; u++)                              // PARALLEL over nodes
    for (int k = rowPtr[u]; k < rowPtr[u + 1]; k++) {    // neighbours (serial inside)
        int v = colIdx[k];
    }
```
Build: count degrees → prefix sum → rowPtr → place each edge (file 25).

### F. Other
| # | Problem | Key idea |
|---|---|---|
| 32 | **k-means clustering** | assign: own box + reduction(\|\|:changed); update: local sums + critical |
| 33 | text: pattern count, vowels/digits, word count, uppercase | reductions; word starts if text[i] != ' ' and text[i−1] == ' ' |
| 34 | tasks: Fibonacci, recursive array sum, binary tree sum | task + shared(x) + taskwait + cutoff |
| 35 | scheduling: static vs dynamic(100) vs dynamic(1) vs guided | repeating vs growing work |
| 36 | sections + single, master, barrier, firstprivate, lastprivate | different jobs at once |
| 37 | N-body forces; pairs with a target sum; closest pair | own box; triangle loop + dynamic + reduction(min) |
| 38 | **EMPTY TEMPLATE** | copy it for any new problem |

---

## PART 4: Problem → pattern quick lookup

| If the question says… | Do this |
|---|---|
| "for each element compute …" | parallel for |
| "for each cell / pixel …" | parallel for collapse(2) |
| "total / count / how many / average" | reduction(+) |
| "largest / smallest" | reduction(max/min) |
| "largest and WHERE" / "top 2" | own copy + critical |
| "frequency / histogram / per category" | local array + critical |
| "neighbours" (grid, cells, pixels) | old/new arrays, collapse(2) |
| "repeat until it stops changing / converges" | while loop serial; inside parallel; reduction(\|\| or max) for the stop test |
| "graph / nodes / edges / CSR" | parallel over nodes, schedule(dynamic) |
| "shortest path / spanning tree" | steps serial; min-selection + update parallel |
| "sort" | odd-even (bubble), tasks (merge/quick), rank sort, counting sort |
| "recursive / divide and conquer / tree" | tasks |
| "uneven work / primes / triangular loop" | schedule(dynamic) |
| "random" | per-thread seed |
| "three different computations on the same data" | sections |

---

## PART 5: Checks and common errors

**Correctness check:** whole numbers `==`; decimals `fabs(a − b) < 1e-6`; vectors `v1 == v2`; plain arrays: loop;
sorted: is sorted AND == serial; random results: close to the expected value.

| Error | Fix |
|---|---|
| undefined reference to `omp_…` | add `-fopenmp` |
| `ignoring #pragma omp …` | spelling (critical, reduction, collapse) |
| `for statement expected` | no `{` between `parallel for` and `for` |
| `Assertion '__n < this->size()' failed` | index past the end: `<` vs `<=`, sizes, i/j swapped |
| different answer every run | race: missing reduction / critical / `{ }`, or updating in place instead of old/new |
| parallel slower than serial | too little work per step (Prim/Dijkstra on small n), atomic/critical in the hot loop, tiny n |

**Before compiling:** `{ }` on multi-line `if`/`critical` · `long` for big sums · `(double)` for averages ·
start values (sum 0, product 1, max INT_MIN, min INT_MAX, first position n) · `#include <climits>`, `<cmath>`, `<algorithm>` ·
separate output array.

---

## PART 6: Viva one-liners
- **Race condition:** two threads update the same memory without protection → lost updates.
- **Reduction:** private copy per thread, combined once at the end.
- **critical vs atomic:** any block, one thread at a time / one simple update, faster.
- **collapse(2):** merge two perfectly nested loops into one iteration space.
- **static / dynamic / guided:** fixed split / chunks on request / shrinking chunks.
- **Tasks vs parallel for:** recursion or irregular work vs loops with a known count.
- **taskwait:** wait for child tasks before combining. **Cutoff:** avoid too many tiny tasks.
- **single / master:** one thread (others wait) / thread 0 (no wait). **sections:** different jobs in parallel.
- **private / firstprivate / lastprivate / shared:** own copy (not set) / own copy (starts with the outside value) / value from the last iteration kept / one copy for all.
- **Why old/new arrays:** every cell must use the same generation's values.
- **Why blocked matrix multiply is faster:** small tiles stay in the cache and are reused.
- **Why Prim/Dijkstra scale poorly:** n serial steps, each with little work, so the overhead dominates.
- **Speedup = Ts/Tp, efficiency = speedup/threads. Amdahl:** serial fraction f limits speedup to 1/f.
