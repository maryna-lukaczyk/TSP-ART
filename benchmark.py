"""Quick benchmark of the optimised pipeline at several sizes."""
import time
import sys

import main as m


def bench(n, image="example.jpg"):
    t0 = time.perf_counter()
    pts, _ = m.load_and_sample_points(image, num_points=n)
    t_sample = time.perf_counter() - t0

    t0 = time.perf_counter()
    cycles, _ = m.solve_degree_tsp(pts, solver="greedy", verbose=False)
    t_solve = time.perf_counter() - t0
    n_cycles = len(cycles)

    t0 = time.perf_counter()
    tour = m.merge_all_cycles(cycles, pts, verbose=False)
    t_merge = time.perf_counter() - t0

    t0 = time.perf_counter()
    tour = m.two_opt_improve(tour, pts, max_iter=300_000, patience=15_000)
    t_2opt = time.perf_counter() - t0

    t0 = time.perf_counter()
    m.plot_tsp(tour, pts, save_path=f"bench_n{n}.png", lw=0.25, dpi=600)
    t_plot = time.perf_counter() - t0

    total = t_sample + t_solve + t_merge + t_2opt + t_plot
    print(
        f"n={n:>5}  cycles={n_cycles:>4}  "
        f"sample={t_sample:5.2f}s  solve={t_solve:5.2f}s  "
        f"merge={t_merge:6.2f}s  2opt={t_2opt:5.2f}s  "
        f"plot={t_plot:5.2f}s  TOTAL={total:6.2f}s"
    )
    return total


if __name__ == "__main__":
    sizes = [int(x) for x in sys.argv[1:]] if len(sys.argv) > 1 else [1500, 3000]
    for n in sizes:
        bench(n)

