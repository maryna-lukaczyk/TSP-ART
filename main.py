import argparse
import os
import random
import sys

import numpy as np
import cv2
import matplotlib.pyplot as plt
from mip import Model, BINARY, MINIMIZE, xsum, OptimizationStatus

# 1. Sampling points from portrait image
def load_and_sample_points(image_path, num_points=2000, resize_width=800, seed=42):
    random.seed(seed)
    np.random.seed(seed)

    # Validate input file before handing it to OpenCV (which silently returns
    # None on missing / unreadable files).
    if not os.path.isfile(image_path):
        raise FileNotFoundError(
            f"Input image not found: {image_path!r}\n"
            f"Place a portrait photo at this path (e.g. './example.jpg') "
            f"or pass another path as the first CLI argument:\n"
            f"    python main.py path/to/photo.jpg"
        )

    # Read and resize image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(
            f"OpenCV could not decode {image_path!r}. "
            f"Make sure it is a valid JPG/PNG/BMP file."
        )
    h, w = img.shape[:2]
    new_h = int(h * resize_width / w)
    img = cv2.resize(img, (resize_width, new_h), interpolation=cv2.INTER_LANCZOS4)

    # Convert to grayscale
    bgr = img.astype(np.float32)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

    H, W = gray.shape

    # Sobel and Laplacian edge detection
    sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    sobel_mag = np.sqrt(sobel_x**2 + sobel_y**2)

    lap = cv2.Laplacian(gray, cv2.CV_32F)
    lap_mag = np.abs(lap)

    sobel_norm = sobel_mag / (sobel_mag.max() + 1e-8)
    lap_norm = lap_mag / (lap_mag.max() + 1e-8)
    W_edges = 0.6 * sobel_norm + 0.4 * lap_norm

    # Local darkness detection
    darkness = 255 - gray
    darkness_blur = cv2.GaussianBlur(darkness, (51, 51), 0)
    local_dark = np.clip(darkness - darkness_blur, 0, None)
    local_dark_norm = local_dark / (local_dark.max() + 1e-8)

    W_local_dark = 0.30 * local_dark_norm

    # Base region boost
    region_boost = np.ones((H, W), dtype=np.float32)

    # Face region enhancement
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    faces = face_cascade.detectMultiScale(gray.astype(np.uint8), 1.1, 5)

    if len(faces) > 0:
        x, y, fw, fh = sorted(faces, key=lambda x: x[2] * x[3])[-1]

        # Face ellipse mask
        face_mask = np.zeros((H, W), float)
        center = (x + fw // 2, y + fh // 2)
        axes = (int(fw * 0.5), int(fh * 0.65))
        cv2.ellipse(face_mask, center, axes, 0, 0, 360, 1, -1)

        # Facial feature Gaussian mask
        features_mask = np.zeros((H, W), float)
        cx = x + fw // 2
        cy = y + int(fh * 0.45)
        rx = int(fw * 0.28)
        ry = int(fh * 0.22)

        Y, X = np.ogrid[:H, :W]
        dist = ((X - cx) / (rx + 1e-8))**2 + ((Y - cy) / (ry + 1e-8))**2
        features_mask = np.exp(-dist * 2.5)

        # Hair enhancement rectangle
        hair_mask = np.zeros((H, W), float)
        hx1 = max(0, x - int(fw * 0.4))
        hx2 = min(W, x + int(fw * 1.4))
        hy1 = max(0, y - int(fh * 1.0))
        hy2 = min(H, y + int(fh * 0.2))
        hair_mask[hy1:hy2, hx1:hx2] = 1.0

        # Combine boosts
        region_boost = (
            0.6
            + 0.4 * face_mask
            + 0.3 * features_mask
            + 2.5 * hair_mask
        )

    # Background suppression (blue tone)
    B = bgr[:, :, 0]
    G = bgr[:, :, 1]
    R = bgr[:, :, 2]

    background_mask = (B > 120) & (R < 130) & (G < 130)
    region_boost[background_mask] *= 0.15

    # Weight aggregation and normalization
    W_total = 1.4 * W_edges + W_local_dark
    W_total = W_total * region_boost

    assert W_total.ndim == 2, f"W_total shape error: {W_total.shape}"

    W_total += 1e-8
    W_total /= W_total.sum()

    # Weighted sampling
    flat_idx = np.random.choice(H * W, num_points, p=W_total.ravel())
    ys, xs = np.unravel_index(flat_idx, (H, W))

    xs_norm = xs / (W - 1)
    ys_norm = 1 - ys / (H - 1)

    points = np.column_stack([xs_norm, ys_norm])

    return points, gray


# 2. Degree-2 Gurobi model (multiple cycles)
def compute_distance_matrix(points):
    diff = points[:, None, :] - points[None, :, :]
    return np.sqrt((diff ** 2).sum(axis=2))


def extract_cycles(chosen_edges, n):
    neigh = [[] for _ in range(n)]
    for i, j in chosen_edges:
        neigh[i].append(j)
        neigh[j].append(i)

    visited = [False] * n
    cycles = []

    for start in range(n):
        if visited[start]:
            continue
        cycle, cur, prev = [], start, -1
        while True:
            cycle.append(cur)
            visited[cur] = True
            nxt = [x for x in neigh[cur] if x != prev]
            if not nxt:
                break
            prev, cur = cur, nxt[0]
            if cur == start:
                break
        cycles.append(cycle)
    return cycles


def solve_degree_tsp(points, time_limit=300, verbose=True, solver="auto",
                     mip_threshold=400, knn_k=25):
    """Build a degree-2 graph (collection of cycles) on the sampled points.

    Parameters
    ----------
    points : (n, 2) ndarray
    time_limit : int
        Seconds budget for the MIP backend (ignored for greedy).
    verbose : bool
    solver : {"auto", "mip", "greedy"}
        - "mip"    : Exact CBC formulation via python-mip. Becomes prohibitively
                     slow / memory-heavy for n above a few hundred.
        - "greedy" : Fast k-nearest-neighbour 2-factor heuristic. Scales to
                     thousands of points.
        - "auto"   : "mip" when n <= mip_threshold, else "greedy".
    mip_threshold : int
        Auto-selection threshold for MIP vs. greedy backend.
    knn_k : int
        Number of nearest neighbours considered in the greedy backend.

    Returns
    -------
    cycles : list of list of int
    model  : the underlying solver model (or None for the greedy backend)
    """
    n = points.shape[0]
    if solver == "auto":
        solver = "mip" if n <= mip_threshold else "greedy"

    if verbose:
        print(f"[solve_degree_tsp] n={n}, backend={solver}")

    if solver == "mip":
        return _solve_degree_tsp_mip(points, time_limit=time_limit, verbose=verbose)
    elif solver == "greedy":
        return _solve_degree_tsp_greedy(points, k=knn_k, verbose=verbose)
    else:
        raise ValueError(f"Unknown solver: {solver!r}")


def _solve_degree_tsp_mip(points, time_limit=300, verbose=True):
    """Exact degree-2 minimum-length graph via CBC (python-mip).

    Equivalent formulation to the original Gurobi model:
        min  sum_{i<j} d_ij * x_ij
        s.t. sum_j x_ij = 2   for all i
             x_ij in {0,1}
    Only practical for small n (≲ a few hundred) on the bundled CBC solver.
    """
    n = points.shape[0]
    dist = compute_distance_matrix(points)

    m = Model("deg2", sense=MINIMIZE)
    m.verbose = 1 if verbose else 0

    # Decision variables x_{ij} for i < j
    x = {}
    for i in range(n):
        for j in range(i + 1, n):
            x[i, j] = m.add_var(var_type=BINARY, obj=float(dist[i, j]))

    # Degree-2 constraints
    for i in range(n):
        m += xsum(x[min(i, j), max(i, j)] for j in range(n) if j != i) == 2

    status = m.optimize(max_seconds=time_limit)
    if verbose:
        print(f"[mip] status={status}, obj={m.objective_value}")
    if status not in (OptimizationStatus.OPTIMAL, OptimizationStatus.FEASIBLE):
        raise RuntimeError(f"MIP solver returned status {status}; no feasible solution.")

    edges = [(i, j) for (i, j), v in x.items() if v.x is not None and v.x > 0.5]
    return extract_cycles(edges, n), m


def _solve_degree_tsp_greedy(points, k=25, verbose=True):
    """Greedy minimum-length 2-factor heuristic.

    1. Build candidate edges from the k-nearest-neighbour graph.
    2. Sort by length and add greedily while respecting degree ≤ 2.
    3. Repair pass: nodes still under-degree pull in their next available
       nearest partner from the full distance matrix.
    4. Final fix-up: pair any remaining degree-1 nodes so every node ends
       at degree exactly 2 (a valid 2-factor = union of cycles).

    Scales to several thousand points within seconds.
    """
    n = points.shape[0]
    if verbose:
        print(f"[greedy] Building 2-factor for n={n} (k={k})...")

    # Pairwise distances (float32 keeps memory at n^2 * 4 bytes)
    diff = points[:, None, :] - points[None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=2)).astype(np.float32)
    np.fill_diagonal(dist, np.float32(np.inf))

    k_eff = min(k, n - 1)
    # k nearest neighbours per node (unsorted within the partition is OK)
    nn = np.argpartition(dist, k_eff, axis=1)[:, :k_eff]

    # Collect unique candidate edges (i, j) with i < j
    seen = set()
    edges = []
    for i in range(n):
        di = dist[i]
        for j in nn[i]:
            j = int(j)
            if j == i:
                continue
            a, b = (i, j) if i < j else (j, i)
            if (a, b) in seen:
                continue
            seen.add((a, b))
            edges.append((float(di[j]), a, b))
    edges.sort()

    degree = np.zeros(n, dtype=np.int8)
    chosen = set()

    # Pass 1: greedy add while respecting degree ≤ 2
    for _, i, j in edges:
        if degree[i] >= 2 or degree[j] >= 2:
            continue
        chosen.add((i, j))
        degree[i] += 1
        degree[j] += 1

    n_after_pass1 = int((degree == 2).sum())
    if verbose:
        print(f"[greedy] Pass 1: {len(chosen)} edges, {n_after_pass1}/{n} nodes saturated")

    # Pass 2: for each remaining low-degree node, scan full sorted neighbour
    # list for the closest still-available partner.
    if int(degree.min()) < 2:
        sorted_nbrs = np.argsort(dist, axis=1)
        for i in range(n):
            while degree[i] < 2:
                placed = False
                for j in sorted_nbrs[i]:
                    j = int(j)
                    if j == i or degree[j] >= 2:
                        continue
                    a, b = (i, j) if i < j else (j, i)
                    if (a, b) in chosen:
                        continue
                    chosen.add((a, b))
                    degree[i] += 1
                    degree[j] += 1
                    placed = True
                    break
                if not placed:
                    break  # no available partner left

    # Pass 3: close any residual open paths by pairing degree-1 endpoints
    deg1 = [i for i in range(n) if degree[i] == 1]
    while len(deg1) >= 2:
        i = deg1.pop()
        # nearest other degree-1 node
        best_j = None
        best_d = float("inf")
        for j in deg1:
            d = float(dist[i, j])
            if d < best_d:
                best_d = d
                best_j = j
        if best_j is None:
            break
        deg1.remove(best_j)
        a, b = (i, best_j) if i < best_j else (best_j, i)
        chosen.add((a, b))
        degree[i] += 1
        degree[best_j] += 1

    if verbose:
        print(f"[greedy] Final: {len(chosen)} edges, "
              f"degree min/max = {int(degree.min())}/{int(degree.max())}")

    cycles = extract_cycles(list(chosen), n)
    if verbose:
        print(f"[greedy] Resulting cycles: {len(cycles)}")
    return cycles, None


# 3. Merge cycles using geometric heuristic
def cycle_distance(c1, c2, points):
    """Closest pair (a in c1, b in c2) and its Euclidean distance.

    Fully vectorised: avoids the O(|c1|·|c2|) Python loop with `np.linalg.norm`
    per call that previously dominated runtime (caused merge to take hours
    on 8000-point inputs).
    """
    p1 = points[c1]                       # (|c1|, 2)
    p2 = points[c2]                       # (|c2|, 2)
    diff = p1[:, None, :] - p2[None, :, :]
    d2 = (diff * diff).sum(axis=2)        # squared distances
    flat = int(np.argmin(d2))
    i, j = divmod(flat, len(c2))
    return float(np.sqrt(d2[i, j])), (i, j)


def merge_two_cycles_geo(c1, c2, points):
    _, (i1, i2) = cycle_distance(c1, c2, points)

    def rotate(c, idx):
        return c[idx:] + c[:idx]

    c1r, c2r = rotate(c1, i1), rotate(c2, i2)

    options = [
        c1r + c2r,
        c1r + list(reversed(c2r)),
        list(reversed(c1r)) + c2r,
        list(reversed(c1r)) + list(reversed(c2r)),
    ]

    def path_len(c):
        p = points[c]
        return np.sum(np.linalg.norm(p[1:] - p[:-1], axis=1))

    return min(options, key=path_len)


def merge_all_cycles(cycles, points, verbose=True):
    """Iteratively merge all cycles into a single tour.

    Optimised vs. the original implementation by **caching pairwise
    cycle distances**. The naive version recomputed O(C²) distances after
    every merge — for 8000 points / a few hundred cycles that was hours.
    Here each cycle pair is evaluated at most twice (once initially and once
    against each new merged cycle).
    """
    # Assign stable ids to cycles so they survive the merge.
    cyc_map = {i: list(c) for i, c in enumerate(cycles)}
    active = set(cyc_map.keys())
    next_id = len(cyc_map)

    # Precompute distances between every active pair (a < b).
    dists = {}
    ids = list(active)
    for ai in range(len(ids)):
        a = ids[ai]
        for bi in range(ai + 1, len(ids)):
            b = ids[bi]
            d, pair = cycle_distance(cyc_map[a], cyc_map[b], points)
            dists[(a, b)] = (d, pair)

    while len(active) > 1:
        # Pick the globally closest pair.
        (a, b), _ = min(dists.items(), key=lambda kv: kv[1][0])

        merged = merge_two_cycles_geo(cyc_map[a], cyc_map[b], points)

        # Retire merged cycles and add the new one.
        active.discard(a)
        active.discard(b)
        del cyc_map[a]
        del cyc_map[b]
        new_id = next_id
        next_id += 1
        cyc_map[new_id] = merged

        # Drop any cached entries that referenced the consumed cycles.
        dists = {
            k: v for k, v in dists.items()
            if k[0] not in (a, b) and k[1] not in (a, b)
        }

        # Compute distances from the new cycle to all remaining active ones.
        for c in active:
            d, pair = cycle_distance(merged, cyc_map[c], points)
            key = (c, new_id) if c < new_id else (new_id, c)
            dists[key] = (d, pair)

        active.add(new_id)

        if verbose and len(active) % 10 == 0:
            print(f"Merged one cycle, remaining cycles = {len(active)}")

    if verbose:
        print(f"Merged one cycle, remaining cycles = {len(active)}")
    return cyc_map[next(iter(active))]


# 4. 2-opt improvement
def tour_length(tour, points):
    p = points[tour]
    return np.sum(np.linalg.norm(p[1:] - p[:-1], axis=1)) + np.linalg.norm(
        p[0] - p[-1]
    )


def two_opt_improve(tour, points, max_iter=100000, patience=10000):
    n = len(tour)
    tour = tour.copy()
    best_len = tour_length(tour, points)
    no_improve = 0

    for it in range(max_iter):
        i = random.randint(0, n - 3)
        j = random.randint(i + 2, n - 1)
        if i == 0 and j == n - 1:
            continue

        a, b = tour[i], tour[(i + 1) % n]
        c, d = tour[j], tour[(j + 1) % n if j + 1 < n else 0]

        pa, pb, pc, pd = points[a], points[b], points[c], points[d]

        old = np.linalg.norm(pa - pb) + np.linalg.norm(pc - pd)
        new = np.linalg.norm(pa - pc) + np.linalg.norm(pb - pd)

        if new + 1e-9 < old:
            tour[i + 1 : j + 1] = reversed(tour[i + 1 : j + 1])
            best_len -= (old - new)
            no_improve = 0
        else:
            no_improve += 1

        if no_improve > patience:
            print(f"2-opt stopped after {patience} non-improving steps")
            break

    print("2-opt improved length:", best_len)
    return tour


# 5. Drawing function
def plot_tsp(tour, points, save_path=None, lw=0.3, dpi=600):
    pts = points[tour + [tour[0]]]
    xs, ys = pts[:, 0], pts[:, 1]

    plt.figure(figsize=(7, 7))
    plt.plot(xs, ys, color="black", linewidth=lw)
    plt.axis("off")
    plt.gca().set_aspect("equal", "box")

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight", pad_inches=0)
        print("Saved:", save_path)
    else:
        plt.show()


# Main execution
def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Single-stroke TSP art from a portrait photo."
    )
    parser.add_argument(
        "image",
        nargs="?",
        default="example.jpg",
        help="Input portrait image (default: ./example.jpg).",
    )
    parser.add_argument(
        "-o", "--output", default="tsp_art_result.png",
        help="Output PNG path (default: ./tsp_art_result.png).",
    )
    parser.add_argument(
        "-n", "--num-points", type=int, default=8000,
        help="Number of sampled points (default: 8000).",
    )
    parser.add_argument(
        "--time-limit", type=int, default=900,
        help="MIP backend time budget in seconds (default: 900).",
    )
    parser.add_argument(
        "--solver", choices=["auto", "mip", "greedy"], default="auto",
        help="Solver backend for the 2-factor model (default: auto).",
    )
    parser.add_argument(
        "--no-preview", action="store_true",
        help="Skip the matplotlib sampling preview window.",
    )
    args = parser.parse_args(argv)

    points, _ = load_and_sample_points(args.image, num_points=args.num_points)
    print("Sampling completed")

    if not args.no_preview:
        img_show = cv2.cvtColor(cv2.imread(args.image), cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(7, 7))
        plt.imshow(img_show)
        plt.scatter(points[:, 0] * img_show.shape[1],
                    (1 - points[:, 1]) * img_show.shape[0],
                    s=3, c='red', alpha=0.8)
        plt.axis("off")
        plt.title("Sampling Visualization")
        plt.show()

    cycles, _ = solve_degree_tsp(points,
                                 time_limit=args.time_limit,
                                 solver=args.solver)
    print(f"Initial number of cycles = {len(cycles)}")

    big_tour = merge_all_cycles(cycles, points)
    print("Merged path length (number of nodes):", len(big_tour))

    big_tour_opt = two_opt_improve(big_tour, points,
                                   max_iter=300000,
                                   patience=15000)

    plot_tsp(big_tour_opt, points, save_path=args.output, lw=0.25, dpi=900)


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)

