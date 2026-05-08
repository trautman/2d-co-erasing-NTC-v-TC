import argparse
from dataclasses import dataclass
from pathlib import Path
import math
import numpy as np
import matplotlib.pyplot as plt
import yaml

from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import time
import pickle

try:
    from scipy.optimize import minimize
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False


@dataclass
class Plan:
    points: np.ndarray
    start: np.ndarray
    end: np.ndarray
    label: str
    family: str


@dataclass
class State:
    name: str
    dirty_mask: np.ndarray
    x_h: np.ndarray
    theta_h: float
    x_r: np.ndarray
    theta_r: float
    xx: np.ndarray
    yy: np.ndarray


def softmax(logits):
    logits = np.asarray(logits, dtype=float)
    z = logits - np.max(logits)
    w = np.exp(z)
    return w / np.sum(w)


def path_length(points):
    return float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))


def make_grid(grid_n):
    xs = np.linspace(0.0, 1.0, grid_n)
    ys = np.linspace(0.0, 1.0, grid_n)
    return np.meshgrid(xs, ys)


def disk_mask(xx, yy, center, radius):
    return ((xx - center[0]) ** 2 + (yy - center[1]) ** 2) <= radius ** 2


def make_dirty_mask(name, xx, yy):
    m = np.zeros_like(xx, dtype=bool)
    if name == "central_patch":
        m |= disk_mask(xx, yy, np.array([0.50, 0.50]), 0.18)
    elif name == "two_unequal_patches":
        m |= disk_mask(xx, yy, np.array([0.37, 0.58]), 0.145)
        m |= disk_mask(xx, yy, np.array([0.64, 0.47]), 0.095)
    elif name == "three_patch_competition":
        m |= disk_mask(xx, yy, np.array([0.32, 0.52]), 0.090)
        m |= disk_mask(xx, yy, np.array([0.50, 0.50]), 0.145)
        m |= disk_mask(xx, yy, np.array([0.68, 0.47]), 0.095)
    elif name == "bottleneck":
        m |= disk_mask(xx, yy, np.array([0.38, 0.50]), 0.135)
        m |= disk_mask(xx, yy, np.array([0.62, 0.50]), 0.135)
        m |= ((xx > 0.38) & (xx < 0.62) & (yy > 0.455) & (yy < 0.545))
    elif name == "sparse_cleanup":
        for c, r in [
            ([0.31, 0.31], 0.055),
            ([0.42, 0.74], 0.045),
            ([0.53, 0.43], 0.065),
            ([0.66, 0.66], 0.050),
            ([0.72, 0.28], 0.050),
        ]:
            m |= disk_mask(xx, yy, np.array(c), r)
    elif name == "asymmetric_reachability":
        m |= disk_mask(xx, yy, np.array([0.36, 0.58]), 0.100)
        m |= disk_mask(xx, yy, np.array([0.52, 0.50]), 0.145)
        m |= disk_mask(xx, yy, np.array([0.71, 0.40]), 0.105)
    else:
        raise ValueError(f"Unknown board mask: {name}")
    return m


def default_location_for_mask(name):
    if name == "central_patch":
        return np.array([0.34, 0.50]), 0.0, np.array([0.66, 0.50]), math.pi
    if name == "two_unequal_patches":
        return np.array([0.31, 0.58]), -0.20, np.array([0.75, 0.42]), math.pi + 0.20
    if name == "three_patch_competition":
        return np.array([0.26, 0.52]), -0.05, np.array([0.74, 0.46]), math.pi + 0.05
    if name == "bottleneck":
        return np.array([0.27, 0.50]), 0.0, np.array([0.73, 0.50]), math.pi
    if name == "sparse_cleanup":
        return np.array([0.28, 0.67]), -0.60, np.array([0.77, 0.35]), math.pi + 0.40
    if name == "asymmetric_reachability":
        return np.array([0.30, 0.58]), -0.18, np.array([0.75, 0.38]), math.pi + 0.18
    raise ValueError(name)


def make_states(cfg):
    grid_n = int(cfg["board"]["grid_n"])
    xx, yy = make_grid(grid_n)
    states = []
    for name in cfg["states"]["board_masks"]:
        dirty = make_dirty_mask(name, xx, yy)
        x_h, th_h, x_r, th_r = default_location_for_mask(name)
        states.append(State(name=name, dirty_mask=dirty, x_h=x_h, theta_h=th_h, x_r=x_r, theta_r=th_r, xx=xx, yy=yy))
    return states


def bezier_points(start, control, end, n_points):
    t = np.linspace(0.0, 1.0, n_points)
    pts = ((1 - t) ** 2)[:, None] * start[None, :] + (2 * (1 - t) * t)[:, None] * control[None, :] + (t ** 2)[:, None] * end[None, :]
    pts[:, 0] = np.clip(pts[:, 0], 0.0, 1.0)
    pts[:, 1] = np.clip(pts[:, 1], 0.0, 1.0)
    return pts


def line_points(start, end, n_points):
    t = np.linspace(0.0, 1.0, n_points)
    pts = start[None, :] + t[:, None] * (end - start)[None, :]
    pts[:, 0] = np.clip(pts[:, 0], 0.0, 1.0)
    pts[:, 1] = np.clip(pts[:, 1], 0.0, 1.0)
    return pts


def connected_components_centers(mask, xx, yy):
    try:
        from scipy import ndimage
        labeled, n = ndimage.label(mask)
        centers = []
        sizes = []
        for k in range(1, n + 1):
            comp = labeled == k
            if np.sum(comp) == 0:
                continue
            centers.append(np.array([float(np.mean(xx[comp])), float(np.mean(yy[comp]))]))
            sizes.append(int(np.sum(comp)))
        order = np.argsort(sizes)[::-1]
        return [centers[i] for i in order]
    except Exception:
        ys, xs = np.where(mask)
        if len(xs) == 0:
            return [np.array([0.5, 0.5])]
        return [np.array([float(np.mean(xx[mask])), float(np.mean(yy[mask]))])]



def rbf_kernel_1d(t, lengthscale, sigma):
    t = np.asarray(t, dtype=float)
    dt = t[:, None] - t[None, :]
    return (float(sigma) ** 2) * np.exp(-0.5 * (dt / float(lengthscale)) ** 2)


def clamp_vector_norm(vectors, max_norm):
    norms = np.linalg.norm(vectors, axis=-1)
    scale = np.ones_like(norms)
    mask = norms > max_norm
    scale[mask] = float(max_norm) / np.maximum(norms[mask], 1e-12)
    return vectors * scale[..., None]


def sample_simulator_omni_di_plans(start, theta, prefix, cfg):
    """
    Normalized-coordinate port of the simulator robot marginal proposal.

    Simulator source model:
      TaskWeightedMarginal(
          proposal=OmniDoubleIntegrator(...),
          terms=[CleanDirtTerm(...)]
      )

    This function implements the proposal piece: smooth GP acceleration samples,
    integrated through a 2D omni-directional double integrator with speed and
    acceleration clamps. The utility is computed later by single_plan_scores(),
    matching CleanDirtTerm: disk-footprint dirt cleaned, with invalid off-board
    trajectories rejected.
    """
    mcfg = cfg["marginals"]["simulator_task_weighted_omni_di"]
    K = int(cfg["trajectory"]["n_samples_per_agent"])
    N = int(cfg["trajectory"]["n_points"])
    horizon = float(mcfg["horizon"])
    vmax = float(mcfg["vmax"])
    amax = float(mcfg["amax"])
    accel_lengthscale = float(mcfg["accel_lengthscale"])
    accel_sigma = float(mcfg["accel_sigma"])
    accel_nugget = float(mcfg["accel_nugget"])
    initial_speed = float(mcfg["initial_speed"])

    # Different but deterministic seeds for H/R so repeated runs are stable.
    base_seed = int(mcfg.get("random_seed", 0))
    seed_offset = 101 if prefix.upper().startswith("H") else 202
    rng = np.random.default_rng(base_seed + seed_offset)

    ts = np.linspace(0.0, horizon, N)
    dt = ts[1] - ts[0] if N > 1 else horizon
    Kmat = rbf_kernel_1d(ts, accel_lengthscale, accel_sigma)
    Kmat = Kmat + np.eye(N) * accel_nugget
    L = np.linalg.cholesky(Kmat)

    Zx = rng.standard_normal((N, K))
    Zy = rng.standard_normal((N, K))
    AX = L @ Zx
    AY = L @ Zy

    x = np.full(K, float(start[0]))
    y = np.full(K, float(start[1]))
    vx = np.full(K, initial_speed * math.cos(theta))
    vy = np.full(K, initial_speed * math.sin(theta))

    xs = np.empty((K, N), dtype=float)
    ys = np.empty((K, N), dtype=float)
    xs[:, 0] = x
    ys[:, 0] = y

    for k in range(N - 1):
        acc = np.stack([AX[k], AY[k]], axis=1)
        acc = clamp_vector_norm(acc, amax)
        vx = vx + acc[:, 0] * dt
        vy = vy + acc[:, 1] * dt
        vel = clamp_vector_norm(np.stack([vx, vy], axis=1), vmax)
        vx = vel[:, 0]
        vy = vel[:, 1]
        x = x + vx * dt
        y = y + vy * dt
        xs[:, k + 1] = x
        ys[:, k + 1] = y

    plans = []
    for i in range(K):
        pts = np.stack([xs[i], ys[i]], axis=1)
        plans.append(Plan(points=pts, start=start.copy(), end=pts[-1].copy(), label=f"{prefix}{i}", family="simulator_omni_di"))
    return plans

def generate_plans(start, theta, state, prefix, cfg):
    if cfg.get("marginals", {}).get("generator", "") == "simulator_task_weighted_omni_di":
        return sample_simulator_omni_di_plans(start, theta, prefix, cfg)

    tcfg = cfg["trajectory"]
    n_points = int(tcfg["n_points"])
    length = float(tcfg["length"])
    families = set(tcfg["proposal_families"])
    plans = []

    def add(points, family):
        plans.append(Plan(points=points, start=start.copy(), end=points[-1].copy(), label=f"{prefix}{len(plans)}", family=family))

    if "straight" in families:
        n = max(5, int(tcfg["n_samples_per_agent"]) // 3)
        for a in np.linspace(-float(tcfg["straight_angle_width"]), float(tcfg["straight_angle_width"]), n):
            direction = np.array([math.cos(theta + a), math.sin(theta + a)])
            add(line_points(start, start + length * direction, n_points), "straight")

    if "one_bend" in families:
        base_dir = np.array([math.cos(theta), math.sin(theta)])
        perp = np.array([-base_dir[1], base_dir[0]])
        end = start + length * base_dir
        for off in tcfg["one_bend_lateral_offsets"]:
            control = start + 0.50 * length * base_dir + float(off) * perp
            add(bezier_points(start, control, end, n_points), "one_bend")

    if "patch_seek" in families:
        centers = connected_components_centers(state.dirty_mask, state.xx, state.yy)
        base_dir = np.array([math.cos(theta), math.sin(theta)])
        perp = np.array([-base_dir[1], base_dir[0]])
        for center in centers[:4]:
            for off in tcfg["patch_seek_lateral_offsets"]:
                target = center + float(off) * perp
                target = np.clip(target, 0.0, 1.0)
                control = start + 0.35 * (target - start)
                add(bezier_points(start, control, target, n_points), "patch_seek")

    # Keep most task-relevant and diverse enough by trimming after scoring later; return all here.
    return plans


def points_to_mask(points, grid_n, radius_px):
    mask = np.zeros((grid_n, grid_n), dtype=bool)
    rr = int(radius_px)

    for p in points:
        # Skip off-board points. The full trajectory will still be marked invalid later.
        if p[0] < 0.0 or p[0] > 1.0 or p[1] < 0.0 or p[1] > 1.0:
            continue

        ix = int(round(p[0] * (grid_n - 1)))
        iy = int(round(p[1] * (grid_n - 1)))

        x0 = max(0, ix - rr)
        x1 = min(grid_n, ix + rr + 1)
        y0 = max(0, iy - rr)
        y1 = min(grid_n, iy + rr + 1)

        if x1 <= x0 or y1 <= y0:
            continue

        ys, xs = np.ogrid[y0:y1, x0:x1]
        mask[y0:y1, x0:x1] |= (xs - ix) ** 2 + (ys - iy) ** 2 <= rr ** 2

    return mask


def compute_masks(plans, grid_n, radius_px):
    return [points_to_mask(p.points, grid_n, radius_px) for p in plans]


def single_plan_scores(plans, masks, dirty_mask, cfg):
    """
    Marginal utility.

    For generator=simulator_task_weighted_omni_di, this intentionally mirrors the
    simulator's TaskWeightedMarginal + CleanDirtTerm: utility is disk-footprint
    dirt cleaned, and invalid off-board trajectories receive -inf utility.
    """
    dirty_area = max(1.0, float(np.sum(dirty_mask)))
    generator = cfg.get("marginals", {}).get("generator", "")
    scores = []
    for plan, mask in zip(plans, masks):
        valid = bool(np.all(plan.points[:, 0] >= 0.0) and np.all(plan.points[:, 0] <= 1.0)
                     and np.all(plan.points[:, 1] >= 0.0) and np.all(plan.points[:, 1] <= 1.0))
        if generator == "simulator_task_weighted_omni_di":
            if not valid:
                scores.append(-1.0e9)
            else:
                # CleanDirtTerm-style score: amount of dirt under the trajectory footprint.
                scores.append(float(np.sum(mask & dirty_mask)) / dirty_area)
        else:
            mcfg = cfg["marginals"]
            board_erased = float(np.sum(mask & dirty_mask)) / dirty_area
            clean_swept = float(np.sum(mask & (~dirty_mask))) / dirty_area
            L = path_length(plan.points)
            U = (
                float(mcfg["board_erased_weight"]) * board_erased
                - float(mcfg["path_length_weight"]) * L
                - float(mcfg.get("wiping_clean_board_weight", 0.0)) * clean_swept
            )
            scores.append(U if valid else -1.0e9)
    return np.array(scores)


def trim_to_top(plans, masks, scores, n_keep):
    if len(plans) <= n_keep:
        return plans, masks, scores
    order = np.argsort(scores)[::-1][:n_keep]
    return [plans[i] for i in order], [masks[i] for i in order], scores[order]


def marginal_from_scores(scores, beta):
    return softmax(float(beta) * scores)


def pairwise_matrices(H, R, H_masks, R_masks, dirty_mask, conflict_distance=0.075):
    n_h, n_r = len(H), len(R)
    dirty_area = max(1.0, float(np.sum(dirty_mask)))
    out = {k: np.zeros((n_h, n_r)) for k in ["m_board_erased", "m_redundant_work", "m_conflict", "m_path_length", "m_wiping_clean_board"]}

    for i, h in enumerate(H):
        for j, r in enumerate(R):
            h_mask, r_mask = H_masks[i], R_masks[j]
            h_dirty = h_mask & dirty_mask
            r_dirty = r_mask & dirty_mask
            union_swept = h_mask | r_mask
            union_dirty = h_dirty | r_dirty
            redundant_dirty = h_dirty & r_dirty

            out["m_board_erased"][i, j] = float(np.sum(union_dirty)) / dirty_area
            out["m_redundant_work"][i, j] = float(np.sum(redundant_dirty)) / dirty_area
            out["m_path_length"][i, j] = path_length(h.points) + path_length(r.points)

            # Plain-language meaning: effort spent wiping board that was already clean / not dirty at state s.
            out["m_wiping_clean_board"][i, j] = float(np.sum(union_swept & (~dirty_mask))) / dirty_area

            d = np.linalg.norm(h.points[:, None, :] - r.points[None, :, :], axis=2)
            dmin = float(np.min(d))
            out["m_conflict"][i, j] = math.exp(-((dmin / conflict_distance) ** 2))
    return out


def make_costs(metrics):
    return {
        "c_nominal": metrics["m_conflict"],
        "c_combined": (
            -metrics["m_board_erased"]
            + 1.25 * metrics["m_redundant_work"]
            + 1.00 * metrics["m_conflict"]
            + 0.20 * metrics["m_path_length"]
            + 0.25 * metrics["m_wiping_clean_board"]
        ),
    }


def gamma_independent(p_h, p_r):
    return np.outer(p_h, p_r)


def gamma_response_sample(p_h, p_r, C, temperature):
    h_star = int(np.argmax(p_h))
    q = p_r * np.exp(-C[h_star, :] / temperature)
    q = q / np.sum(q)
    gamma = np.zeros_like(C)
    gamma[h_star, :] = q
    return gamma


def gamma_response_marginal(p_h, p_r, C, temperature):
    expected_cost_r = p_h @ C
    q = p_r * np.exp(-expected_cost_r / temperature)
    q = q / np.sum(q)
    return np.outer(p_h, q)


def gamma_joint_kl(p_h, p_r, C, temperature):
    g = np.outer(p_h, p_r) * np.exp(-C / temperature)
    return g / np.sum(g)


def gamma_marginal_kl_sinkhorn(
    p_h,
    p_r,
    C,
    lambda_h,
    lambda_r,
    epsilon=0.015,
    max_iter=500,
    tol=1e-8,
):
    eps = 1e-12

    p_h = np.clip(p_h, eps, None)
    p_r = np.clip(p_r, eps, None)

    p_h = p_h / np.sum(p_h)
    p_r = p_r / np.sum(p_r)

    C_shifted = C - np.min(C)

    K = np.outer(p_h, p_r) * np.exp(-C_shifted / epsilon)
    K = np.clip(K, eps, None)

    u = np.ones_like(p_h)
    v = np.ones_like(p_r)

    tau_h = lambda_h / (lambda_h + epsilon)
    tau_r = lambda_r / (lambda_r + epsilon)

    for _ in range(max_iter):
        u_old = u.copy()
        v_old = v.copy()

        Kv = K @ v
        u = (p_h / np.clip(Kv, eps, None)) ** tau_h

        KTu = K.T @ u
        v = (p_r / np.clip(KTu, eps, None)) ** tau_r

        du = np.max(np.abs(u - u_old))
        dv = np.max(np.abs(v - v_old))

        if max(du, dv) < tol:
            break

    gamma = u[:, None] * K * v[None, :]
    gamma = np.clip(gamma, 0.0, None)

    gamma = gamma / np.sum(gamma)

    return gamma


def gamma_marginal_kl_slsqp(p_h, p_r, C, lambda_h, lambda_r):
    if not SCIPY_AVAILABLE:
        return gamma_joint_kl(p_h, p_r, C, temperature=max(lambda_h, lambda_r))
    n_h, n_r = C.shape
    n = n_h * n_r
    eps = 1e-12
    x0 = np.outer(p_h, p_r).reshape(n)

    def unpack(x):
        return x.reshape((n_h, n_r))

    def obj(x):
        g = unpack(x)
        gh = np.sum(g, axis=1)
        gr = np.sum(g, axis=0)
        gh_safe = np.clip(gh, eps, None)
        gr_safe = np.clip(gr, eps, None)
        ph_safe = np.clip(p_h, eps, None)
        pr_safe = np.clip(p_r, eps, None)
        return float(np.sum(g * C)
                     + lambda_h * np.sum(gh_safe * (np.log(gh_safe) - np.log(ph_safe)))
                     + lambda_r * np.sum(gr_safe * (np.log(gr_safe) - np.log(pr_safe))))

    def grad(x):
        g = unpack(x)
        gh = np.sum(g, axis=1)
        gr = np.sum(g, axis=0)
        gh_safe = np.clip(gh, eps, None)
        gr_safe = np.clip(gr, eps, None)
        ph_safe = np.clip(p_h, eps, None)
        pr_safe = np.clip(p_r, eps, None)
        G = C + lambda_h * (np.log(gh_safe / ph_safe) + 1.0)[:, None] + lambda_r * (np.log(gr_safe / pr_safe) + 1.0)[None, :]
        return G.reshape(n)

    cons = [{"type": "eq", "fun": lambda x: np.sum(x) - 1.0, "jac": lambda x: np.ones_like(x)}]
    res = minimize(obj, x0, jac=grad, method="SLSQP", bounds=[(0.0, 1.0)] * n, constraints=cons,
                   options={"maxiter": 250, "ftol": 1e-9, "disp": False})
    if not res.success:
        print(f"WARNING: marginal-KL solve failed ({res.message}); falling back to joint_kl")
        return gamma_joint_kl(p_h, p_r, C, temperature=max(lambda_h, lambda_r))
    g = unpack(res.x)
    g = np.clip(g, 0.0, None)
    return g / np.sum(g)



def gamma_marginal_kl(
    p_h,
    p_r,
    C,
    lambda_h,
    lambda_r,
    solver="slsqp",
    epsilon=0.015,
    max_iter=500,
    tol=1e-8,
):
    if solver == "sinkhorn_unbalanced":
        return gamma_marginal_kl_sinkhorn(
            p_h=p_h,
            p_r=p_r,
            C=C,
            lambda_h=lambda_h,
            lambda_r=lambda_r,
            epsilon=epsilon,
            max_iter=max_iter,
            tol=tol,
        )

    if solver == "slsqp":
        return gamma_marginal_kl_slsqp(
            p_h=p_h,
            p_r=p_r,
            C=C,
            lambda_h=lambda_h,
            lambda_r=lambda_r,
        )

    raise ValueError(f"Unknown marginal KL solver: {solver}")





def expected_metric(gamma, M):
    return float(np.sum(gamma * M))


def benefit_value(metric_name, baseline, ntc):
    higher = {"m_board_erased"}
    if metric_name in higher:
        return ntc - baseline
    return baseline - ntc


# def model_dict(p_h, p_r, C, cfg):
#     ocfg = cfg["ot"]
#     models = {
#         "ind": gamma_independent(p_h, p_r),
#         "resp_sample": gamma_response_sample(p_h, p_r, C, float(ocfg["response_temperature"])),
#         "resp_marg": gamma_response_marginal(p_h, p_r, C, float(ocfg["response_temperature"])),
#         "joint_kl": gamma_joint_kl(p_h, p_r, C, float(ocfg["joint_kl_temperature"])),
#     }
#     # models["ntc_marginal_kl"] = gamma_marginal_kl(
#     #     p_h, p_r, C, float(ocfg["marginal_kl_lambda_h"]), float(ocfg["marginal_kl_lambda_r"])
#     # )
#     solver_name = ocfg.get("marginal_kl_solver", "slsqp")
#     if bool(cfg.get("progress", {}).get("print_solver", True)):
#         print(f"  using marginal_kl_solver={solver_name}", flush=True)

#     models["ntc_marginal_kl"] = gamma_marginal_kl(
#         p_h=p_h,
#         p_r=p_r,
#         C=C,
#         lambda_h=float(ocfg["marginal_kl_lambda_h"]),
#         lambda_r=float(ocfg["marginal_kl_lambda_r"]),
#         solver=solver_name,
#         epsilon=float(ocfg.get("marginal_kl_entropy_epsilon", 0.015)),
#         max_iter=int(ocfg.get("sinkhorn_max_iter", 500)),
#         tol=float(ocfg.get("sinkhorn_tol", 1e-8)),
#     )
#     return models
def model_dict(p_h, p_r, C, cfg):
    ocfg = cfg["ot"]
    models_to_compute = cfg["models"].get(
        "models_to_compute",
        ["ind", "resp_sample", "resp_marg", "joint_kl", "ntc_marginal_kl"],
    )

    models = {}

    if "ind" in models_to_compute:
        models["ind"] = gamma_independent(p_h, p_r)

    if "resp_sample" in models_to_compute:
        models["resp_sample"] = gamma_response_sample(
            p_h,
            p_r,
            C,
            float(ocfg["response_temperature"]),
        )

    if "resp_marg" in models_to_compute:
        models["resp_marg"] = gamma_response_marginal(
            p_h,
            p_r,
            C,
            float(ocfg["response_temperature"]),
        )

    if "joint_kl" in models_to_compute:
        models["joint_kl"] = gamma_joint_kl(
            p_h,
            p_r,
            C,
            float(ocfg["joint_kl_temperature"]),
        )

    if "ntc_marginal_kl" in models_to_compute:
        solver_name = ocfg.get("marginal_kl_solver", "slsqp")
        print(f"  using marginal_kl_solver={solver_name}", flush=True)

        models["ntc_marginal_kl"] = gamma_marginal_kl(
            p_h=p_h,
            p_r=p_r,
            C=C,
            lambda_h=float(ocfg["marginal_kl_lambda_h"]),
            lambda_r=float(ocfg["marginal_kl_lambda_r"]),
            solver=solver_name,
            epsilon=float(ocfg.get("marginal_kl_entropy_epsilon", 0.015)),
            max_iter=int(ocfg.get("sinkhorn_max_iter", 500)),
            tol=float(ocfg.get("sinkhorn_tol", 1e-8)),
        )

    return models


def top_indices_1d(p, k):
    return list(np.argsort(p)[::-1][:k])


def top_pairs(gamma, k):
    flat = np.argsort(gamma.reshape(-1))[::-1][:k]
    pairs = []
    for idx in flat:
        ij = np.unravel_index(idx, gamma.shape)
        pairs.append((int(ij[0]), int(ij[1]), float(gamma[ij])))
    return pairs


def setup_axis(ax, state, title):
    ax.imshow(state.dirty_mask, extent=(0, 1, 0, 1), origin="lower", alpha=0.35)
    ax.scatter([state.x_h[0]], [state.x_h[1]], s=55, marker="o")
    ax.scatter([state.x_r[0]], [state.x_r[1]], s=55, marker="s")
    ax.arrow(state.x_h[0], state.x_h[1], 0.045 * math.cos(state.theta_h), 0.045 * math.sin(state.theta_h),
             width=0.003, head_width=0.018, color="black")
    ax.arrow(state.x_r[0], state.x_r[1], 0.045 * math.cos(state.theta_r), 0.045 * math.sin(state.theta_r),
             width=0.003, head_width=0.018, color="black")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=9)


def plot_plan(ax, plan, lw=1.7, alpha=0.9):
    ax.plot(plan.points[:, 0], plan.points[:, 1], linewidth=lw, alpha=alpha)
    ax.scatter([plan.points[-1, 0]], [plan.points[-1, 1]], s=12, alpha=alpha)


def plot_metric_key(out_dir):
    text = [
        ("m_board_erased", "Fraction of initially dirty board erased by the two-agent team. Higher is better."),
        ("m_redundant_work", "Fraction of initially dirty board swept by both agents. This is redundant work on dirt. Lower is better."),
        ("m_conflict", "How close the two end effectors get, using a soft proximity penalty. Lower is better."),
        ("m_path_length", "Total distance traveled by both end effectors. Lower is better."),
        ("m_wiping_clean_board", "Fraction of sweeping spent over board that was already clean/not dirty at state s. Lower is better."),
        ("marginal generator", "simulator_task_weighted_omni_di: smooth double-integrator trajectory samples scored by CleanDirtTerm-style dirt cleaned."),
        ("proposal family", "simulator_omni_di only in this run; no hand-coded straight/patch-seek families."),
        ("c_nominal", "Simple proximity-only cost: c_nominal = m_conflict."),
        ("c_combined", "Task cost: -board_erased + redundant_work + conflict + path_length + wiping_clean_board terms."),
    ]
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.axis("off")
    y = 0.95
    ax.text(0.02, y, "Metric and cost key", fontsize=18, weight="bold", va="top")
    y -= 0.08
    for name, desc in text:
        ax.text(0.04, y, name, fontsize=13, weight="bold", va="top")
        ax.text(0.27, y, desc, fontsize=12, va="top", wrap=True)
        y -= 0.105
    fig.tight_layout()
    fig.savefig(out_dir / "metric_key_page.png", dpi=180)
    plt.close(fig)


def plot_configuration_page(records, cfg, out_dir):
    k = int(cfg["plotting"]["top_k_marginal_samples"])
    fig, axes = plt.subplots(len(records), 2, figsize=(9, 4.1 * len(records)))
    if len(records) == 1:
        axes = np.array([axes])
    for row, rec in enumerate(records):
        state = rec["state"]
        setup_axis(axes[row, 0], state, f"{state.name}: state")
        ax = axes[row, 1]
        setup_axis(ax, state, f"{state.name}: top marginals")
        for idx in top_indices_1d(rec["p_h"], k):
            plot_plan(ax, rec["H"][idx], lw=1.7)
        for idx in top_indices_1d(rec["p_r"], k):
            plot_plan(ax, rec["R"][idx], lw=1.7)
    fig.tight_layout()
    fig.savefig(out_dir / "configuration_page.png", dpi=180)
    plt.close(fig)


def plot_expected_metric_page(results, cost_name, cfg, out_dir):
    metrics = cfg["metrics_to_run"]
    models = cfg["models"]["baselines"] + [cfg["models"]["ntc_model"]]
    states = [r["state"].name for r in results]
    fig, axes = plt.subplots(len(metrics), 1, figsize=(13, 3.2 * len(metrics)))
    if len(metrics) == 1:
        axes = [axes]
    x = np.arange(len(states))
    width = 0.14
    for ax, m in zip(axes, metrics):
        for k, model in enumerate(models):
            vals = [r["by_cost"][cost_name]["summary"][model][m] for r in results]
            ax.bar(x + (k - (len(models)-1)/2) * width, vals, width, label=model)
        ax.set_title(f"Expected metric: {m} | cost: {cost_name}")
        ax.set_xticks(x); ax.set_xticklabels(states, rotation=25, ha="right")
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / f"expected_metric_page_{cost_name}.png", dpi=180)
    plt.close(fig)


def plot_metric_page(results, cost_name, cfg, out_dir):
    metrics = cfg["metrics_to_run"]
    baselines = cfg["models"]["baselines"]
    ntc = cfg["models"]["ntc_model"]
    states = [r["state"].name for r in results]
    fig, axes = plt.subplots(len(metrics), 1, figsize=(13, 3.2 * len(metrics)))
    if len(metrics) == 1:
        axes = [axes]
    x = np.arange(len(states))
    width = 0.17
    for ax, m in zip(axes, metrics):
        for k, base in enumerate(baselines):
            vals = []
            for r in results:
                s = r["by_cost"][cost_name]["summary"]
                vals.append(benefit_value(m, s[base][m], s[ntc][m]))
            ax.bar(x + (k - (len(baselines)-1)/2) * width, vals, width, label=f"{base} → ntc")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(f"collaboration benefit: {m} | cost: {cost_name}")
        ax.set_xticks(x); ax.set_xticklabels(states, rotation=25, ha="right")
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / f"metric_page_{cost_name}.png", dpi=180)
    plt.close(fig)


def plot_top_k_pair_page(results, cost_name, cfg, out_dir):
    models = cfg["models"]["baselines"] + [cfg["models"]["ntc_model"]]
    top_k = int(cfg["plotting"]["top_k_pairs"])
    nrows = len(results)
    ncols = len(models)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 3.4 * nrows))
    if nrows == 1:
        axes = np.array([axes])
    for row, rec in enumerate(results):
        state = rec["state"]
        for col, model in enumerate(models):
            ax = axes[row, col]
            setup_axis(ax, state, f"{state.name}\n{model}")
            gamma = rec["by_cost"][cost_name]["models"][model]
            for rank, (i, j, prob) in enumerate(top_pairs(gamma, top_k)):
                alpha = max(0.25, 0.95 - 0.18 * rank)
                lw = max(0.8, 2.2 - 0.25 * rank)
                plot_plan(ax, rec["H"][i], lw=lw, alpha=alpha)
                plot_plan(ax, rec["R"][j], lw=lw, alpha=alpha)
            i, j, p = top_pairs(gamma, 1)[0]
            ax.text(0.02, 0.02, f"top H{i}, R{j}\np={p:.3f}", transform=ax.transAxes,
                    fontsize=7, bbox={"facecolor": "white", "alpha": 0.65, "edgecolor": "none"})
    fig.tight_layout()
    fig.savefig(out_dir / f"top_k_pair_page_{cost_name}.png", dpi=180)
    plt.close(fig)









def save_results(results, cfg, out_dir):
    out_path = out_dir / "results.pkl"

    payload = {
        "results": results,
        "config": cfg,
    }

    with open(out_path, "wb") as f:
        pickle.dump(payload, f)

    print(f"\nSaved results to: {out_path}")


def load_results(out_dir):
    in_path = out_dir / "results.pkl"

    with open(in_path, "rb") as f:
        payload = pickle.load(f)

    print(f"\nLoaded results from: {in_path}")

    return payload["results"], payload["config"]

def analyze_state(state, cfg):
    t0 = time.perf_counter()

    worker_logs = bool(cfg.get("progress", {}).get("worker_logs", False))

    def log_step(message):
        if worker_logs:
            elapsed = time.perf_counter() - t0
            print(f"[{state.name}] {message} | elapsed={elapsed:.1f}s", flush=True)
    grid_n = int(cfg["board"]["grid_n"])
    radius = int(cfg["trajectory"]["footprint_radius_px"])
    n_keep = int(cfg["trajectory"]["n_samples_per_agent"])

    H_all = generate_plans(state.x_h, state.theta_h, state, "H", cfg)
    R_all = generate_plans(state.x_r, state.theta_r, state, "R", cfg)
    H_masks_all = compute_masks(H_all, grid_n, radius)
    R_masks_all = compute_masks(R_all, grid_n, radius)

    h_scores_all = single_plan_scores(H_all, H_masks_all, state.dirty_mask, cfg)
    r_scores_all = single_plan_scores(R_all, R_masks_all, state.dirty_mask, cfg)

    H, H_masks, h_scores = trim_to_top(H_all, H_masks_all, h_scores_all, n_keep)
    R, R_masks, r_scores = trim_to_top(R_all, R_masks_all, r_scores_all, n_keep)

    p_h = marginal_from_scores(h_scores, float(cfg["marginals"]["beta_h"]))
    p_r = marginal_from_scores(r_scores, float(cfg["marginals"]["beta_r"]))

    metrics = pairwise_matrices(H, R, H_masks, R_masks, state.dirty_mask)
    costs = make_costs(metrics)

    by_cost = {}
    for cost_name in cfg["costs_to_run"]:
        C = costs[cost_name]
        models = model_dict(p_h, p_r, C, cfg)
        summary = {}
        for model_name, gamma in models.items():
            summary[model_name] = {}
            for m in cfg["metrics_to_run"]:
                summary[model_name][m] = expected_metric(gamma, metrics[m])
        by_cost[cost_name] = {"models": models, "summary": summary, "C": C}

    return {
        "state": state, "H": H, "R": R, "H_masks": H_masks, "R_masks": R_masks,
        "p_h": p_h, "p_r": p_r, "metrics": metrics, "costs": costs, "by_cost": by_cost
    }


def print_summary(results, cfg):
    for cost_name in cfg["costs_to_run"]:
        print(f"\nCost: {cost_name}")
        print("=" * (6 + len(cost_name)))
        for rec in results:
            print(f"\nState: {rec['state'].name}")
            summary = rec["by_cost"][cost_name]["summary"]
            for model_name, vals in summary.items():
                compact = ", ".join([f"{m}={vals[m]:.4f}" for m in cfg["metrics_to_run"]])
                print(f"  {model_name}: {compact}")


def summarize_collaboration_benefit(results, cfg, out_dir):
    rows = []

    ntc = cfg["models"]["ntc_model"]
    baselines = cfg["models"]["baselines"]

    for cost_name in cfg["costs_to_run"]:
        for metric_name in cfg["metrics_to_run"]:
            for baseline in baselines:
                values = []

                for rec in results:
                    summary = rec["by_cost"][cost_name]["summary"]
                    baseline_value = summary[baseline][metric_name]
                    ntc_value = summary[ntc][metric_name]

                    values.append(
                        benefit_value(
                            metric_name,
                            baseline_value,
                            ntc_value,
                        )
                    )

                rows.append(
                    {
                        "cost": cost_name,
                        "metric": metric_name,
                        "comparison": f"{baseline}_to_{ntc}",
                        "mean_cb": float(np.mean(values)),
                        "std_cb": float(np.std(values)),
                        "min_cb": float(np.min(values)),
                        "max_cb": float(np.max(values)),
                    }
                )

    out_path = out_dir / "collaboration_benefit_summary.csv"

    with open(out_path, "w") as f:
        f.write("cost,metric,comparison,mean_cb,std_cb,min_cb,max_cb\n")
        for row in rows:
            f.write(
                f"{row['cost']},"
                f"{row['metric']},"
                f"{row['comparison']},"
                f"{row['mean_cb']:.8f},"
                f"{row['std_cb']:.8f},"
                f"{row['min_cb']:.8f},"
                f"{row['max_cb']:.8f}\n"
            )

    print(f"Wrote summary: {out_path}")





def plot_summarized_cb_page(results, cost_name, cfg, out_dir):
    metrics = cfg["metrics_to_run"]
    baselines = cfg["models"]["baselines"]
    ntc = cfg["models"]["ntc_model"]

    fig, axes = plt.subplots(
        len(metrics),
        1,
        figsize=(10, 3.0 * len(metrics)),
        sharex=False,
    )

    if len(metrics) == 1:
        axes = [axes]

    x = np.arange(len(baselines))

    for ax, metric_name in zip(axes, metrics):
        means = []
        stds = []

        for baseline in baselines:
            vals = []

            for rec in results:
                summary = rec["by_cost"][cost_name]["summary"]

                if baseline not in summary:
                    continue

                if ntc not in summary:
                    continue

                vals.append(
                    benefit_value(
                        metric_name,
                        summary[baseline][metric_name],
                        summary[ntc][metric_name],
                    )
                )

            means.append(float(np.mean(vals)))
            stds.append(float(np.std(vals)))

        ax.axhline(0.0, linewidth=1.0)
        ax.bar(x, means, yerr=stds, capsize=4)

        ax.set_title(
            f"scenario-averaged collaboration benefit: {metric_name} | cost: {cost_name}"
        )
        ax.set_xticks(x)
        ax.set_xticklabels([f"{b} → {ntc}" for b in baselines], rotation=20, ha="right")
        ax.set_ylabel("mean CB")
        ax.grid(True, axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_dir / f"summarized_cb_page_{cost_name}.png", dpi=180)
    plt.close(fig)








def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    out_dir = Path(cfg.get("output_dir", "outputs"))
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Config:")
    print(f"  states: {cfg['states']['board_masks']}")
    print(f"  costs_to_run: {cfg['costs_to_run']}")
    print(f"  metrics_to_run: {cfg['metrics_to_run']}")
    print(f"  n_samples_per_agent: {cfg['trajectory']['n_samples_per_agent']}")
    print(f"  marginal_generator: {cfg.get('marginals', {}).get('generator', 'hand_built_proposals')}")
    if 'proposal_families' in cfg.get('trajectory', {}):
        print(f"  proposal_families: {cfg['trajectory']['proposal_families']}")
    print(f"  scipy_available: {SCIPY_AVAILABLE}")
    print(f"  marginal_kl_solver: {cfg.get('ot', {}).get('marginal_kl_solver', 'slsqp')}")

    states = make_states(cfg)
 
    max_workers = int(cfg.get("parallel", {}).get("max_workers", 1))

    if max_workers > 1:
        print(f"Parallel state analysis with max_workers={max_workers}")

        results_by_name = {}

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_state_name = {
                executor.submit(analyze_state, state, cfg): state.name
                for state in states
            }

            with tqdm(
                total=len(future_to_state_name),
                desc="Analyzing states",
                unit="state",
                dynamic_ncols=True,
            ) as pbar:
                for future in as_completed(future_to_state_name):
                    state_name = future_to_state_name[future]
                    results_by_name[state_name] = future.result()
                    pbar.update(1)
                    pbar.set_postfix_str(f"finished={state_name}")

        results = [results_by_name[state.name] for state in states]

    else:
        results = []

        with tqdm(
            total=len(states),
            desc="Analyzing states",
            unit="state",
            dynamic_ncols=True,
        ) as pbar:
            for state in states:
                results.append(analyze_state(state, cfg))
                pbar.update(1)
                pbar.set_postfix_str(f"finished={state.name}")

    save_results(results, cfg, out_dir)

    summarize_collaboration_benefit(results, cfg, out_dir)

    if cfg["plotting"].get("make_metric_key_page", True):
        plot_metric_key(out_dir)
    if cfg["plotting"].get("make_configuration_page", True):
        plot_configuration_page(results, cfg, out_dir)

    for cost_name in cfg["costs_to_run"]:
        if cfg["plotting"].get("make_expected_metric_pages", True):
            plot_expected_metric_page(results, cost_name, cfg, out_dir)
        if cfg["plotting"].get("make_metric_pages", True):
            plot_metric_page(results, cost_name, cfg, out_dir)
        if cfg["plotting"].get("make_top_k_pair_pages", True):
            plot_top_k_pair_page(results, cost_name, cfg, out_dir)
        if cfg["plotting"].get("make_summarized_cb_pages", True):
            plot_summarized_cb_page(results, cost_name, cfg, out_dir)

    print_summary(results, cfg)

    print("\nWrote:")
    print(f"  {out_dir / 'metric_key_page.png'}")
    print(f"  {out_dir / 'configuration_page.png'}")
    for cost_name in cfg["costs_to_run"]:
        print(f"  {out_dir / ('expected_metric_page_' + cost_name + '.png')}")
        print(f"  {out_dir / ('metric_page_' + cost_name + '.png')}")
        print(f"  {out_dir / ('top_k_pair_page_' + cost_name + '.png')}")


if __name__ == "__main__":
    main()
