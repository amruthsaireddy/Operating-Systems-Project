"""
Adaptive Z-Domain Feedback Scheduler (AZFS)
============================================

Reference simulation implementation for the AZFS project.

Implements:
    - Queue plant model
    - AZFS PI controller
    - Adaptive service-rate estimation using EWMA
    - Adaptive integral gain
    - FCFS baseline
    - Round Robin baseline
    - Tustin-discretized PID baseline
    - Step workload disturbance
    - Performance metrics
    - CSV result export
    - Summary report
    - Visualization
    - Optional multi-seed experiments

IMPORTANT IMPLEMENTATION NOTE
-----------------------------
The theoretical paper proposes the pure integral controller:

        C(z) = KI * z / (z - 1)

The queue plant already contains an integrating pole at z = 1.
Therefore, the pure-I controller creates a double-integrator loop.

The implementation uses the following PI refinement:

        C(z) = KI * z / (z - 1) + KP

with the incremental realization:

        u[k] = u[k-1]
               + KI[k] * e[k]
               + KP * (e[k] - e[k-1])

The integral component preserves the intended zero-bias tracking
behavior, while the proportional component provides damping.

The simulation also includes actuator saturation:

        0 <= u[k] <= U_MAX

Therefore, the simulated system is nonlinear whenever the actuator
saturates.

Run:
    python azfs_sim.py

Optional:
    python azfs_sim.py --multi-seed

Produces:
    azfs_results.csv
    azfs_summary.txt
    azfs_comparison.png
    azfs_zoomed_comparison.png
    azfs_adaptation.png

With --multi-seed:
    azfs_multiseed_results.csv
    azfs_multiseed_summary.txt
"""

import argparse
import csv
from dataclasses import dataclass

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ======================================================================
# Configuration
# ======================================================================

@dataclass
class Config:
    # Simulation
    N_EPOCHS: int = 2000
    Ts: float = 1.0

    # Workload
    LAMBDA_NOM: float = 12.0
    STEP_EPOCH: int = 50
    STEP_SIZE: float = 8.0

    # Queue reference
    Q_REF: float = 20.0

    # Service rate
    MU_NOM: float = 10.0
    SERVICE_RATE_DRIFT: bool = True
    SERVICE_RATE_STD: float = 0.6

    # AZFS
    KI_NOM: float = -0.1
    KP_DAMPING: float = -0.1
    ALPHA: float = 0.9

    # Actuator
    U_MIN: float = 0.0
    U_MAX: float = 3.0

    # Metrics
    SS_WINDOW: int = 100
    VARIANCE_WINDOW: int = 200
    SETTLING_WINDOW: int = 30
    SETTLING_FRAC: float = 0.05
    SETTLING_FLOOR: float = 1.0

    # Reproducibility
    SEED: int = 42


# ======================================================================
# Queue Plant
# ======================================================================

class QueuePlant:
    """
    Discrete-time queue model:

        q[k+1] = q[k] + lambda*Ts - mu*u[k]*Ts

    where:

        q      = queue depth
        lambda = arrival rate
        mu     = service rate
        u      = dispatch/service control fraction
        Ts     = scheduling interval

    The actuator is constrained to a non-negative dispatch value.
    """

    def __init__(self, q0, Ts, u_min=0.0, u_max=3.0):
        self.q = float(q0)
        self.Ts = float(Ts)
        self.u_min = float(u_min)
        self.u_max = float(u_max)

    def step(self, lam, mu, u):
        u = np.clip(u, self.u_min, self.u_max)

        self.q = max(
            0.0,
            self.q
            + lam * self.Ts
            - mu * u * self.Ts
        )

        return self.q


# ======================================================================
# AZFS Controller
# ======================================================================

class AZFSController:
    """
    Adaptive Z-domain PI controller.

    The theoretical controller is:

        C(z) = KI * z / (z - 1)

    The implemented refinement is:

        C(z) = KI * z / (z - 1) + KP

    Incremental realization:

        u[k] = u[k-1]
               + KI[k] * e[k]
               + KP * (e[k] - e[k-1])

    where KI[k] is adapted using:

        KI[k] = KI_nom * mu_nom / mu_hat[k]

    and:

        mu_hat[k] =
            alpha * mu_hat[k-1]
            + (1-alpha) * mu_obs[k]

    The error convention used here is:

        e[k] = q_ref - q[k]

    Therefore, because KI and KP are negative:

        q < q_ref  -> e > 0 -> control decreases
        q > q_ref  -> e < 0 -> control increases

    This matches the plant physics:
    increasing u drains the queue.
    """

    def __init__(
        self,
        KI_nom,
        mu_nom,
        Ts,
        alpha=0.9,
        KP_DAMPING=-0.1
    ):
        self.KI_nom = KI_nom
        self.mu_nom = mu_nom
        self.Ts = Ts
        self.alpha = alpha
        self.KP_DAMPING = KP_DAMPING

        self.mu_hat = mu_nom

        self.u_prev = 0.0
        self.e_prev = 0.0

    @staticmethod
    def jury_stable_range(mu, Ts):
        """
        Nominal gain interval given in the paper:

            -4/(mu*Ts) < KI < 0

        Note:
        This interval comes from the paper's stated Jury analysis.
        For the pure-I + integrating-plant configuration, the constant
        term is exactly 1, so strict Schur stability requires additional
        care. The implemented PI refinement addresses this issue.
        """

        lower = -4.0 / (mu * Ts)
        upper = 0.0

        return lower, upper

    def update_mu_estimate(self, mu_obs):
        """
        EWMA service-rate estimator.
        """

        self.mu_hat = (
            self.alpha * self.mu_hat
            + (1.0 - self.alpha) * mu_obs
        )

    def adaptive_gain(self):
        """
        Adaptive gain:

            KI[k] = KI_nom * mu_nom / mu_hat[k]
        """

        return (
            self.KI_nom
            * self.mu_nom
            / max(self.mu_hat, 1e-6)
        )

    def control(self, error):
        """
        Calculate the next control command.

        Incremental PI realization:

            u[k] = u[k-1]
                   + KI[k] * e[k]
                   + KP * (e[k] - e[k-1])
        """

        KI = self.adaptive_gain()

        u = (
            self.u_prev
            + KI * error
            + self.KP_DAMPING * (error - self.e_prev)
        )

        self.u_prev = u
        self.e_prev = error

        return u, KI


# ======================================================================
# Workload / Service Profiles
# ======================================================================

def generate_profiles(cfg, seed=None):
    """
    Generate workload and service-rate sequences.

    The same generated sequences are passed to all schedulers so that
    comparisons are performed under identical disturbances.

    Workload:

        epochs < STEP_EPOCH:
            lambda = LAMBDA_NOM

        epochs >= STEP_EPOCH:
            lambda = LAMBDA_NOM + STEP_SIZE

    Service rate:

        MU_NOM + Gaussian drift

    when SERVICE_RATE_DRIFT is enabled.
    """

    if seed is None:
        seed = cfg.SEED

    rng = np.random.default_rng(seed)

    # ------------------------------
    # Workload profile
    # ------------------------------

    lam_series = np.full(
        cfg.N_EPOCHS,
        cfg.LAMBDA_NOM,
        dtype=float
    )

    lam_series[cfg.STEP_EPOCH:] += cfg.STEP_SIZE

    # ------------------------------
    # Service profile
    # ------------------------------

    if cfg.SERVICE_RATE_DRIFT:
        mu_series = (
            cfg.MU_NOM
            + rng.normal(
                0.0,
                cfg.SERVICE_RATE_STD,
                size=cfg.N_EPOCHS
            )
        )

        mu_series = np.clip(
            mu_series,
            cfg.MU_NOM * 0.5,
            cfg.MU_NOM * 1.5
        )

    else:
        mu_series = np.full(
            cfg.N_EPOCHS,
            cfg.MU_NOM,
            dtype=float
        )

    return lam_series, mu_series


# ======================================================================
# Scheduler Simulations
# ======================================================================

def run_fcfs(cfg, lam_series, mu_series):
    """
    FCFS baseline.

    The scheduler always dispatches at full nominal control:

        u = 1

    No feedback or adaptation is applied.
    """

    plant = QueuePlant(
        cfg.Q_REF,
        cfg.Ts,
        cfg.U_MIN,
        cfg.U_MAX
    )

    q_hist = []

    for k in range(cfg.N_EPOCHS):

        q = plant.step(
            lam_series[k],
            mu_series[k],
            u=1.0
        )

        q_hist.append(q)

    return np.asarray(q_hist)


def run_round_robin(cfg, lam_series, mu_series):
    """
    Round Robin baseline.

    A fixed dispatch fraction is used to represent a static
    time-sharing policy.

    No feedback or adaptation is applied.
    """

    plant = QueuePlant(
        cfg.Q_REF,
        cfg.Ts,
        cfg.U_MIN,
        cfg.U_MAX
    )

    q_hist = []

    quantum_fraction = 0.9

    for k in range(cfg.N_EPOCHS):

        q = plant.step(
            lam_series[k],
            mu_series[k],
            u=quantum_fraction
        )

        q_hist.append(q)

    return np.asarray(q_hist)


def run_pid_tustin(
    cfg,
    lam_series,
    mu_series,
    Kp=0.08,
    Ki=0.03,
    Kd=0.01
):
    """
    Tustin-discretized PID baseline.

    Error convention:

        e[k] = q[k] - q_ref

    Positive error means queue is above target, so positive control
    increases dispatch and drains the queue.

    The control signal is saturated to the configured actuator limits.
    """

    plant = QueuePlant(
        cfg.Q_REF,
        cfg.Ts,
        cfg.U_MIN,
        cfg.U_MAX
    )

    q_hist = []

    e_prev = 0.0
    e_prev2 = 0.0
    u_prev = 0.0

    Ts = cfg.Ts

    # Tustin-discretized PID coefficients
    a0 = (
        Kp
        + Ki * Ts / 2.0
        + 2.0 * Kd / Ts
    )

    a1 = (
        -Kp
        + Ki * Ts / 2.0
        - 4.0 * Kd / Ts
    )

    a2 = 2.0 * Kd / Ts

    for k in range(cfg.N_EPOCHS):

        e_k = plant.q - cfg.Q_REF

        u_k = (
            u_prev
            + a0 * e_k
            + a1 * e_prev
            + a2 * e_prev2
        )

        u_k = np.clip(
            u_k,
            cfg.U_MIN,
            cfg.U_MAX
        )

        q = plant.step(
            lam_series[k],
            mu_series[k],
            u=u_k
        )

        q_hist.append(q)

        e_prev2 = e_prev
        e_prev = e_k
        u_prev = u_k

    return np.asarray(q_hist)


def run_azfs(cfg, lam_series, mu_series):
    """
    Run the Adaptive Z-Domain Feedback Scheduler.

    Returns:

        q_hist
        ki_hist
        mu_hat_hist
        u_hist
    """

    plant = QueuePlant(
        cfg.Q_REF,
        cfg.Ts,
        cfg.U_MIN,
        cfg.U_MAX
    )

    controller = AZFSController(
        KI_nom=cfg.KI_NOM,
        mu_nom=cfg.MU_NOM,
        Ts=cfg.Ts,
        alpha=cfg.ALPHA,
        KP_DAMPING=cfg.KP_DAMPING
    )

    q_hist = []
    ki_hist = []
    mu_hat_hist = []
    u_hist = []

    for k in range(cfg.N_EPOCHS):

        q_star = cfg.Q_REF

        # Paper's error convention:
        #
        # e[k] = q*[k] - q[k]
        error = q_star - plant.q

        u_k, KI_k = controller.control(error)

        # Physical actuator limitation
        u_k = np.clip(
            u_k,
            cfg.U_MIN,
            cfg.U_MAX
        )

        q = plant.step(
            lam_series[k],
            mu_series[k],
            u=u_k
        )

        # Update service-rate estimate after observing current service rate
        controller.update_mu_estimate(mu_series[k])

        q_hist.append(q)
        ki_hist.append(KI_k)
        mu_hat_hist.append(controller.mu_hat)
        u_hist.append(u_k)

    return (
        np.asarray(q_hist),
        np.asarray(ki_hist),
        np.asarray(mu_hat_hist),
        np.asarray(u_hist)
    )


# ======================================================================
# Metrics
# ======================================================================

def compute_metrics(
    q_hist,
    q_ref,
    disturbance_epoch,
    ss_window=100,
    variance_window=200,
    settling_window=30,
    settling_frac=0.05,
    settling_floor=1.0
):

    disturbance_epoch = int(disturbance_epoch)

    # Only analyze the response to the actual workload step.
    response_q = np.asarray(q_hist[disturbance_epoch:], dtype=float)

    response_e = q_ref - response_q

    if len(response_q) == 0:
        raise ValueError("No samples available after disturbance epoch.")

    # --------------------------------------------------------------
    # Steady-state window
    # --------------------------------------------------------------

    ss_window = min(ss_window, len(response_q))

    ss_q = response_q[-ss_window:]
    ss_e = response_e[-ss_window:]

    # Signed bias
    signed_ss_error = float(np.mean(ss_e))

    # Average magnitude of error
    mean_abs_ss_error = float(np.mean(np.abs(ss_e)))

    # --------------------------------------------------------------
    # Steady-state variance
    # --------------------------------------------------------------

    variance_window = min(
        variance_window,
        len(response_q)
    )

    variance_ss = float(
        np.var(response_q[-variance_window:])
    )

    # --------------------------------------------------------------
    # Overshoot
    # --------------------------------------------------------------

    maximum_queue = float(np.max(response_q))

    overshoot = max(
        0.0,
        maximum_queue - q_ref
    )

    # --------------------------------------------------------------
    # Settling time
    # --------------------------------------------------------------

    # Error immediately after disturbance
    e0 = abs(response_e[0])

    band = max(
        settling_frac * e0,
        settling_floor
    )

    settling_window = min(
        settling_window,
        len(response_e)
    )

    settling_idx = None

    if len(response_e) >= settling_window:

        for k in range(
            len(response_e) - settling_window + 1
        ):

            window_error = np.abs(
                response_e[
                    k:k + settling_window
                ]
            )

            if np.mean(window_error) <= band:

                settling_idx = k
                break

    if settling_idx is None:
        settling_time = -1
    else:
        settling_time = int(settling_idx)

    return {
        "signed_ss_error": signed_ss_error,
        "mean_abs_ss_error": mean_abs_ss_error,
        "settling_time_epochs": settling_time,
        "max_overshoot": overshoot,
        "queue_variance_ss": variance_ss,
        "settling_band": band
    }


# ======================================================================
# Formatting
# ======================================================================

def print_summary(metrics):
    """
    Print metrics in a readable table.
    """

    print()

    print(
        f"{'Scheduler':<15}"
        f"{'Signed SS Error':>18}"
        f"{'Mean Abs SS Error':>20}"
        f"{'Settling':>12}"
        f"{'Overshoot':>14}"
        f"{'SS Variance':>16}"
    )

    print("-" * 95)

    for name, m in metrics.items():

        print(
            f"{name:<15}"
            f"{m['signed_ss_error']:>18.4f}"
            f"{m['mean_abs_ss_error']:>20.4f}"
            f"{m['settling_time_epochs']:>12d}"
            f"{m['max_overshoot']:>14.4f}"
            f"{m['queue_variance_ss']:>16.4f}"
        )


# ======================================================================
# CSV Export
# ======================================================================

def save_csv(
    filename,
    cfg,
    lam_series,
    mu_series,
    results
):
    """
    Save epoch-level results.
    """

    with open(
        filename,
        "w",
        newline=""
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "epoch",
            "lambda",
            "mu",
            "disturbance",
            "q_FCFS",
            "q_RoundRobin",
            "q_PID_Tustin",
            "q_AZFS",
            "error_AZFS",
            "KI_AZFS",
            "mu_hat_AZFS",
            "u_AZFS"
        ])

        for k in range(cfg.N_EPOCHS):

            disturbance = (
                1
                if k >= cfg.STEP_EPOCH
                else 0
            )

            azfs_error = (
                cfg.Q_REF
                - results["AZFS"][k]
            )

            writer.writerow([
                k,
                lam_series[k],
                mu_series[k],
                disturbance,
                results["FCFS"][k],
                results["RoundRobin"][k],
                results["PID_Tustin"][k],
                results["AZFS"][k],
                azfs_error,
                results["KI_AZFS"][k],
                results["MU_HAT_AZFS"][k],
                results["U_AZFS"][k]
            ])


# ======================================================================
# Summary File
# ======================================================================

def save_summary(
    filename,
    cfg,
    metrics
):
    """
    Save a human-readable performance summary.
    """

    with open(filename, "w") as f:

        f.write(
            "AZFS Simulation Performance Summary\n"
        )

        f.write(
            "===================================\n\n"
        )

        f.write(
            f"Epochs: {cfg.N_EPOCHS}\n"
        )

        f.write(
            f"Sampling interval Ts: {cfg.Ts}\n"
        )

        f.write(
            f"Reference queue q*: {cfg.Q_REF}\n"
        )

        f.write(
            f"Initial arrival rate lambda: "
            f"{cfg.LAMBDA_NOM}\n"
        )

        f.write(
            f"Step epoch: {cfg.STEP_EPOCH}\n"
        )

        f.write(
            f"Step size: {cfg.STEP_SIZE}\n"
        )

        f.write(
            f"Post-step arrival rate: "
            f"{cfg.LAMBDA_NOM + cfg.STEP_SIZE}\n"
        )

        f.write(
            f"Nominal service rate mu: "
            f"{cfg.MU_NOM}\n"
        )

        f.write(
            f"Service-rate drift: "
            f"{cfg.SERVICE_RATE_DRIFT}\n"
        )

        f.write(
            f"Random seed: {cfg.SEED}\n\n"
        )

        f.write(
            "AZFS Controller\n"
        )

        f.write(
            "---------------\n"
        )

        f.write(
            f"KI nominal: {cfg.KI_NOM}\n"
        )

        f.write(
            f"KP damping: {cfg.KP_DAMPING}\n"
        )

        f.write(
            f"EWMA alpha: {cfg.ALPHA}\n"
        )

        f.write(
            f"Actuator range: "
            f"[{cfg.U_MIN}, {cfg.U_MAX}]\n\n"
        )

        f.write(
            "Metrics\n"
        )

        f.write(
            "-------\n"
        )

        f.write(
            f"{'Scheduler':<15}"
            f"{'Signed SS Error':>18}"
            f"{'Mean Abs SS Error':>20}"
            f"{'Settling':>12}"
            f"{'Overshoot':>14}"
            f"{'SS Variance':>16}\n"
        )

        f.write("-" * 95 + "\n")

        for name, m in metrics.items():

            f.write(
                f"{name:<15}"
                f"{m['signed_ss_error']:>18.4f}"
                f"{m['mean_abs_ss_error']:>20.4f}"
                f"{m['settling_time_epochs']:>12d}"
                f"{m['max_overshoot']:>14.4f}"
                f"{m['queue_variance_ss']:>16.4f}\n"
            )

        f.write("\n")

        f.write(
            "Metric definitions\n"
        )

        f.write(
            "------------------\n"
        )

        f.write(
            "Signed SS Error: mean(q* - q) over the final "
            f"{cfg.SS_WINDOW} post-disturbance epochs.\n"
        )

        f.write(
            "Mean Abs SS Error: mean(abs(q* - q)) over the "
            f"final {cfg.SS_WINDOW} post-disturbance epochs.\n"
        )

        f.write(
            "Settling Time: first post-disturbance epoch at "
            "which the absolute error remains inside the "
            f"configured settling band for {cfg.SETTLING_WINDOW} "
            "consecutive epochs.\n"
        )

        f.write(
            "Overshoot: maximum queue depth above q* after "
            "the workload disturbance.\n"
        )

        f.write(
            "SS Variance: queue-depth variance over the final "
            f"{cfg.VARIANCE_WINDOW} post-disturbance epochs.\n"
        )


# ======================================================================
# Plotting
# ======================================================================

def plot_comparison(
    filename,
    cfg,
    results
):
    """
    Full-scale comparison.

    Shows queue depth and queue error.
    """

    epochs = np.arange(cfg.N_EPOCHS)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(11, 8),
        sharex=True
    )

    # --------------------------------------------------------------
    # Queue depth
    # --------------------------------------------------------------

    ax = axes[0]

    for name in [
        "FCFS",
        "RoundRobin",
        "PID_Tustin",
        "AZFS"
    ]:

        ax.plot(
            epochs,
            results[name],
            label=name,
            linewidth=1.1
        )

    ax.axhline(
        cfg.Q_REF,
        linestyle="--",
        linewidth=1.0,
        label="q* (reference)"
    )

    ax.axvline(
        cfg.STEP_EPOCH,
        linestyle=":",
        linewidth=1.0,
        label="Workload step"
    )

    ax.set_ylabel(
        "Queue depth q[k]"
    )

    ax.set_title(
        "Queue Depth Response: AZFS vs. Baselines"
    )

    ax.legend()

    ax.grid(alpha=0.2)

    # --------------------------------------------------------------
    # Queue error
    # --------------------------------------------------------------

    ax2 = axes[1]

    for name in [
        "FCFS",
        "RoundRobin",
        "PID_Tustin",
        "AZFS"
    ]:

        error = cfg.Q_REF - results[name]

        ax2.plot(
            epochs,
            error,
            label=f"{name} error",
            linewidth=1.0
        )

    ax2.axhline(
        0,
        linestyle="--",
        linewidth=1.0
    )

    ax2.axvline(
        cfg.STEP_EPOCH,
        linestyle=":",
        linewidth=1.0
    )

    ax2.set_xlabel(
        "Epoch k"
    )

    ax2.set_ylabel(
        "Queue error e[k]"
    )

    ax2.set_title(
        "Queue Error Over Time"
    )

    ax2.legend()

    ax2.grid(alpha=0.2)

    plt.tight_layout()

    plt.savefig(
        filename,
        dpi=150
    )

    plt.close(fig)


def plot_zoomed_controller_comparison(
    filename,
    cfg,
    results
):
    """
    Controller-only comparison.

    FCFS and Round Robin are excluded because their unbounded queue
    growth would dominate the vertical scale.
    """

    epochs = np.arange(cfg.N_EPOCHS)

    fig, ax = plt.subplots(
        figsize=(11, 5)
    )

    ax.plot(
        epochs,
        results["AZFS"],
        label="AZFS",
        linewidth=1.4
    )

    ax.plot(
        epochs,
        results["PID_Tustin"],
        label="PID (Tustin)",
        linewidth=1.4,
        linestyle="--"
    )

    ax.axhline(
        cfg.Q_REF,
        linestyle=":",
        linewidth=1.0,
        label="q* (reference)"
    )

    ax.axvline(
        cfg.STEP_EPOCH,
        linestyle=":",
        linewidth=1.0,
        label="Workload step"
    )

    ax.set_xlabel(
        "Epoch k"
    )

    ax.set_ylabel(
        "Queue depth q[k]"
    )

    ax.set_title(
        "AZFS vs. Tustin-PID Under Step Workload Disturbance"
    )

    ax.legend()

    ax.grid(alpha=0.2)

    plt.tight_layout()

    plt.savefig(
        filename,
        dpi=150
    )

    plt.close(fig)


def plot_adaptation(
    filename,
    cfg,
    mu_series,
    azfs_mu_hat,
    azfs_ki
):
    """
    Plot service-rate variation, EWMA estimate and adaptive KI.
    """

    epochs = np.arange(cfg.N_EPOCHS)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(11, 8),
        sharex=True
    )

    # --------------------------------------------------------------
    # Service rate
    # --------------------------------------------------------------

    ax = axes[0]

    ax.plot(
        epochs,
        mu_series,
        label="Observed μ[k]",
        linewidth=0.9
    )

    ax.plot(
        epochs,
        azfs_mu_hat,
        label="EWMA μ_hat[k]",
        linewidth=1.5
    )

    ax.axhline(
        cfg.MU_NOM,
        linestyle="--",
        linewidth=1.0,
        label="Nominal μ"
    )

    ax.axvline(
        cfg.STEP_EPOCH,
        linestyle=":",
        linewidth=1.0
    )

    ax.set_ylabel(
        "Service rate"
    )

    ax.set_title(
        "Service-Rate Variation and EWMA Estimate"
    )

    ax.legend()

    ax.grid(alpha=0.2)

    # --------------------------------------------------------------
    # Adaptive gain
    # --------------------------------------------------------------

    ax2 = axes[1]

    ax2.plot(
        epochs,
        azfs_ki,
        label="Adaptive KI[k]",
        linewidth=1.3
    )

    ax2.axhline(
        cfg.KI_NOM,
        linestyle="--",
        linewidth=1.0,
        label="Nominal KI"
    )

    ax2.axvline(
        cfg.STEP_EPOCH,
        linestyle=":",
        linewidth=1.0
    )

    ax2.set_xlabel(
        "Epoch k"
    )

    ax2.set_ylabel(
        "KI[k]"
    )

    ax2.set_title(
        "AZFS Adaptive Integral Gain"
    )

    ax2.legend()

    ax2.grid(alpha=0.2)

    plt.tight_layout()

    plt.savefig(
        filename,
        dpi=150
    )

    plt.close(fig)


# ======================================================================
# Single Experiment
# ======================================================================

def run_single_experiment(cfg, seed=None):
    """
    Run one complete experiment.
    """

    if seed is None:
        seed = cfg.SEED

    lam_series, mu_series = generate_profiles(
        cfg,
        seed=seed
    )

    # --------------------------------------------------------------
    # Stability information
    # --------------------------------------------------------------

    jury_lower, jury_upper = (
        AZFSController.jury_stable_range(
            cfg.MU_NOM,
            cfg.Ts
        )
    )

    print(
        f"Jury nominal KI interval: "
        f"({jury_lower:.4f}, {jury_upper:.4f})"
    )

    print(
        f"AZFS nominal KI: {cfg.KI_NOM:.4f}"
    )

    # --------------------------------------------------------------
    # Run schedulers
    # --------------------------------------------------------------

    fcfs = run_fcfs(
        cfg,
        lam_series,
        mu_series
    )

    round_robin = run_round_robin(
        cfg,
        lam_series,
        mu_series
    )

    pid = run_pid_tustin(
        cfg,
        lam_series,
        mu_series
    )

    (
        azfs,
        azfs_ki,
        azfs_mu_hat,
        azfs_u
    ) = run_azfs(
        cfg,
        lam_series,
        mu_series
    )

    results = {
        "FCFS": fcfs,
        "RoundRobin": round_robin,
        "PID_Tustin": pid,
        "AZFS": azfs,

        # Additional AZFS signals
        "KI_AZFS": azfs_ki,
        "MU_HAT_AZFS": azfs_mu_hat,
        "U_AZFS": azfs_u
    }

    # --------------------------------------------------------------
    # Metrics
    # --------------------------------------------------------------

    metrics = {}

    for name in [
        "FCFS",
        "RoundRobin",
        "PID_Tustin",
        "AZFS"
    ]:

        metrics[name] = compute_metrics(
            results[name],
            cfg.Q_REF,
            disturbance_epoch=cfg.STEP_EPOCH,
            ss_window=cfg.SS_WINDOW,
            variance_window=cfg.VARIANCE_WINDOW,
            settling_window=cfg.SETTLING_WINDOW,
            settling_frac=cfg.SETTLING_FRAC,
            settling_floor=cfg.SETTLING_FLOOR
        )

    return (
        lam_series,
        mu_series,
        results,
        metrics
    )


# ======================================================================
# Multi-Seed Experiments
# ======================================================================

def run_multi_seed(cfg, seeds):
    """
    Run multiple stochastic experiments and aggregate:

        mean ± standard deviation

    for each performance metric.
    """

    all_metrics = {
        "FCFS": [],
        "RoundRobin": [],
        "PID_Tustin": [],
        "AZFS": []
    }

    for seed in seeds:

        print(
            f"\nRunning seed {seed}..."
        )

        _, _, _, metrics = run_single_experiment(
            cfg,
            seed=seed
        )

        for scheduler in all_metrics:

            all_metrics[scheduler].append(
                metrics[scheduler]
            )

    metric_names = [
        "signed_ss_error",
        "mean_abs_ss_error",
        "settling_time_epochs",
        "max_overshoot",
        "queue_variance_ss"
    ]

    rows = []

    for scheduler in all_metrics:

        row = {
            "scheduler": scheduler
        }

        metric_list = all_metrics[scheduler]

        for metric_name in metric_names:

            values = np.array([
                m[metric_name]
                for m in metric_list
            ], dtype=float)

            # Settling time of -1 means "not settled".
            # Report it separately as the raw mean/std, while also
            # calculating a settled-only statistic when applicable.
            if metric_name == "settling_time_epochs":

                valid = values[values >= 0]

                row[f"{metric_name}_mean"] = (
                    float(np.mean(values))
                    if len(values)
                    else float("nan")
                )

                row[f"{metric_name}_std"] = (
                    float(np.std(values, ddof=1))
                    if len(values) > 1
                    else 0.0
                )

                row[f"{metric_name}_success_rate"] = (
                    float(np.mean(values >= 0))
                )

                row[f"{metric_name}_settled_mean"] = (
                    float(np.mean(valid))
                    if len(valid)
                    else float("nan")
                )

            else:

                row[f"{metric_name}_mean"] = (
                    float(np.mean(values))
                )

                row[f"{metric_name}_std"] = (
                    float(np.std(values, ddof=1))
                    if len(values) > 1
                    else 0.0
                )

        rows.append(row)

    return rows


def save_multiseed_results(
    filename,
    rows
):
    """
    Save multi-seed aggregate results.
    """

    fields = [
        "scheduler",

        "signed_ss_error_mean",
        "signed_ss_error_std",

        "mean_abs_ss_error_mean",
        "mean_abs_ss_error_std",

        "settling_time_epochs_mean",
        "settling_time_epochs_std",
        "settling_time_epochs_success_rate",
        "settling_time_epochs_settled_mean",

        "max_overshoot_mean",
        "max_overshoot_std",

        "queue_variance_ss_mean",
        "queue_variance_ss_std"
    ]

    with open(
        filename,
        "w",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(row)


def save_multiseed_summary(
    filename,
    rows,
    seeds
):
    """
    Save a readable multi-seed summary.
    """

    with open(filename, "w", encoding="utf-8") as f:

        f.write(
            "AZFS Multi-Seed Simulation Summary\n"
        )

        f.write(
            "=================================\n\n"
        )

        f.write(
            f"Seeds: {', '.join(map(str, seeds))}\n\n"
        )

        f.write(
            "Values are reported as mean ± standard deviation.\n"
        )

        f.write(
            "Settling success rate is the fraction of runs that "
            "satisfied the configured settling criterion.\n\n"
        )

        f.write(
            f"{'Scheduler':<15}"
            f"{'Signed Error':>22}"
            f"{'Abs Error':>22}"
            f"{'Overshoot':>30}"
            f"{'Variance':>30}"
            f"{'Settle Success':>18}\n"
        )

        f.write("-" * 150 + "\n")

        for row in rows:

            signed = (
                f"{row['signed_ss_error_mean']:.4f} ± "
                f"{row['signed_ss_error_std']:.4f}"
            )

            abs_error = (
                f"{row['mean_abs_ss_error_mean']:.4f} ± "
                f"{row['mean_abs_ss_error_std']:.4f}"
            )

            overshoot = (
                f"{row['max_overshoot_mean']:.4f} ± "
                f"{row['max_overshoot_std']:.4f}"
            )

            variance = (
                f"{row['queue_variance_ss_mean']:.4f} ± "
                f"{row['queue_variance_ss_std']:.4f}"
            )

            success = (
                f"{100.0 * row['settling_time_epochs_success_rate']:.1f}%"
            )

            f.write(
                f"{row['scheduler']:<15}"
                f"{signed:>22}"
                f"{abs_error:>22}"
                f"{overshoot:>30}"
                f"{variance:>30}"
                f"{success:>18}\n"
            )

        f.write("\n")

        f.write(
            "Settling time details\n"
        )

        f.write(
            "---------------------\n"
        )

        for row in rows:

            f.write(
                f"{row['scheduler']}:\n"
            )

            f.write(
                f"  Mean raw settling time: "
                f"{row['settling_time_epochs_mean']:.4f}\n"
            )

            f.write(
                f"  Settled-run mean: "
                f"{row['settling_time_epochs_settled_mean']}\n"
            )

            f.write(
                f"  Success rate: "
                f"{100.0 * row['settling_time_epochs_success_rate']:.1f}%\n\n"
            )


# ======================================================================
# Main
# ======================================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "AZFS Adaptive Z-Domain Feedback Scheduler simulation"
        )
    )

    parser.add_argument(
        "--multi-seed",
        action="store_true",
        help="Run multiple stochastic seeds"
    )

    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        help="Seeds used with --multi-seed"
    )

    args = parser.parse_args()

    cfg = Config()

    print(
        "=============================================="
    )

    print(
        "AZFS Adaptive Z-Domain Feedback Scheduler"
    )

    print(
        "=============================================="
    )

    print()

    print(
        f"Simulation epochs: {cfg.N_EPOCHS}"
    )

    print(
        f"Reference queue: {cfg.Q_REF}"
    )

    print(
        f"Workload step: epoch {cfg.STEP_EPOCH}"
    )

    print(
        f"Lambda: {cfg.LAMBDA_NOM} -> "
        f"{cfg.LAMBDA_NOM + cfg.STEP_SIZE}"
    )

    print(
        f"Nominal mu: {cfg.MU_NOM}"
    )

    print(
        f"Service-rate drift: "
        f"{cfg.SERVICE_RATE_DRIFT}"
    )

    print(
        f"AZFS KI_nom: {cfg.KI_NOM}"
    )

    print(
        f"AZFS KP: {cfg.KP_DAMPING}"
    )

    print()

    # ==============================================================
    # Single experiment
    # ==============================================================

    (
        lam_series,
        mu_series,
        results,
        metrics
    ) = run_single_experiment(
        cfg,
        seed=cfg.SEED
    )

    print(
        "\nSingle-seed results:"
    )

    print_summary(metrics)

    # ==============================================================
    # Save CSV
    # ==============================================================

    save_csv(
        "azfs_results.csv",
        cfg,
        lam_series,
        mu_series,
        results
    )

    # ==============================================================
    # Save summary
    # ==============================================================

    save_summary(
        "azfs_summary.txt",
        cfg,
        metrics
    )

    # ==============================================================
    # Generate plots
    # ==============================================================

    plot_comparison(
        "azfs_comparison.png",
        cfg,
        results
    )

    plot_zoomed_controller_comparison(
        "azfs_zoomed_comparison.png",
        cfg,
        results
    )

    plot_adaptation(
        "azfs_adaptation.png",
        cfg,
        mu_series,
        results["MU_HAT_AZFS"],
        results["KI_AZFS"]
    )

    print()

    print(
        "Generated:"
    )

    print(
        "  azfs_results.csv"
    )

    print(
        "  azfs_summary.txt"
    )

    print(
        "  azfs_comparison.png"
    )

    print(
        "  azfs_zoomed_comparison.png"
    )

    print(
        "  azfs_adaptation.png"
    )

    # ==============================================================
    # Multi-seed
    # ==============================================================

    if args.multi_seed:

        print(
            "\n=============================================="
        )

        print(
            "Running multi-seed evaluation"
        )

        print(
            "=============================================="
        )

        rows = run_multi_seed(
            cfg,
            args.seeds
        )

        save_multiseed_results(
            "azfs_multiseed_results.csv",
            rows
        )

        save_multiseed_summary(
            "azfs_multiseed_summary.txt",
            rows,
            args.seeds
        )

        print(
            "\nMulti-seed evaluation complete."
        )

        print(
            "Generated:"
        )

        print(
            "  azfs_multiseed_results.csv"
        )

        print(
            "  azfs_multiseed_summary.txt"
        )


# ======================================================================
# Entry Point
# ======================================================================

if __name__ == "__main__":
    main()