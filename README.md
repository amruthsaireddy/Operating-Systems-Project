# Adaptive Z-Domain Feedback Scheduler (AZFS)

A simulation-based implementation of an **Adaptive Z-Domain Feedback Scheduler (AZFS)** for queue/workload regulation under changing workload and service-rate conditions.

The project models a discrete-time queue and evaluates AZFS against three baseline scheduling/control approaches:

* **FCFS (First-Come, First-Served)**
* **Round Robin**
* **Tustin-discretized PID**
* **AZFS (Adaptive Z-Domain Feedback Scheduler)**

The simulator evaluates queue tracking, steady-state error, overshoot, settling behavior, and steady-state variance under workload disturbances and service-rate variation.

---

## Table of Contents

* [Overview](#overview)
* [System Model](#system-model)
* [AZFS Controller](#azfs-controller)
* [Adaptive Service-Rate Estimation](#adaptive-service-rate-estimation)
* [Schedulers](#schedulers)
* [Simulation Configuration](#simulation-configuration)
* [Performance Metrics](#performance-metrics)
* [Project Structure](#project-structure)
* [Requirements](#requirements)
* [Installation](#installation)
* [Running the Simulation](#running-the-simulation)
* [Multi-Seed Evaluation](#multi-seed-evaluation)
* [Generated Outputs](#generated-outputs)
* [Understanding the Results](#understanding-the-results)
* [Implementation Notes](#implementation-notes)
* [Reproducibility](#reproducibility)
* [Limitations](#limitations)
* [Future Work](#future-work)
* [License](#license)

---

## Overview

AZFS is designed as a feedback-based scheduling approach for maintaining a queue around a desired reference value despite changes in workload and service capacity.

The simulation consists of:

1. A discrete-time queue plant.
2. A workload model with a step disturbance.
3. A time-varying service-rate model.
4. An adaptive AZFS PI controller.
5. Baseline schedulers/controllers.
6. Performance metric calculation.
7. CSV and text-based result export.
8. Visualization of queue behavior and controller adaptation.
9. Optional multi-seed stochastic evaluation.

The implementation uses identical workload and service-rate sequences for the different schedulers during each experiment so that their responses can be compared under the same disturbance conditions.

---

## System Model

The queue is modeled in discrete time as:

```text
q[k+1] = q[k] + λ[k]Ts - μ[k]u[k]Ts
```

where:

| Symbol | Description                    |
| ------ | ------------------------------ |
| `q[k]` | Queue depth at epoch `k`       |
| `λ[k]` | Workload/arrival rate          |
| `μ[k]` | Service rate                   |
| `u[k]` | Dispatch/service control value |
| `Ts`   | Sampling/scheduling interval   |

The actuator is constrained by:

```text
U_MIN ≤ u[k] ≤ U_MAX
```

In the current configuration:

```text
U_MIN = 0
U_MAX = 3
```

The queue is also constrained to remain non-negative.

---

## Workload Model

The simulation starts with a nominal workload:

```text
λ_nom = 12
```

At the configured disturbance epoch, the workload experiences a step increase.

Current configuration:

```text
STEP_EPOCH = 50
STEP_SIZE  = 8
```

Therefore:

```text
Before epoch 50:
λ = 12

From epoch 50 onward:
λ = 20
```

This step disturbance is used to evaluate how each scheduler responds to a sudden increase in workload.

---

## Service-Rate Model

The nominal service rate is:

```text
μ_nom = 10
```

When service-rate drift is enabled, the simulation generates:

```text
μ[k] = μ_nom + Gaussian noise
```

with:

```text
SERVICE_RATE_STD = 0.6
```

The generated service-rate sequence is clipped to:

```text
0.5 × μ_nom ≤ μ[k] ≤ 1.5 × μ_nom
```

The same service-rate sequence is supplied to every scheduler in a given experiment.

---

# AZFS Controller

The theoretical controller described in the project is the pure integral controller:

```text
C(z) = KI · z / (z - 1)
```

The queue plant already contains an integrating pole at:

```text
z = 1
```

Therefore, directly combining the pure integral controller with the integrating queue plant produces a double-integrator loop.

To introduce damping, the implementation uses a PI refinement:

```text
C(z) = KI · z / (z - 1) + KP
```

The controller is implemented using the following incremental equation:

```text
u[k] = u[k-1]
       + KI[k] · e[k]
       + KP · (e[k] - e[k-1])
```

where:

```text
e[k] = q* - q[k]
```

and:

* `q*` is the target queue depth.
* `q[k]` is the current queue depth.
* `KI[k]` is the adaptive integral gain.
* `KP` is the damping/proportional gain.

The current target queue is:

```text
q* = 20
```

---

## Error Convention

The implementation uses:

```text
e[k] = q* - q[k]
```

With the current negative controller gains:

```text
KI < 0
KP < 0
```

the controller behaves as follows:

### Queue below target

```text
q[k] < q*
e[k] > 0
```

The negative gain reduces the control signal.

### Queue above target

```text
q[k] > q*
e[k] < 0
```

The negative gain increases the control signal.

Increasing `u[k]` increases service and therefore drains the queue.

---

# Adaptive Service-Rate Estimation

AZFS estimates the current service rate using an **Exponentially Weighted Moving Average (EWMA)**.

The estimate is:

```text
μ_hat[k] =
    α · μ_hat[k-1]
    + (1 - α) · μ_obs[k]
```

where:

* `μ_hat[k]` = estimated service rate.
* `μ_obs[k]` = observed service rate.
* `α` = EWMA smoothing factor.

The current configuration uses:

```text
α = 0.9
```

A larger `α` gives more weight to the previous estimate and produces a smoother but slower response.

---

## Adaptive Integral Gain

The integral gain is adapted according to the estimated service rate:

```text
KI[k] = KI_nom · μ_nom / μ_hat[k]
```

Current nominal gain:

```text
KI_nom = -0.1
```

This allows the controller gain to change as the estimated service capacity changes.

The simulator records:

* `KI[k]`
* `μ_hat[k]`
* `u[k]`

for every simulation epoch.

---

# Schedulers

## 1. FCFS

The FCFS baseline uses a fixed dispatch value:

```text
u = 1
```

No feedback control or service-rate adaptation is applied.

This provides a simple non-adaptive reference.

---

## 2. Round Robin

The Round Robin baseline is represented using a fixed dispatch fraction:

```text
u = 0.9
```

No feedback or adaptive control is applied.

This represents a static time-sharing scheduling policy.

---

## 3. Tustin PID

A conventional PID controller is included as a feedback baseline.

The PID uses a Tustin-discretized implementation.

The current default parameters are:

```text
Kp = 0.08
Ki = 0.03
Kd = 0.01
```

The PID uses the error convention:

```text
e[k] = q[k] - q*
```

A positive error therefore indicates that the queue is above the target and the controller increases service.

The resulting control signal is saturated using the same actuator limits as AZFS.

---

## 4. AZFS

AZFS combines:

* Z-domain feedback control
* Integral action
* Proportional damping
* EWMA service-rate estimation
* Adaptive integral gain
* Actuator saturation

The current parameters are:

```text
KI_nom      = -0.1
KP_DAMPING  = -0.1
ALPHA       = 0.9
```

---

# Simulation Configuration

The default simulation configuration is:

| Parameter                       | Value |
| ------------------------------- | ----: |
| Simulation epochs               |  2000 |
| Sampling interval `Ts`          |   1.0 |
| Initial arrival rate            |  12.0 |
| Workload step epoch             |    50 |
| Workload step size              |   8.0 |
| Post-step arrival rate          |  20.0 |
| Reference queue                 |  20.0 |
| Nominal service rate            |  10.0 |
| Service-rate standard deviation |   0.6 |
| `KI_nom`                        |  -0.1 |
| `KP_DAMPING`                    |  -0.1 |
| EWMA `α`                        |   0.9 |
| Minimum actuator                |   0.0 |
| Maximum actuator                |   3.0 |
| Default random seed             |    42 |

These values are defined in the `Config` dataclass in `azfs_sim.py`.

---

# Performance Metrics

The simulation evaluates each scheduler using several metrics.

## 1. Signed Steady-State Error

The signed steady-state error is:

```text
mean(q* - q)
```

over the final configured steady-state window.

A value close to zero indicates little systematic steady-state bias.

The default steady-state window is:

```text
100 epochs
```

---

## 2. Mean Absolute Steady-State Error

Defined as:

```text
mean(|q* - q|)
```

over the final steady-state window.

Unlike signed error, positive and negative errors do not cancel each other.

This metric is useful for determining the actual magnitude of tracking error.

---

## 3. Settling Time

Settling time is measured after the workload disturbance.

The simulator determines an error band based on:

```text
band = max(
    settling_fraction × initial_error,
    settling_floor
)
```

The current configuration is:

```text
Settling window = 30 epochs
Settling fraction = 5%
Settling floor = 1.0
```

A system is considered settled when the mean absolute error remains inside the configured band for the required consecutive window.

If the criterion is never satisfied:

```text
settling_time_epochs = -1
```

which represents **not settled**.

---

## 4. Maximum Overshoot

Overshoot is calculated as:

```text
max(0, maximum_queue - q*)
```

It therefore measures how far the queue rises above the target following the disturbance.

---

## 5. Steady-State Variance

The steady-state queue variance is:

```text
Var(q)
```

calculated over the final configured variance window.

The default variance window is:

```text
200 epochs
```

A lower variance indicates less fluctuation around the steady-state operating point.

---

# Project Structure

A typical repository can be organized as:

```text
AZFS/
│
├── azfs_sim.py
├── README.md
│
├── azfs_results.csv
├── azfs_summary.txt
│
├── azfs_comparison.png
├── azfs_zoomed_comparison.png
├── azfs_adaptation.png
│
├── azfs_multiseed_results.csv
└── azfs_multiseed_summary.txt
```

The generated result files are produced after running the simulation.

---

# Requirements

The project requires Python 3 and the following Python packages:

```text
numpy
matplotlib
```

The simulation also uses Python standard-library modules including:

```text
argparse
csv
dataclasses
```

---

# Installation

Clone the repository

Create a virtual environment:

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install numpy matplotlib
```

---

# Running the Simulation

Run a standard single-seed experiment:

```bash
python azfs_sim.py
```

The default experiment uses:

```text
Seed = 42
```

The simulation runs all four schedulers:

```text
FCFS
Round Robin
PID (Tustin)
AZFS
```

and calculates their performance metrics.

---

# Multi-Seed Evaluation

To evaluate performance across multiple random seeds:

```bash
python azfs_sim.py --multi-seed
```

The default seeds are:

```text
1 2 3 4 5 6 7 8 9 10
```

Custom seeds can also be supplied:

```bash
python azfs_sim.py --multi-seed --seeds 1 2 3 4 5
```

Multi-seed evaluation is useful because the service-rate profile contains stochastic variation.

Instead of relying on a single random realization, the simulator reports:

```text
mean ± standard deviation
```

for the performance metrics.

For settling time, the simulator additionally reports the fraction of runs that successfully satisfied the settling criterion.

---

# Generated Outputs

## Single-Seed Outputs

Running:

```bash
python azfs_sim.py
```

generates:

### `azfs_results.csv`

Epoch-level simulation data including:

```text
epoch
lambda
mu
disturbance
q_FCFS
q_RoundRobin
q_PID_Tustin
q_AZFS
error_AZFS
KI_AZFS
mu_hat_AZFS
u_AZFS
```

This file can be used for further analysis or plotting.

---

### `azfs_summary.txt`

A human-readable summary containing:

* Simulation configuration
* AZFS controller parameters
* Scheduler metrics
* Metric definitions

---

### `azfs_comparison.png`

Full-scale comparison of:

* Queue depth
* Queue error

for all schedulers.

---

### `azfs_zoomed_comparison.png`

Focused comparison between:

* AZFS
* Tustin PID

This avoids the large queue growth of the static baselines dominating the plot scale.

---

### `azfs_adaptation.png`

Visualization of:

1. Observed service rate
2. EWMA service-rate estimate
3. Adaptive integral gain

This provides visibility into the adaptive portion of AZFS.

---

# Multi-Seed Outputs

When using:

```bash
python azfs_sim.py --multi-seed
```

the following additional files are generated:

### `azfs_multiseed_results.csv`

Contains aggregated mean and standard deviation values for the evaluated metrics.

---

### `azfs_multiseed_summary.txt`

Provides a human-readable multi-seed summary including:

* Seeds used
* Mean ± standard deviation
* Signed steady-state error
* Mean absolute steady-state error
* Overshoot
* Steady-state variance
* Settling success rate
* Settling-time details

---

# Understanding the Results

The primary purpose of the simulation is to observe how the schedulers respond to the workload step at:

```text
epoch = 50
```

The target queue is:

```text
q* = 20
```

When interpreting the results, consider the metrics together rather than relying on a single value.

### Tracking

Use:

```text
Signed SS Error
Mean Abs SS Error
```

to evaluate how closely the queue tracks the reference.

### Transient Response

Use:

```text
Overshoot
Settling Time
```

to examine the response immediately following the workload disturbance.

### Stability / Fluctuation

Use:

```text
Queue Variance
```

to examine steady-state fluctuations.

### Adaptation

Use:

```text
μ_hat[k]
KI[k]
u[k]
```

to understand how AZFS responds to changes in service rate.

---

# Stability Considerations

The implementation includes a helper function that reports the nominal gain interval stated by the theoretical analysis:

```text
-4 / (μTs) < KI < 0
```

For the configured nominal values:

```text
μ = 10
Ts = 1
```

the reported interval is:

```text
-0.4 < KI < 0
```

The implementation also explicitly notes that the pure integral controller combined with the integrating queue plant requires additional care because of the resulting double-integrator structure.

The implemented proportional damping term is therefore used as a practical PI refinement.

---

# Actuator Saturation

The controller output is constrained to:

```text
0 ≤ u[k] ≤ 3
```

This represents a physical limitation on the available dispatch/service control.

Because of this saturation, the simulated system is nonlinear whenever the controller attempts to produce a value outside the actuator range.

Therefore, theoretical linear-controller stability analysis should not automatically be interpreted as a guarantee of stability for the saturated simulation.

---

# Reproducibility

The simulator uses NumPy's random number generator with an explicit seed.

The default single-run seed is:

```text
42
```

Multi-seed experiments use:

```text
1, 2, 3, 4, 5, 6, 7, 8, 9, 10
```

Because each scheduler receives the same generated workload and service-rate sequences within an experiment, the scheduler comparison is performed under identical disturbance conditions.

---

# Limitations

The current implementation is a simulation model rather than a deployment-ready scheduler.

Important limitations include:

1. **Simplified queue dynamics**

   The plant is represented by a single queue equation:

   ```text
   q[k+1] = q[k] + λ[k]Ts - μ[k]u[k]Ts
   ```

2. **Simplified scheduling baselines**

   FCFS and Round Robin are represented through fixed dispatch fractions rather than complete packet/task scheduling implementations.

3. **Synthetic workload**

   The workload consists primarily of a nominal arrival rate followed by a step disturbance.

4. **Synthetic service-rate variation**

   Service-rate changes are generated using Gaussian stochastic variation.

5. **Actuator saturation**

   Saturation introduces nonlinear behavior that is not fully captured by simple linear Z-domain analysis.

6. **Simulation-specific controller parameters**

   The current gains and configuration values are experimental parameters and should not be interpreted as universally optimal values.

---

# Future Work

Possible extensions include:

* More realistic workload traces
* Multiple simultaneous queues
* Different workload disturbance types
* More realistic FCFS and Round Robin implementations
* Dynamic actuator constraints
* Formal closed-loop stability analysis of the PI refinement
* Anti-windup mechanisms
* More sophisticated service-rate prediction
* Additional adaptive control strategies
* Large-scale parameter sweeps
* Confidence intervals for multi-seed experiments
* Statistical significance testing
* Real-world scheduler integration
* Comparison against additional adaptive controllers

---




