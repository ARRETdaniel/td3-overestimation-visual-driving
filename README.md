# Does removing critic overestimation make a better visual driver? Code, configurations and per-run results

This repository accompanies the paper *Does Removing Critic Overestimation Make a Better Visual Driver? A
Pre-Registered TD3 Study in CARLA*. It holds the system that trained and evaluated the agents, the configuration of
every arm the paper reports, the per-run results behind its tables and figures, and the scripts that compute the
critic's bias and the statistics.

The agents drive urban routes with traffic in the [CARLA](https://carla.org) 0.9.16 simulator, from four stacked
84×84 grey front-camera frames and a route vector, with continuous steering and throttle/brake. The paper compares:

- **TD3** against **AHE**, the TD3 authors' ablation baseline: the same code with one critic instead of two, a policy
  update at every step and no target-action noise;
- two controls: **AHE-2H** (AHE with two critic heads fitted to AHE's target) and **AHE+CDQ** (AHE with clipped
  double Q-learning alone);
- the same pairs with an actor-smoothness regulariser (**CAPS**) in both arms.

The critic's bias is the critic's estimate minus the discounted return the policy then realised from the same
evaluation state, Q1 − G.

## Contents

| Path | What it holds |
|---|---|
| `src/` | The Gymnasium environment (CARLA client, sensors, route, reward, episode rules, route pool), the TD3/AHE agent, the networks, the replay buffer, the classical controller. |
| `scripts/train_td3.py` | Training. |
| `scripts/measure_route_progress_v5.py` | Evaluation of a checkpoint or of the classical controller, with per-step telemetry. |
| `config/` | Agent, environment, training and controller configurations; evaluation configurations; routes with checksums. |
| `analysis/g10_analyse.py` | The bias measure: Q1 − G per evaluation state, from one evaluation telemetry file. |
| `analysis/d2_analyse.py` | The statistics: Welch t, exact permutation test over all relabellings, exact Mann–Whitney test, intervals. |
| `analysis/reproduce_statistics.py` | Recomputes the paper's per-wave contrasts from `results/per_run.csv`. |
| `results/per_run.csv` | One row per training run: the bias and the driving measures the paper reports. |
| `Dockerfile` | The training and evaluation image: the CARLA 0.9.16 client, Python 3.10, PyTorch 2.4.1, Gymnasium 0.29.1. |

## Arms and configurations

Every arm uses `config/training_config_competence.yaml`.

| Arm | Agent configuration | Environment configuration | Steps |
|---|---|---|---|
| TD3, single route | `config/td3_config.yaml` | `config/carla_config_competence_s2c.yaml` | 300,000 |
| AHE, single route | `config/td3_config_ahe.yaml` | `config/carla_config_competence_s2c.yaml` | 300,000 |
| AHE-2H, single route | `config/td3_config_ahe2h.yaml` | `config/carla_config_competence_s2c.yaml` | 300,000 |
| TD3, multi-route | `config/td3_config.yaml` | `config/carla_config_v2.yaml` | 600,000 |
| AHE, multi-route | `config/td3_config_ahe.yaml` | `config/carla_config_v2.yaml` | 600,000 |
| TD3 + CAPS, multi-route | `config/td3_config_caps_preact.yaml` | `config/carla_config_v2.yaml` | 600,000 |
| AHE + CAPS, multi-route | `config/td3_config_ahe_caps_preact.yaml` | `config/carla_config_v2.yaml` | 600,000 |
| AHE+CDQ, multi-route | `config/td3_config_ahe_cdq.yaml` | `config/carla_config_v2.yaml` | 600,000 |

- **AHE against TD3:** `networks.critic.n_critics` 1 instead of 2, `algorithm.policy_freq` 1 instead of 2,
  `algorithm.noise_clip` 0.0 instead of 0.5. Nothing else differs.
- **AHE+CDQ against AHE:** `networks.critic.n_critics` 2 instead of 1, so two heads are fitted and the target takes
  their minimum. Nothing else differs.
- **AHE-2H against AHE:** both heads are fitted to the single head's target (`networks.critic.target_min: false`).
- **CAPS:** five keys added to the agent configuration (`algorithm.caps.*` and `algorithm.pre_activation_weight`).
- **The classical controller** (PID speed control, Pure Pursuit steering): `config/baseline_config_c08.yaml`.

## Results and statistics

`results/per_run.csv` has one row per run: `wave`, `arm`, `seed`, then the measures, empty where the paper reports
none for that wave.

| Column | Measure |
|---|---|
| `bias_q1_npc20` | Critic bias Q1 − G on the training route at 20 NPC vehicles, the run's mean over its evaluated checkpoints |
| `completion_npc20`, `completion_npc50` | Share of evaluation episodes that reach the goal, training route, 20 and 50 NPC vehicles |
| `heldout_rc_npc20`, `town02_rc_npc20` | Route completion on the three held-out Town01 routes and the three Town02 routes, 20 NPC vehicles |

A *wave* is a set of runs trained together, five per arm: `earlier` (an earlier configuration, one route, 600,000
steps), `ablation` (the one-at-a-time ablation in that configuration, 300,000 steps), `single`, `single_ahe2h`,
`multi`, `multi_caps` and `multi_cdq`. Seed labels do not make runs reproducible: two runs with the same code and
seed diverge early, so every run is an independent draw and the arms are compared unpaired.

```bash
pip install numpy scipy
python3 analysis/reproduce_statistics.py
```

prints, for every contrast in the paper, the difference of the arm means, its Welch 95% interval, the exact
permutation and Mann–Whitney p-values and Cohen's d.

## Running the system

See the commands below; they are those of the training and evaluation scripts used for the paper, with the parts
specific to our machines removed. Run them from the repository root. One CARLA server serves one process.

```bash
docker build -t td3-carla:py310 .
docker run -d --name carla-run1 --runtime=nvidia --net=host \
  --env=NVIDIA_VISIBLE_DEVICES=all --env=NVIDIA_DRIVER_CAPABILITIES=all \
  carlasim/carla:0.9.16 bash CarlaUE4.sh -RenderOffScreen -nosound -quality-level=Epic -carla-rpc-port=2000
docker run --rm --network host --runtime nvidia \
  -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=all -e PYTHONUNBUFFERED=1 -e PYTHONPATH=/workspace \
  -v "$(pwd)":/workspace -w /workspace td3-carla:py310 \
  python3 scripts/train_td3.py --scenario 0 --max-timesteps 600000 --eval-freq 999999 --checkpoint-freq 25000 \
    --seed 42 --log-level WARNING --device cuda --carla-port 2000 --tm-port 8000 \
    --agent-config config/td3_config_ahe_cdq.yaml --carla-config config/carla_config_v2.yaml \
    --training-config config/training_config_competence.yaml --env-isolation
```

A checkpoint is evaluated with `scripts/measure_route_progress_v5.py` (10 deterministic episodes per cell, a fresh
CARLA server per cell; `CARLA_SCENARIO_INDEX` 0 is 20 NPC vehicles and 1 is 50; the multi-route cells used
`--max-steps 3000`, the single-route cells the default 1000), and its telemetry file gives the bias with
`python3 analysis/g10_analyse.py <telemetry.csv>`.

## Notes

- **Revisions.** These files are the revision that trained the AHE+CDQ wave. The CAPS wave was trained on the
  same code without the AHE+CDQ configuration, the multi-route waves without CAPS on the revision before it, and the
  single-route runs on an earlier one; the configurations each of them used are unchanged here. The earlier
  configuration and its ablation (the `earlier` and `ablation` waves) were trained on an earlier revision that is not
  included.
- **The single-route environment** sets `route.use_dynamic_generation: true`; the image built from this `Dockerfile`
  lacks CARLA's `agents` package, so the environment reads the route from `config/waypoints.txt`, as in the reported
  runs.
- **Edited for release.** Comments were removed and each module keeps the first line of its docstring; identifying
  fields were removed from the configuration files. Every Python file has the same syntax tree as the file that ran,
  docstrings aside, and every YAML file parses to the same content, those fields aside.
- **Not included.** Trained checkpoints, the per-step evaluation telemetry and the pre-registration documents; they
  are available from the corresponding author on request.
