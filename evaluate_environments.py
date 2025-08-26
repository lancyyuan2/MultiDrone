import numpy as np
import time
from multi_drone import MultiDrone
from centralised_prm_star import CentralisedPRMStarPlanner
import os
import scipy.stats as st

# ====== Settings ======
ENV_FILES = [
    "env_empty.yaml",
    "env_simple.yaml",
    "env_medium.yaml",
    "env_cluttered.yaml",
    "env_dense.yaml"
]
NUM_TRIALS = 10
TIMEOUT = 120  # seconds

# ====== Helper: Confidence Interval ======
def confidence_interval(data, confidence=0.95):
    arr = np.array(data)
    mean = np.mean(arr)
    sem = st.sem(arr)
    interval = sem * st.t.ppf((1 + confidence) / 2., len(arr)-1) if len(arr) > 1 else 0
    return mean, interval

# ====== Evaluation Loop ======
for env in ENV_FILES:
    print(f"\n Testing environment: {env}")
    success_times = []
    failures = 0

    for i in range(NUM_TRIALS):
        try:
            sim = MultiDrone(num_drones=2, environment_file=env)
            planner = CentralisedPRMStarPlanner(sim)
            start = time.time()
            path = planner.plan(timeout=TIMEOUT)
            duration = time.time() - start

            if path:
                success_times.append(duration)
                print(f" Trial {i+1}: Path found in {duration:.2f}s")
            else:
                failures += 1
                print(f" Trial {i+1}: No path found")
        except Exception as e:
            failures += 1
            print(f" Trial {i+1}: Exception - {str(e)}")

    # Summary
    print("\n Summary:")
    if success_times:
        mean_time, ci = confidence_interval(success_times)
        print(f"  Success rate: {len(success_times)}/{NUM_TRIALS}")
        print(f"  Mean time: {mean_time:.2f}s ± {ci:.2f}s (95% CI)")
    else:
        print("   No successful runs")
