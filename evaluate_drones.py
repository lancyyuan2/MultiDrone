from multi_drone import MultiDrone
from centralised_prm_star import CentralisedPRMStarPlanner
import time
import numpy as np

env_files = [
    ("num_drones_2.yaml", 2),
    ("num_drones_4.yaml", 4),
    ("num_drones_6.yaml", 6),
    ("num_drones_8.yaml", 8),
    ("num_drones_10.yaml", 10),
]

trials_per_env = 10

for env_file, num_drones in env_files:
    times = []
    print(f"\n Testing environment with {num_drones} drones...")
    for i in range(trials_per_env):
        try:
            sim = MultiDrone(num_drones=num_drones, environment_file=env_file)
            planner = CentralisedPRMStarPlanner(sim)
            start = time.time()
            path = planner.plan(timeout=120)
            end = time.time()
            if path:
                duration = end - start
                times.append(duration)
                print(f" Trial {i+1}: Path found in {duration:.2f}s")
                if i == 0:  # only show the first successful trial
                    sim.visualize_paths(path)# generate the path
            else:
                print(f" Trial {i+1}: No path found")
        except Exception as e:
            print(f" Trial {i+1}: Exception - {e}")

    if times:
        mean_time = np.mean(times)
        ci = 1.96 * np.std(times) / np.sqrt(len(times))
        print(f"\n {num_drones} Drones Summary:")
        print(f"  Success: {len(times)}/{trials_per_env}")
        print(f"  Avg. Time: {mean_time:.2f}s")
        print(f"  95% CI: ±{ci:.2f}s")
    else:
        print(f"\n {num_drones} Drones Summary: No successful runs")
