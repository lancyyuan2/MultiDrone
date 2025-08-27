import argparse
import yaml
import numpy as np
from multi_drone import MultiDrone
from centralised_prm_star import CentralisedPRMStarPlanner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", required=True, help="Path to environment YAML")
    parser.add_argument("--timeout", type=int, default=120, help="Planning timeout in seconds")
    parser.add_argument("--samples", type=int, default=5000, help="Number of samples for PRM*")
    parser.add_argument("--radius", type=float, default=120.0, help="Connection radius for PRM*")
    args = parser.parse_args()

    # 自动读取无人机数量
    with open(args.env, "r") as f:
        config = yaml.safe_load(f)
    num_drones = len(config["initial_configuration"])

    print("=====================================")
    print(f"Using environment: {args.env}")
    print(f"Number of drones: {num_drones}")
    print(f"Timeout: {args.timeout}s, Samples: {args.samples}, Radius: {args.radius}")
    print("=====================================")

    # 初始化仿真环境
    sim = MultiDrone(num_drones=num_drones, environment_file=args.env)

    # 初始化规划器
    planner = CentralisedPRMStarPlanner(
        sim,
        num_samples=args.samples,
        connection_radius=args.radius
    )

    # 执行规划
    solution_path = planner.plan(timeout=args.timeout)

    if solution_path:
        print(f"[SUCCESS] Path found with {len(solution_path)} steps.")
        sim.visualize_paths(solution_path)
    else:
        print("[FAIL] No path found within timeout.")


if __name__ == "__main__":
    main()
