import numpy as np
import networkx as nx
import time


class CentralisedPRMStarPlanner:
    def __init__(self, sim, num_samples=2000, connection_radius=30.0):
        self.sim = sim
        self.num_samples = num_samples
        self.connection_radius = connection_radius
        self.dim = sim.initial_configuration.shape[1]
        self.num_drones = sim.initial_configuration.shape[0]

        # Apply margin to bounding box
        bounds_array = np.array(sim._bounds, dtype=np.float32)
        margin = 1.5
        self.lower_bound = bounds_array[:, 0] + margin
        self.upper_bound = bounds_array[:, 1] - margin

    def sample_configuration(self):
        config = np.random.uniform(
            low=self.lower_bound[None, :],
            high=self.upper_bound[None, :],
            size=(self.num_drones, 3)
        ).astype(np.float32)
        return config

    def plan(self, timeout=120, seed=None, k_neighbors=20):
        if seed is not None:
            np.random.seed(seed)

        t0 = time.time()

        def time_left():
            return timeout - (time.time() - t0)

        start = self.sim.initial_configuration
        goal = self.sim.goal_positions
        # 1) 起终点合法性
        if not self.sim.is_valid(start) or not self.sim.is_valid(goal):
            print("Start/Goal invalid");
            return []

        # ---- Step 1: 采样 ----
        samples = [start]
        print("Sampling configurations...")
        while len(samples) < self.num_samples and time_left() > timeout * 0.5:
            cfg = self.sample_configuration()
            if self.sim.is_valid(cfg):
                samples.append(cfg)

        samples.append(goal)
        num_nodes = len(samples)
        print(f"Collected {num_nodes} nodes (including start/goal)")

        # ---- Step 2: 建图（k-NN + 联合L2距离）----
        print("Building PRM* graph...")
        import networkx as nx
        import numpy as np
        from scipy.spatial import cKDTree

        G = nx.Graph()
        for i in range(num_nodes):
            G.add_node(i, config=samples[i])

        X = np.vstack([s.ravel() for s in samples])  # (n, 3K)
        tree = cKDTree(X)

        # 可选：PRM* 半径
        d = X.shape[1]
        rn = (np.log(num_nodes) / num_nodes) ** (1.0 / max(d, 1))
        gamma = 50.0  # 经验系数，可调
        radius = max(self.connection_radius, gamma * rn)

        for i in range(num_nodes):
            if time_left() <= timeout * 0.4: break
            # 半径搜索 + k 上限
            idxs = tree.query_ball_point(X[i], r=radius)
            if len(idxs) > k_neighbors + 1:
                # 取最近 k 个（去掉自身）
                dists, neigh = tree.query(X[i], k=k_neighbors + 1)
                idxs = neigh.tolist()
            for j in idxs:
                if j <= i: continue
                ci, cj = samples[i], samples[j]
                dist = np.linalg.norm((ci - cj).ravel())
                if self.sim.motion_valid(ci, cj):
                    G.add_edge(i, j, weight=dist)

        # 超时兜底
        if time_left() <= 0:
            print("Timeout before search");
            return []

        # ---- Step 3: 搜索 ----
        print("Searching for shortest path...")
        try:
            path_idx = nx.shortest_path(G, source=0, target=num_nodes - 1, weight='weight')
            path = [samples[k] for k in path_idx]
            print(
                f"Path found with {len(path)} steps. time={time.time() - t0:.2f}s | |V|={G.number_of_nodes()} |E|={G.number_of_edges()}")
            return path
        except nx.NetworkXNoPath:
            print(f"No path found. time={time.time() - t0:.2f}s | |V|={G.number_of_nodes()} |E|={G.number_of_edges()}")
            return []
