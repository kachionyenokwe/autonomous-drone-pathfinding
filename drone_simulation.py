import heapq
import time

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


# Environment configuration
GRID_SIZE = 20
START = (0, 0)
GOAL = (19, 19)
STATIC_OBSTACLE_DENSITY = 0.20
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)


def create_grid():
    """Create a reproducible grid containing randomly placed obstacles."""
    new_grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)

    for row in range(GRID_SIZE):
        for column in range(GRID_SIZE):
            position = (row, column)

            if position not in (START, GOAL):
                if np.random.rand() < STATIC_OBSTACLE_DENSITY:
                    new_grid[row, column] = 1

    return new_grid


def heuristic(position, goal):
    """Calculate Manhattan distance between two grid positions."""
    return abs(position[0] - goal[0]) + abs(position[1] - goal[1])


def astar(grid_map, start, goal):
    """Find the shortest available path using the A* algorithm."""
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    open_set = []
    heapq.heappush(
        open_set,
        (heuristic(start, goal), 0, start, [start]),
    )

    best_cost = {start: 0}

    while open_set:
        _, current_cost, current, path = heapq.heappop(open_set)

        if current == goal:
            return path

        if current_cost > best_cost.get(current, float("inf")):
            continue

        for row_change, column_change in directions:
            neighbor = (
                current[0] + row_change,
                current[1] + column_change,
            )

            row, column = neighbor

            if not (0 <= row < GRID_SIZE and 0 <= column < GRID_SIZE):
                continue

            if grid_map[row, column] != 0:
                continue

            new_cost = current_cost + 1

            if new_cost < best_cost.get(neighbor, float("inf")):
                best_cost[neighbor] = new_cost
                priority = new_cost + heuristic(neighbor, goal)

                heapq.heappush(
                    open_set,
                    (priority, new_cost, neighbor, path + [neighbor]),
                )

    return None


class DynamicObstacle:
    """A moving obstacle that reverses direction at grid boundaries."""

    def __init__(self, start_position, direction):
        self.position = list(start_position)
        self.direction = direction

    def move(self):
        new_row = self.position[0] + self.direction[0]
        new_column = self.position[1] + self.direction[1]

        if not (
            0 <= new_row < GRID_SIZE
            and 0 <= new_column < GRID_SIZE
        ):
            self.direction = (
                -self.direction[0],
                -self.direction[1],
            )

            new_row = self.position[0] + self.direction[0]
            new_column = self.position[1] + self.direction[1]

        self.position = [new_row, new_column]
        return tuple(self.position)


class DroneSimulation:
    """Control the drone, obstacles, rerouting, and performance metrics."""

    def __init__(self):
        self.grid = create_grid()
        self.drone_position = START
        self.goal = GOAL
        self.path = astar(self.grid, self.drone_position, self.goal)

        if self.path is None:
            raise RuntimeError(
                "The generated grid has no valid initial path. "
                "Try another random seed."
            )

        self.dynamic_obstacles = [
            DynamicObstacle((10, 5), (0, 1)),
            DynamicObstacle((5, 12), (1, 0)),
        ]

        self.step_count = 0
        self.reroute_count = 0
        self.pause_count = 0
        self.start_time = time.perf_counter()
        self.trajectory = [START]
        self.completed = False
        self.total_time = 0.0

    def update_frame(self):
        if self.completed:
            return

        self.step_count += 1

        # Move all dynamic obstacles.
        for obstacle in self.dynamic_obstacles:
            obstacle.move()

        if len(self.path) <= 1:
            self.complete_simulation()
            return

        next_step = self.path[1]

        # Simulate an unexpected moving obstacle entering the
        # drone's planned route at step 8.
        if self.step_count == 8:
            interceptor = self.dynamic_obstacles[0]
            interceptor.position = list(next_step)
            interceptor.direction = (0, 1)

            print(
                f"[Step {self.step_count}] A moving obstacle "
                f"entered the planned route at {next_step}."
            )

        # Build the sensor map after all obstacles have moved.
        temporary_grid = self.grid.copy()
        dynamic_positions = []

        for obstacle in self.dynamic_obstacles:
            position = tuple(obstacle.position)
            dynamic_positions.append(position)

            if position not in (self.drone_position, self.goal):
                temporary_grid[position[0], position[1]] = 2

        # Virtual sensor and autonomous decision logic.
        if next_step in dynamic_positions:
            self.pause_count += 1

            print(
                f"[Step {self.step_count}] Collision threat at "
                f"{next_step}. Action: PAUSE AND REROUTE."
            )

            new_path = astar(
                temporary_grid,
                self.drone_position,
                self.goal,
            )

            if new_path and len(new_path) > 1:
                self.path = new_path
                self.reroute_count += 1
                next_step = self.path[1]

                print(
                    f"New route calculated. "
                    f"Next safe position: {next_step}."
                )
            else:
                print(
                    "All paths are temporarily blocked. "
                    "Drone is holding position."
                )
                return

        # Move the drone along the current safe path.
        self.drone_position = next_step
        self.trajectory.append(self.drone_position)
        self.path = self.path[1:]

        if self.drone_position == self.goal:
            self.complete_simulation()

    def complete_simulation(self):
        self.completed = True
        self.total_time = time.perf_counter() - self.start_time
        self.print_metrics()

    def print_metrics(self):
        optimal_distance = heuristic(START, GOAL)
        actual_distance = len(self.trajectory) - 1
        efficiency = (
            optimal_distance / max(1, actual_distance)
        ) * 100

        print("\n" + "=" * 50)
        print("SIMULATION PERFORMANCE METRICS")
        print("=" * 50)
        print(f"Total steps             : {self.step_count}")
        print(f"Reroute count           : {self.reroute_count}")
        print(f"Hazard pauses           : {self.pause_count}")
        print(f"Optimal grid distance   : {optimal_distance} steps")
        print(f"Actual trajectory length: {actual_distance} steps")
        print(f"Path efficiency         : {efficiency:.2f}%")
        print(f"Execution time          : {self.total_time:.4f} seconds")
        print("=" * 50)


simulation = DroneSimulation()

figure, axis = plt.subplots(figsize=(9, 8))
color_map = ListedColormap(["#FFFFFF", "#333333", "#FF0000"])


def animate(_frame):
    if not simulation.completed:
        simulation.update_frame()

    axis.clear()

    display_grid = simulation.grid.copy()

    for obstacle in simulation.dynamic_obstacles:
        row, column = obstacle.position

        if (row, column) not in (
            simulation.drone_position,
            GOAL,
        ):
            display_grid[row, column] = 2

    axis.imshow(
        display_grid,
        cmap=color_map,
        origin="upper",
        vmin=0,
        vmax=2,
    )

    if simulation.path:
        path_rows, path_columns = zip(*simulation.path)
        axis.plot(
            path_columns,
            path_rows,
            color="#0066FF",
            linestyle="--",
            linewidth=2,
            label="Planned route",
        )

    trajectory_rows, trajectory_columns = zip(
        *simulation.trajectory
    )
    axis.plot(
        trajectory_columns,
        trajectory_rows,
        color="#00AA44",
        linewidth=2.5,
        label="Flight path",
    )

    axis.scatter(
        START[1],
        START[0],
        color="blue",
        s=120,
        marker="s",
        label="Start",
    )
    axis.scatter(
        GOAL[1],
        GOAL[0],
        color="gold",
        edgecolors="black",
        s=180,
        marker="*",
        label="Goal",
    )
    axis.scatter(
        simulation.drone_position[1],
        simulation.drone_position[0],
        color="cyan",
        edgecolors="black",
        s=150,
        marker="o",
        label="Drone",
    )

    axis.set_title(
        f"Step: {simulation.step_count} | "
        f"Reroutes: {simulation.reroute_count} | "
        f"Position: {simulation.drone_position}"
    )
    axis.set_xticks(range(GRID_SIZE))
    axis.set_yticks(range(GRID_SIZE))
    axis.grid(
        True,
        color="#CCCCCC",
        linestyle=":",
        linewidth=0.5,
    )
    axis.legend(loc="upper left", bbox_to_anchor=(1, 1))
    figure.tight_layout()


drone_animation = animation.FuncAnimation(
    figure,
    animate,
    frames=100,
    interval=300,
    repeat=False,
    cache_frame_data=False,
)

plt.show()