from dataclasses import dataclass
from datetime import datetime
import itertools
import json
import matplotlib.pyplot as plt
from multiprocessing import Pool
import numpy as np
from typing import List, Tuple, Dict, Any
import os


@dataclass
class City:
    id: int
    x: float
    y: float


class TSPGeneticAlgorithm:
    def __init__(
        self,
        cities: List[City],
        population_size: int = 200,
        elite_size: int = 40,
        mutation_rate: float = 0.02,
        tournament_size: int = 5,
    ):
        self.cities = cities
        self.population_size = population_size
        self.elite_size = elite_size
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.distance_matrix = self._create_distance_matrix()

    def _create_distance_matrix(self) -> np.ndarray:
        """Create matrix of distances between all cities."""
        n = len(self.cities)
        matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.sqrt(
                    (self.cities[i].x - self.cities[j].x) ** 2
                    + (self.cities[i].y - self.cities[j].y) ** 2
                )
                matrix[i][j] = dist
                matrix[j][i] = dist
        return matrix

    def _calculate_route_distance(self, route: List[int]) -> float:
        """Calculate total distance of a route."""
        total_distance = 0
        for i in range(len(route)):
            from_city = route[i]
            to_city = route[(i + 1) % len(route)]
            total_distance += self.distance_matrix[from_city][to_city]
        return total_distance

    def _create_initial_population(self) -> List[List[int]]:
        """Create initial random population."""
        population = []
        for _ in range(self.population_size):
            route = list(range(len(self.cities)))
            np.random.shuffle(route)
            population.append(route)
        return population

    def _tournament_selection(
        self, population: List[List[int]], fitness: List[float]
    ) -> List[int]:
        """Select parent using tournament selection."""
        tournament = np.random.choice(
            len(population), size=self.tournament_size, replace=False
        )
        tournament_fitness = [fitness[i] for i in tournament]
        return population[tournament[np.argmin(tournament_fitness)]]

    def _ordered_crossover(self, parent1: List[int], parent2: List[int]) -> List[int]:
        """Perform ordered crossover between two parents."""
        size = len(parent1)
        start, end = sorted(np.random.choice(size, 2, replace=False))
        offspring = [-1] * size
        offspring[start:end] = parent1[start:end]
        remaining_cities = [
            city for city in parent2 if city not in offspring[start:end]
        ]

        for i in range(start):
            offspring[i] = remaining_cities.pop(0)

        for i in range(end, size):
            offspring[i] = remaining_cities.pop(0)

        return offspring

    def _mutate(self, route: List[int]) -> List[int]:
        """Perform swap mutation on route."""
        for i in range(len(route)):
            if np.random.random() < self.mutation_rate:
                j = np.random.randint(0, len(route))
                route[i], route[j] = route[j], route[i]
        return route

    def evolve(
        self, generations: int
    ) -> Tuple[List[float], List[float], List[List[int]]]:
        """
        Evolve population for specified number of generations.
        Returns history of best and average fitness, and best routes.
        """
        population = self._create_initial_population()
        best_fitness_history = []
        avg_fitness_history = []
        best_routes_history = []

        for gen in range(generations):
            fitness = [self._calculate_route_distance(route) for route in population]

            best_idx = np.argmin(fitness)
            best_fitness_history.append(fitness[best_idx])
            avg_fitness_history.append(np.mean(fitness))
            best_routes_history.append(population[best_idx])

            new_population = []

            elite_idx = np.argsort(fitness)[: self.elite_size]
            new_population.extend([population[i] for i in elite_idx])

            while len(new_population) < self.population_size:
                parent1 = self._tournament_selection(population, fitness)
                parent2 = self._tournament_selection(population, fitness)
                offspring = self._ordered_crossover(parent1, parent2)
                offspring = self._mutate(offspring)
                new_population.append(offspring)

            population = new_population

        return best_fitness_history, avg_fitness_history, best_routes_history


def read_tsp_file(filename: str) -> List[City]:
    """Read TSP file and return list of cities."""
    cities = []
    with open(filename, "r") as f:
        lines = f.readlines()

    start_idx = lines.index("NODE_COORD_SECTION\n") + 1

    for line in lines[start_idx:]:
        if line.strip() == "EOF":
            break
        id_, x, y = map(float, line.strip().split())
        cities.append(City(int(id_) - 1, x, y))  # Convert to 0-based indexing

    return cities


def plot_route(cities: List[City], route: List[int], title: str = ""):
    """Plot a route on a 2D plane."""
    plt.figure(figsize=(10, 10))

    x = [city.x for city in cities]
    y = [city.y for city in cities]
    plt.scatter(x, y, c="red", s=50)

    for i in range(len(route)):
        start = cities[route[i]]
        end = cities[route[(i + 1) % len(route)]]
        plt.plot([start.x, end.x], [start.y, end.y], "b-", alpha=0.5)

    plt.title(title)
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid(True)
    plt.show()


def plot_convergence(best_fitness: List[float], avg_fitness: List[float]):
    """Plot convergence of best and average fitness."""
    plt.figure(figsize=(10, 6))
    generations = range(len(best_fitness))
    plt.plot(generations, best_fitness, "b-", label="Best Fitness")
    plt.plot(generations, avg_fitness, "r-", label="Average Fitness")
    plt.title("Convergence Plot")
    plt.xlabel("Generation")
    plt.ylabel("Route Distance")
    plt.legend()
    plt.grid(True)
    plt.show()

@dataclass
class ExperimentResult:
    params: Dict[str, Any]
    best_fitness: float
    avg_fitness: float
    convergence_history: List[float]
    runtime: float
    route: List[int]


def run_experiment(config: Dict[str, Any]) -> ExperimentResult:
    """
    Run a single experiment with given parameters.
    Returns ExperimentResult with metrics.
    """
    start_time = datetime.now()

    ga = TSPGeneticAlgorithm(
        cities=config["cities"],
        population_size=config["population_size"],
        elite_size=config["elite_size"],
        mutation_rate=config["mutation_rate"],
        tournament_size=config["tournament_size"],
    )

    best_fitness, avg_fitness, best_routes = ga.evolve(
        generations=config["generations"]
    )

    runtime = (datetime.now() - start_time).total_seconds()

    return ExperimentResult(
        params={k: v for k, v in config.items() if k != "cities"},
        best_fitness=best_fitness[-1],
        avg_fitness=avg_fitness[-1],
        convergence_history=best_fitness,
        runtime=runtime,
        route=best_routes[-1],
    )


class ParameterSearch:
    def __init__(
        self,
        cities: List[City],
        param_grid: Dict[str, List[Any]],
        generations: int = 1000,
    ):
        """
        Initialize parameter search with parameter grid.

        param_grid format:
        {
            'population_size': [50, 100, 200],
            'elite_size': [10, 20, 40],
            'mutation_rate': [0.01, 0.05, 0.1],
            'tournament_size': [3, 5, 7]
        }
        """
        self.cities = cities
        self.param_grid = param_grid
        self.generations = generations

    def _generate_configs(self) -> List[Dict[str, Any]]:
        """Generate all possible parameter combinations."""
        keys = self.param_grid.keys()
        values = self.param_grid.values()
        configs = []

        for combination in itertools.product(*values):
            config = dict(zip(keys, combination))
            config["cities"] = self.cities
            config["generations"] = self.generations
            configs.append(config)

        return configs

    def search(self, n_workers: int = -1) -> List[ExperimentResult]:
        """
        Run parameter search using process pool.
        n_workers: number of parallel processes (-1 for CPU count)
        """
        configs = self._generate_configs()

        if n_workers == -1:
            n_workers = os.cpu_count()

        print(f"Running {len(configs)} configurations using {n_workers} workers...")

        with Pool(n_workers) as pool:
            results = pool.map(run_experiment, configs)

        return results

    @staticmethod
    def save_results(results: List[ExperimentResult], filename: str):
        """Save results to JSON file."""
        serializable_results = []
        for result in results:
            result_dict = {
                "params": result.params,
                "best_fitness": result.best_fitness,
                "avg_fitness": result.avg_fitness,
                # 'convergence_history': result.convergence_history,
                "runtime": result.runtime,
                "route": result.route,
            }
            serializable_results.append(result_dict)

        with open(filename, "w") as f:
            json.dump(serializable_results, f, indent=2)

    @staticmethod
    def load_results(filename: str) -> List[ExperimentResult]:
        """Load results from JSON file."""
        with open(filename, "r") as f:
            data = json.load(f)

        results = []
        for result_dict in data:
            result = ExperimentResult(
                params=result_dict["params"],
                best_fitness=result_dict["best_fitness"],
                avg_fitness=result_dict["avg_fitness"],
                convergence_history=result_dict["convergence_history"],
                runtime=result_dict["runtime"],
            )
            results.append(result)

        return results


def analyze_results(results: List[ExperimentResult]) -> None:
    """Analyze and visualize parameter search results."""
    sorted_results = sorted(results, key=lambda x: x.best_fitness)

    print("\nTop 10 Parameter Configurations:")
    for i, result in enumerate(sorted_results[:10], 1):
        print(f"\n{i}. Best Fitness: {result.best_fitness:.2f}")
        print(f"   Average Fitness: {result.avg_fitness:.2f}")
        print(f"   Runtime: {result.runtime:.2f} seconds")
        print(f"   Route: {result.route}")
        print("   Parameters:")
        for param, value in result.params.items():
            print(f"      {param}: {value}")

    # print("\n" + "=" * 50)
    # print("BEST CONFIGURATION FOUND:")
    # print(f"Best Fitness: {sorted_results[0].best_fitness:.2f}")
    # print(f"Average Fitness: {sorted_results[0].avg_fitness:.2f}")
    # print(f"Runtime: {sorted_results[0].runtime:.2f} seconds")
    # print(f"   Route: {result.route:.2f}")
    # print("Parameters:")
    # for param, value in sorted_results[0].params.items():
    #     if param == "route":
    #         print(f"   Best Route: {' -> '.join(map(str, value))}")
    #     else:
    #         print(f"   {param}: {value}")
    # print("=" * 50 + "\n")

    plt.figure(figsize=(15, 10))

    # Plot convergence histories of top 5 configurations
    for i, result in enumerate(sorted_results[:10]):
        plt.plot(
            result.convergence_history,
            label=f"Config {i+1} (Best: {result.best_fitness:.2f})",
        )

    plt.title("Convergence History of Top 5 Configurations")
    plt.xlabel("Generation")
    plt.ylabel("Best Fitness")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    # cities = read_tsp_file("ch130.tsp")

    # ga = TSPGeneticAlgorithm(
    #     cities=cities,
    #     population_size=100,
    #     elite_size=20,
    #     mutation_rate=0.01,
    #     tournament_size=5,
    # )

    # best_fitness, avg_fitness, best_routes = ga.evolve(generations=1000)
    # print(np.min(best_fitness))

    # plot_route(cities, best_routes[-1], "Best Route Found")
    # plot_convergence(best_fitness, avg_fitness)

    param_grid = {
        "population_size": [50, 100, 200],
        "elite_size": [10, 20, 40],
        "mutation_rate": [0.01, 0.05, 0.1],
        "tournament_size": [3, 5, 7],
    }

    cities = read_tsp_file("ch130.tsp")
    search = ParameterSearch(cities, param_grid, generations=5000)
    results = search.search()
    search.save_results(results, "parameter_search_results.json")

    analyze_results(results)
