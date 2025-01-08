# Traveling Salesman Problem Solver using Genetic Algorithms

This repository contains a Python implementation of a genetic algorithm to solve the Traveling Salesman Problem (TSP). The implementation includes a comprehensive parameter optimization framework to find the best genetic algorithm configurations for specific TSP instances.

## Overview

The Traveling Salesman Problem (TSP) is a classic NP-hard optimization problem in computer science. Given a set of cities and the distances between them, the goal is to find the shortest possible route that visits each city exactly once and returns to the starting city.

This implementation uses genetic algorithms with the following features:
- Ordered crossover (OX) operator for maintaining valid tour sequences
- Tournament selection for parent choice
- Swap mutation for local search
- Elitism for preserving best solutions
- Comprehensive parameter tuning framework
- Performance visualization and analysis tools

## Requirements

- Python 3.7+
- NumPy
- Matplotlib
- Multiprocessing (standard library)
- Dataclasses (standard library)

## Installation

```bash
git clone https://github.com/denis-tsariov/evolutionaryTSP.git
cd evolutionaryTSP
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
from tsp_solver import TSPGeneticAlgorithm, read_tsp_file

# Load cities from TSP file
cities = read_tsp_file("ch130.tsp")

# Create GA instance
ga = TSPGeneticAlgorithm(
    cities=cities,
    population_size=100,
    elite_size=20,
    mutation_rate=0.01,
    tournament_size=5
)

# Run evolution
best_fitness, avg_fitness, best_routes = ga.evolve(generations=1000)

# Plot results
plot_route(cities, best_routes[-1], "Best Route Found")
plot_convergence(best_fitness, avg_fitness)
```

### Parameter Optimization

```python
from parameter_search import ParameterSearch

# Define parameter grid
param_grid = {
    "population_size": [50, 100, 200],
    "elite_size": [10, 20, 40],
    "mutation_rate": [0.01, 0.05, 0.1],
    "tournament_size": [3, 5, 7]
}

# Create and run parameter search
search = ParameterSearch(cities, param_grid, generations=5000)
results = search.search()

# Save and analyze results
search.save_results(results, "parameter_search_results.json")
analyze_results(results)
```

## Key Components

### TSPGeneticAlgorithm Class
- Main implementation of the genetic algorithm
- Handles evolution process, crossover, mutation, and selection
- Supports customizable parameters

### ParameterSearch Class
- Framework for testing different parameter configurations
- Parallel processing support for faster experimentation
- Results storage and analysis tools

### Visualization Tools
- Route plotting
- Convergence analysis
- Parameter impact visualization

## Best Found Parameters

Based on extensive testing with the ch130 dataset, the best performing configuration was:
- Population size: 100
- Elite size: 20
- Mutation rate: 0.01
- Tournament size: 3

This configuration achieved a tour length of 7,777.97 (compared to the known optimal of 6,110).

## Performance

- Runtime scales approximately linearly with population size
- Typical runtime for 10,000 generations: 120-290 seconds (depending on parameters)
- Best solutions typically found between 3,000-5,000 generations
- Consistent performance across multiple runs with the same parameters
