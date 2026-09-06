# Heat Equation Simulation: CPU vs GPU (Julia)

A 2D heat diffusion simulation implemented in Julia, comparing sequential CPU performance against parallel GPU execution using CUDA.jl .

## Overview

This project solves the 2D heat equation on a 1024x1024 grid using a finite-difference method, simulating heat diffusing outward from a heated block in the center of the grid. The simulation is implemented twice — once running sequentially on the CPU, and once running in parallel on the GPU — to benchmark and visualize the performance difference between the two approaches.

## What it does

- Initializes a 1024x1024 grid with a 100°C heated block in the center
- Steps the simulation forward 1000 iterations using a 2D finite-difference stencil
- Times 100 iterations on the CPU (sequential, nested loops) and on the GPU (parallel kernel)
- Generates a heatmap of the final temperature distribution
- Generates a bar chart comparing CPU vs GPU execution time, including the computed speedup factor

## Requirements

- Julia 1.9+
- An NVIDIA GPU (for `CUDA.jl`) or AMD GPU with ROCm support (for `AMDGPU.jl`)
- Julia packages: `CUDA` (or `AMDGPU`), `Plots`, `BenchmarkTools`

Install dependencies:
```julia
import Pkg
Pkg.add(["CUDA", "Plots", "BenchmarkTools"])
```

## Running

```
julia heat_equation.jl
```

Note: GPU execution requires compatible hardware. This was developed and tested with CUDA on an NVIDIA GPU (via Google Colab's T4 runtime); an AMDGPU.jl variant was also written for AMD hardware, though ROCm support on Windows can be inconsistent depending on GPU model and driver setup.

## Output

- A heatmap image showing the final temperature distribution across the grid
- A bar chart comparing CPU and GPU execution time (log scale), with the calculated speedup

## Background

The heat equation models how temperature diffuses through a medium over time. Each grid cell's next temperature value depends on the temperatures of its neighboring cells, making this an ideal candidate for GPU parallelization — every cell's update is independent of the others within a single timestep, allowing thousands of cells to be computed simultaneously rather than one at a time.
