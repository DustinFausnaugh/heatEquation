ENV["GKSwstype"] = "100"
using CUDA
using Plots
using BenchmarkTools
import Plots: bar

# Define grid size
Nx = 1024
Ny = 1024

# Allocate matrices on the GPU
U = CUDA.zeros(Float32, Nx, Ny)
U_next = CUDA.zeros(Float32, Nx, Ny)

# Set initial temperature block (100°C in the center)
U[400:600, 400:600] .= 100.0f0

# Define the GPU kernel function
function heatKernel!(U, U_next, dx, dy, dt, alpha)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    j = threadIdx().y + (blockIdx().y - 1) * blockDim().y

    if i > 1 && i < size(U,1) && j > 1 && j < size(U,2)
        grad_x = (U[i+1, j] - 2 * U[i,j] + U[i-1, j]) / (dx^2)
        grad_y = (U[i, j+1] - 2 * U[i,j] + U[i, j-1]) / (dy^2)

        U_next[i,j] = U[i,j] + dt * alpha * (grad_x + grad_y)
    end
    return nothing
end

# CPU version for comparison
function cpu_heat_step!(U, U_next, dx, dy, dt, alpha)
    Nx, Ny = size(U)
    for j in 2:(Ny-1)
        for i in 2:(Nx-1)
            grad_x = (U[i+1, j] - 2 * U[i, j] + U[i-1, j]) / (dx^2)
            grad_y = (U[i, j+1] - 2 * U[i, j] + U[i, j-1]) / (dy^2)
            U_next[i, j] = U[i, j] + dt * alpha * (grad_x + grad_y)
        end
    end
end

# Define physical constants and thread structure
dx = 0.01f0
dy = 0.01f0
alpha = 0.001f0
dt = 0.02f0

threads_per_block = (16, 16)
blocks_per_grid = (64, 64)

# The main time stepping loop
println("Running simulation...")
for step in 1:1000
    @cuda threads=threads_per_block blocks=blocks_per_grid heatKernel!(U, U_next, dx, dy, dt, alpha)
    global U, U_next = U_next, U
end
println("Simulation complete!")

# Bring data back to CPU and save the heatmap
U_cpu = Array(U)
gr()
p1 = heatmap(U_cpu, c=:thermal, title="Final Temperature Distribution")
savefig(p1, "heatmap.png")
println("Saved heatmap.png")

# Create fresh test matrices for benchmarking
U_cpu = zeros(Float32, Nx, Ny)
U_cpu_next = zeros(Float32, Nx, Ny)
U_gpu = CUDA.zeros(Float32, Nx, Ny)
U_gpu_next = CUDA.zeros(Float32, Nx, Ny)

println("Warming up and compiling kernels...")
cpu_heat_step!(U_cpu, U_cpu_next, dx, dy, dt, alpha)
@cuda threads=threads_per_block blocks=blocks_per_grid heatKernel!(U_gpu, U_gpu_next, dx, dy, dt, alpha)
CUDA.synchronize()

println("Timing CPU (this might take a few seconds)...")
cpu_time = @belapsed for step in 1:100
    cpu_heat_step!($U_cpu, $U_cpu_next, $dx, $dy, $dt, $alpha)
    $U_cpu, $U_cpu_next = $U_cpu_next, $U_cpu
end

println("Timing GPU...")
gpu_time = @belapsed begin
    for step in 1:100
        @cuda threads=$threads_per_block blocks=$blocks_per_grid heatKernel!($U_gpu, $U_gpu_next, $dx, $dy, $dt, $alpha)
        $U_gpu, $U_gpu_next = $U_gpu_next, $U_gpu
    end
    CUDA.synchronize()
end

# Plot the performance comparison graph
gr()
categories = ["CPU (Sequential)", "GPU (Parallel)"]
times = [cpu_time, gpu_time]
speedup = round(cpu_time / gpu_time, digits=1)

p2 = bar(categories, times,
    ylabel="Time for 100 steps (seconds)",
    title="Execution Speed: CPU vs. GPU ($speedup" * "x Speedup)",
    legend=false,
    color=[:red, :blue],
    yscale=:log10
)
savefig(p2, "speed_comparison.png")
println("Saved speed_comparison.png")
println("Speedup: $(speedup)x")