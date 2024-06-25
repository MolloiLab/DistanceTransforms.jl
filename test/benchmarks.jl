using DistanceTransforms

using KernelAbstractions
using CUDA, AMDGPU, Metal
using oneAPI
using Random

using BenchmarkTools
using CSV
using DataFrames

s = 10 # number of samples
e = 1 # number of evals

# Create an array of input sizes to benchmark
sizes = [10, 50, 100]

# Create a DataFrame to store the benchmark results
df = DataFrame(size = Int[], dt = Float64[])

# Benchmark for each input size
for n in sizes
    f = Float32.(rand([0, 1], n, n))

    # Proposed-CUDA (DistanceTransforms.jl)
    if CUDA.functional()
        f_cuda = CuArray(f)
        dt = @benchmark(transform($boolean_indicator($f_cuda)), samples=s, evals=e)
        
        # Append the benchmark result to the DataFrame
        push!(df, (n, minimum(dt).time))
    end
end

# Create a directory to store the benchmark results
mkpath("benchmark_results")

# Save the DataFrame to a CSV file
CSV.write("benchmark_results/cuda_benchmarks.csv", df)