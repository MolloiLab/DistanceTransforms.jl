using DistanceTransforms
using Suppressor
using BenchmarkTools
using Random
using JSON

const SUITE = BenchmarkGroup()
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 5   # Reduce this from 120 seconds
BenchmarkTools.DEFAULT_PARAMETERS.gcsample = false

const BENCHMARK_GROUP = get(ENV, "BENCHMARK_GROUP", "CPU")
const BENCHMARK_CPU_THREADS = Threads.nthreads()

@info "Running $(BENCHMARK_GROUP) benchmarks with $(BENCHMARK_CPU_THREADS) thread(s)"

include("setup.jl")
setup_benchmarks(SUITE, BENCHMARK_GROUP, BENCHMARK_CPU_THREADS)
results = BenchmarkTools.run(SUITE; verbose=true)

filepath = joinpath(@__DIR__, "results")
mkpath(filepath)

# Save timing benchmarks
filename = BENCHMARK_GROUP == "CPU" ? 
    string("CPUbenchmarks", BENCHMARK_CPU_THREADS, "threads.json") : 
    string(BENCHMARK_GROUP, "benchmarks.json")

BenchmarkTools.save(joinpath(filepath, filename), median(results))
@info "Saved results to $(joinpath(filepath, filename))"

# Handle memory info
if BENCHMARK_GROUP != "CPU"
    memory_filename = "$(BENCHMARK_GROUP)_memory_info.json"
    memory_filepath = joinpath(filepath, memory_filename)
    
    # Print memory info to console for manual backup
    if isfile(memory_filepath)
        memory_data = JSON.parsefile(memory_filepath)
        println("\nMemory Info for $BENCHMARK_GROUP:")
        println(JSON.json(memory_data, 4))  # indent with 4 spaces
    end
    
    # Try to upload both files
    # run(`buildkite-agent artifact upload "benchmarks/results/$(filename)"`)
    run(`buildkite-agent artifact upload "benchmarks/results/$(memory_filename)"`)
end