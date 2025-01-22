using DistanceTransforms
using Suppressor
using BenchmarkTools
using Random
# using JSON

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

filename = BENCHMARK_GROUP == "CPU" ? 
    string("CPUbenchmarks", BENCHMARK_CPU_THREADS, "threads.json") : 
    string(BENCHMARK_GROUP, "benchmarks.json")

BenchmarkTools.save(joinpath(filepath, filename), median(results))
@info "Saved results to $(joinpath(filepath, filename))"