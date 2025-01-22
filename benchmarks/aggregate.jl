using BenchmarkTools
using JSON

const RESULTS = BenchmarkGroup()
const MEMORY_RESULTS = Dict{String, Dict{String, Float64}}()

# Load CPU benchmarks
for threads in [1, 4]
    filename = joinpath(@__DIR__, "results", string("CPUbenchmarks", threads, "threads.json"))
    if isfile(filename)
        RESULTS["CPU_$(threads)thread"] = BenchmarkTools.load(filename)[1]
    end
end

# Load GPU benchmarks and memory info
for backend in ["Metal", "CUDA", "AMDGPU", "oneAPI"]
    # Load timing benchmarks
    filename = joinpath(@__DIR__, "results", "$(backend)benchmarks.json")
    if isfile(filename)
        RESULTS[backend] = BenchmarkTools.load(filename)[1]
    end
    
    # Load memory info
    memory_filename = joinpath(@__DIR__, "results", "$(backend)_memory_info.json")
    if isfile(memory_filename)
        MEMORY_RESULTS[backend] = JSON.parsefile(memory_filename)
    end
end

# Save combined results
mkpath(joinpath(@__DIR__, "results"))
BenchmarkTools.save(joinpath(@__DIR__, "results", "combinedbenchmarks.json"), RESULTS)
open(joinpath(@__DIR__, "results", "combined_memory_info.json"), "w") do io
    JSON.print(io, MEMORY_RESULTS)
end