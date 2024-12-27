using BenchmarkTools

const GPU_BACKENDS = ["AMDGPU", "CUDA", "Metal", "oneAPI"]
const NUM_CPU_THREADS = [1, 2, 4, 8]

const RESULTS = BenchmarkGroup()

# Aggregate CPU results
for n in NUM_CPU_THREADS
    filename = string("CPUbenchmarks", n, "threads.json")
    filepath = joinpath(@__DIR__, "results", filename)
    if ispath(filepath)
        nthreads_results = BenchmarkTools.load(filepath)[1]
        for benchmark in keys(nthreads_results)
            RESULTS[benchmark] = RESULTS.get(benchmark, BenchmarkGroup())
            for size in keys(nthreads_results[benchmark])
                RESULTS[benchmark][size] = RESULTS[benchmark].get(size, BenchmarkGroup())
                for algorithm in keys(nthreads_results[benchmark][size])
                    RESULTS[benchmark][size][algorithm] = RESULTS[benchmark][size].get(algorithm, BenchmarkGroup())
                    RESULTS[benchmark][size][algorithm]["CPU"] = RESULTS[benchmark][size][algorithm].get("CPU", BenchmarkGroup())
                    RESULTS[benchmark][size][algorithm]["CPU"][string(n, " thread(s)")] = nthreads_results[benchmark][size][algorithm]["CPU"][string(n, " thread(s)")]
                end
            end
        end
    else
        @warn "No CPU benchmark file found at path: $(filepath)"
    end
end

# Aggregate GPU results
for backend in GPU_BACKENDS
    filename = string(backend, "benchmarks.json")
    filepath = joinpath(@__DIR__, "results", filename)
    if ispath(filepath)
        backend_results = BenchmarkTools.load(filepath)[1]
        for benchmark in keys(backend_results)
            RESULTS[benchmark] = RESULTS.get(benchmark, BenchmarkGroup())
            for size in keys(backend_results[benchmark])
                RESULTS[benchmark][size] = RESULTS[benchmark].get(size, BenchmarkGroup())
                for algorithm in keys(backend_results[benchmark][size])
                    RESULTS[benchmark][size][algorithm] = RESULTS[benchmark][size].get(algorithm, BenchmarkGroup())
                    RESULTS[benchmark][size][algorithm]["GPU"] = RESULTS[benchmark][size][algorithm].get("GPU", BenchmarkGroup())
                    RESULTS[benchmark][size][algorithm]["GPU"][backend] = backend_results[benchmark][size][algorithm]["GPU"][backend]
                end
            end
        end
    else
        @warn "No GPU benchmark file found at path: $(filepath)"
    end
end

# Save the combined results
BenchmarkTools.save(joinpath(@__DIR__, "results", "combinedbenchmarks.json"), RESULTS)