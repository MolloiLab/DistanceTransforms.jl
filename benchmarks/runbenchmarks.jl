using BenchmarkTools: BenchmarkTools, BenchmarkGroup, @btime, @benchmarkable
using InteractiveUtils: versioninfo

const BENCHMARK_GROUP = get(ENV, "BENCHMARK_GROUP", "CPU")
@info "Running benchmarks for $BENCHMARK_GROUP"
@info sprint(versioninfo)
@info "Number of threads: $(Threads.nthreads())"


const BENCHMARK_CPU_THREADS = Threads.nthreads()
# Number of CPU threads to benchmarks on
if BENCHMARK_CPU_THREADS > Threads.nthreads()
    @error """
    More CPU threads were requested than are available. Change the
    JULIA_NUM_THREADS environment variable or pass
    --threads=$(BENCHMARK_CPU_THREADS) as a julia argument
    """
end