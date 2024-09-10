using BenchmarkTools: BenchmarkTools, BenchmarkGroup, @benchmark, @benchmarkable
using InteractiveUtils: versioninfo
using ImageMorphology: distance_transform, feature_transform
using DistanceTransforms: boolean_indicator, transform

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

# range_size_2D = [2^i for i in 3:12]
# range_names_2D = [L"2^3", L"2^4", L"2^5", L"2^6", L"2^7", L"2^8", L"2^9", L"2^{10}", L"2^{11}", L"2^{12}"]

range_size_2D = [2^i for i in 3:4]

begin
    sizes = Float64[]

    dt_maurer = Float64[]
    dt_maurer_std = Float64[]

    dt_fenz = Float64[]
    dt_fenz_std = Float64[]

    dt_fenz_multi = Float64[]
    dt_fenz_multi_std = Float64[]
    
	for n in range_size_2D
		@info n
		push!(sizes, n^2)
		f = Float32.(rand([0, 1], n, n))
	
		# Maurer (ImageMorphology.jl)
		bool_f = Bool.(f)
		dt = @benchmark(distance_transform($feature_transform($bool_f)), samples=s, evals=e)
		push!(dt_maurer, minimum(dt).time) # ns
		push!(dt_maurer_std, std(dt).time)
		
		# Felzenszwalb (DistanceTransforms.jl)
		dt = @benchmark(transform($boolean_indicator($f); threaded = false), samples=s, evals=e)
		push!(dt_fenz, minimum(dt).time)
		push!(dt_fenz_std, std(dt).time)
	
		# Felzenszwalb Multi-threaded (DistanceTransforms.jl)
		dt = @benchmark(transform($boolean_indicator($f)), samples=s, evals=e)
		push!(dt_fenz_multi, minimum(dt).time)
		push!(dt_fenz_multi_std, std(dt).time)
	
	end
end

@info sizes
@info dt_maurer
@info dt_fenz
@info dt_fenz_multi
