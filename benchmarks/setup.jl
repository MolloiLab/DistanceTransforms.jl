using BenchmarkTools: BenchmarkGroup, @benchmarkable
using DistanceTransforms
using ImageMorphology: distance_transform, feature_transform
using KernelAbstractions
using Random, Statistics
import InteractiveUtils

const BENCHMARK_GROUP = get(ENV, "BENCHMARK_GROUP", "CPU")

InteractiveUtils.versioninfo()

# Function to monitor GPU memory based on backend
function monitor_gpu_memory(backend::String, duration=0.1)
    memory_readings = Float64[]
    timestamps = Float64[]
    
    function get_memory_usage()
        if backend == "Metal"
            device = Metal.device()
            return Float64(device.currentAllocatedSize) / (1024 * 1024)
        elseif backend == "oneAPI"
            # Get the first device since that's what we're using
            device = oneAPI.devices()[1]
            props = oneL0.memory_properties(device)
            return (props.totalSize - props.freeSize) / (1024 * 1024)
        elseif backend == "AMDGPU"
            free, total = AMDGPU.HIP.memory_info()
            return (total - free) / (1024 * 1024)
        elseif backend == "CUDA"
            free, total = CUDA.memory_info()
            return (total - free) / (1024 * 1024)
        end
        return 0.0
    end

    start_time = time()
    while (time() - start_time < duration)
        push!(memory_readings, get_memory_usage())
        push!(timestamps, time() - start_time)
        sleep(0.01)
    end

    peak_mb = maximum(memory_readings)
    mean_mb = mean(memory_readings)
    current_mb = last(memory_readings)
    return (peak_mb, mean_mb, current_mb)
end

# Benchmark wrapper that includes memory monitoring
function benchmark_with_memory(f, backend::String)
    b = @benchmarkable begin
        $f
        $(backend != "CPU" ? :(
            if $backend == "Metal"
                Metal.synchronize()
            elseif $backend == "CUDA"
                CUDA.synchronize()
            elseif $backend == "AMDGPU"
                AMDGPU.synchronize()
            elseif $backend == "oneAPI"
                oneAPI.synchronize()
            end
        ) : :())
    end setup=(GC.gc())
    
    if backend == "CPU"
        return (benchmark=b, peak_memory_mb=0.0, mean_memory_mb=0.0, current_memory_mb=0.0)
    end
    
    # Run once with memory monitoring
    GC.gc()
    mem_start = monitor_gpu_memory(backend, 0.001)
    f()
    mem_during = monitor_gpu_memory(backend, 0.1)
    
    peak_mb = maximum([mem_start[1], mem_during[1]])
    mean_mb = mem_during[2]
    current_mb = mem_during[3]
    
    return (benchmark=b, peak_memory_mb=peak_mb, mean_memory_mb=mean_mb, current_memory_mb=current_mb)
end

# Only load the selected backend to avoid unnecessary initializations
if BENCHMARK_GROUP == "Metal"
    using Metal
    Metal.versioninfo()
elseif BENCHMARK_GROUP == "CUDA"
    using CUDA
    CUDA.versioninfo()
elseif BENCHMARK_GROUP == "AMDGPU"
    using AMDGPU
    AMDGPU.versioninfo()
elseif BENCHMARK_GROUP == "oneAPI"
    using oneAPI
    oneAPI.versioninfo()
end

function setup_benchmarks(suite::BenchmarkGroup, backend::String, num_cpu_threads::Int64)
    sizes_2D = [2^i for i in 3:12]
    sizes_3D = [2^i for i in 0:8]

    if backend == "CPU"
        # 2D benchmarks
        for n in sizes_2D
            f = Float32.(rand([0, 1], n, n))
            bool_f = Bool.(f)
            suite["2D"]["Size_$n"]["Maurer"]["CPU"][string(num_cpu_threads, " thread(s)")] =
                benchmark_with_memory(() -> distance_transform(feature_transform(bool_f)), backend).benchmark

            suite["2D"]["Size_$n"]["Felzenszwalb"]["CPU"][string(num_cpu_threads, " thread(s)")] =
                benchmark_with_memory(() -> transform(boolean_indicator(f); threaded = false), backend).benchmark

            suite["2D"]["Size_$n"]["Felzenszwalb_MT"]["CPU"][string(num_cpu_threads, " thread(s)")] =
                benchmark_with_memory(() -> transform(boolean_indicator(f)), backend).benchmark
        end

        # 3D benchmarks
        for n in sizes_3D
            f = Float32.(rand([0, 1], n, n, n))
            bool_f = Bool.(f)
            suite["3D"]["Size_$n"]["Maurer"]["CPU"][string(num_cpu_threads, " thread(s)")] =
                benchmark_with_memory(() -> distance_transform(feature_transform(bool_f)), backend).benchmark

            suite["3D"]["Size_$n"]["Felzenszwalb"]["CPU"][string(num_cpu_threads, " thread(s)")] =
                benchmark_with_memory(() -> transform(boolean_indicator(f); threaded = false), backend).benchmark

            suite["3D"]["Size_$n"]["Felzenszwalb_MT"]["CPU"][string(num_cpu_threads, " thread(s)")] =
                benchmark_with_memory(() -> transform(boolean_indicator(f)), backend).benchmark
        end
    elseif backend == "Metal"
        @info "Running Metal benchmarks"
        for n in sizes_2D
            f = Float32.(rand([0, 1], n, n))
            f_gpu = MtlArray(f)
            suite["2D"]["Size_$n"]["Felzenszwalb"]["GPU"][backend] =
                benchmark_with_memory(() -> transform(boolean_indicator(f_gpu)), backend).benchmark
        end

        for n in sizes_3D
            f = Float32.(rand([0, 1], n, n, n))
            f_gpu = MtlArray(f)
            suite["3D"]["Size_$n"]["Felzenszwalb"]["GPU"][backend] =
                benchmark_with_memory(() -> transform(boolean_indicator(f_gpu)), backend).benchmark
        end
    elseif backend == "CUDA"
        @info "Running CUDA benchmarks"
        for n in sizes_2D
            f = Float32.(rand([0, 1], n, n))
            f_gpu = CUDA.CuArray(f)
            suite["2D"]["Size_$n"]["Felzenszwalb"]["GPU"][backend] =
                benchmark_with_memory(() -> transform(boolean_indicator(f_gpu)), backend).benchmark
        end

        for n in sizes_3D
            f = Float32.(rand([0, 1], n, n, n))
            f_gpu = CUDA.CuArray(f)
            suite["3D"]["Size_$n"]["Felzenszwalb"]["GPU"][backend] =
                benchmark_with_memory(() -> transform(boolean_indicator(f_gpu)), backend).benchmark
        end
    elseif backend == "AMDGPU"
        @info "Running AMDGPU benchmarks"
        for n in sizes_2D
            f = Float32.(rand([0, 1], n, n))
            f_gpu = ROCArray(f)
            suite["2D"]["Size_$n"]["Felzenszwalb"]["GPU"][backend] =
                benchmark_with_memory(() -> transform(boolean_indicator(f_gpu)), backend).benchmark
        end

        for n in sizes_3D
            f = Float32.(rand([0, 1], n, n, n))
            f_gpu = ROCArray(f)
            suite["3D"]["Size_$n"]["Felzenszwalb"]["GPU"][backend] =
                benchmark_with_memory(() -> transform(boolean_indicator(f_gpu)), backend).benchmark
        end
    elseif backend == "oneAPI"
        @info "Running oneAPI benchmarks"
        for n in sizes_2D
            f = Float32.(rand([0, 1], n, n))
            f_gpu = oneArray(f)
            suite["2D"]["Size_$n"]["Felzenszwalb"]["GPU"][backend] =
                benchmark_with_memory(() -> transform(boolean_indicator(f_gpu)), backend).benchmark
        end

        for n in sizes_3D
            f = Float32.(rand([0, 1], n, n, n))
            f_gpu = oneArray(f)
            suite["3D"]["Size_$n"]["Felzenszwalb"]["GPU"][backend] =
                benchmark_with_memory(() -> transform(boolean_indicator(f_gpu)), backend).benchmark
        end
    else
        @error "Unknown backend: $backend"
    end
end