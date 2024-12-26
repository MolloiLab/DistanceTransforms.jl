using BenchmarkTools: BenchmarkGroup, @benchmarkable
using DistanceTransforms
using ImageMorphology: distance_transform, feature_transform
using KernelAbstractions
using Random
import InteractiveUtils

const BENCHMARK_GROUP = get(ENV, "BENCHMARK_GROUP", "CPU")

InteractiveUtils.versioninfo()

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
    # sizes_2D = [2^i for i in 3:12]
    # sizes_3D = [2^i for i in 0:8]

    # sizes_2D = [2^i for i in 3:4]
    # sizes_3D = [2^i for i in 0:1]

    if backend == "CPU"
        # 2D benchmarks
        for n in sizes_2D
            f = Float32.(rand([0, 1], n, n))
            bool_f = Bool.(f)
            suite["2D"]["Size_$n"]["Maurer"]["CPU"][string(num_cpu_threads, " thread(s)")] =
                @benchmarkable distance_transform(feature_transform($bool_f)) setup=(GC.gc())

            suite["2D"]["Size_$n"]["Felzenszwalb"]["CPU"][string(num_cpu_threads, " thread(s)")] =
                @benchmarkable transform($boolean_indicator($f); threaded = false) setup=(GC.gc())

            suite["2D"]["Size_$n"]["Felzenszwalb_MT"]["CPU"][string(num_cpu_threads, " thread(s)")] =
                @benchmarkable transform($boolean_indicator($f)) setup=(GC.gc())
        end

        # 3D benchmarks
        for n in sizes_3D
            f = Float32.(rand([0, 1], n, n, n))
            bool_f = Bool.(f)
            suite["3D"]["Size_$n"]["Maurer"]["CPU"][string(num_cpu_threads, " thread(s)")] =
                @benchmarkable distance_transform(feature_transform($bool_f)) setup=(GC.gc())

            suite["3D"]["Size_$n"]["Felzenszwalb"]["CPU"][string(num_cpu_threads, " thread(s)")] =
                @benchmarkable transform($boolean_indicator($f); threaded = false) setup=(GC.gc())

            suite["3D"]["Size_$n"]["Felzenszwalb_MT"]["CPU"][string(num_cpu_threads, " thread(s)")] =
                @benchmarkable transform($boolean_indicator($f)) setup=(GC.gc())
        end
    elseif backend == "Metal"
        @info "Running Metal benchmarks"
        # 2D benchmarks
        for n in sizes_2D
            f = Float32.(rand([0, 1], n, n))
            f_gpu = MtlArray(f)
            suite["2D"]["Size_$n"]["Felzenszwalb"]["GPU"][backend] =
                @benchmarkable transform($boolean_indicator($f_gpu)) setup=(GC.gc())
        end

        # 3D benchmarks
        for n in sizes_3D
            f = Float32.(rand([0, 1], n, n, n))
            f_gpu = MtlArray(f)
            suite["3D"]["Size_$n"]["Felzenszwalb"]["GPU"][backend] =
                @benchmarkable transform($boolean_indicator($f_gpu)) setup=(GC.gc())
        end
    elseif backend == "CUDA"
        @info "Running CUDA benchmarks"
        # 2D benchmarks
        for n in sizes_2D
            f = Float32.(rand([0, 1], n, n))
            f_gpu = CUDA.CuArray(f)
            suite["2D"]["Size_$n"]["Felzenszwalb"]["GPU"][backend] =
                @benchmarkable transform($boolean_indicator($f_gpu)) setup=(GC.gc())
        end

        # 3D benchmarks
        for n in sizes_3D
            f = Float32.(rand([0, 1], n, n, n))
            f_gpu = CUDA.CuArray(f)
            suite["3D"]["Size_$n"]["Felzenszwalb"]["GPU"][backend] =
                @benchmarkable transform($boolean_indicator($f_gpu)) setup=(GC.gc())
        end
    elseif backend == "AMDGPU"
        @info "Running AMDGPU benchmarks"
        # 2D benchmarks
        for n in sizes_2D
            f = Float32.(rand([0, 1], n, n))
            f_gpu = ROCArray(f)
            suite["2D"]["Size_$n"]["Felzenszwalb"]["GPU"][backend] =
                @benchmarkable transform($boolean_indicator($f_gpu)) setup=(GC.gc())
        end

        # 3D benchmarks
        for n in sizes_3D
            f = Float32.(rand([0, 1], n, n, n))
            f_gpu = ROCArray(f)
            suite["3D"]["Size_$n"]["Felzenszwalb"]["GPU"][backend] =
                @benchmarkable transform($boolean_indicator($f_gpu)) setup=(GC.gc())
        end
    elseif backend == "oneAPI"
        @info "Running oneAPI benchmarks"
        # 2D benchmarks
        for n in sizes_2D
            f = Float32.(rand([0, 1], n, n))
            f_gpu = oneArray(f)
            suite["2D"]["Size_$n"]["Felzenszwalb"]["GPU"][backend] =
                @benchmarkable transform($boolean_indicator($f_gpu)) setup=(GC.gc())
        end

        # 3D benchmarks
        for n in sizes_3D
            f = Float32.(rand([0, 1], n, n, n))
            f_gpu = oneArray(f)
            suite["3D"]["Size_$n"]["Felzenszwalb"]["GPU"][backend] =
                @benchmarkable transform($boolean_indicator($f_gpu)) setup=(GC.gc())
        end
    else
        @error "Unknown backend: $backend"
    end
end