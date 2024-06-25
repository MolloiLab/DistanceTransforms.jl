using DistanceTransforms
using Test

using KernelAbstractions
using CUDA, AMDGPU, Metal
using oneAPI
using Random

using BenchmarkTools

if CUDA.functional()
    @info "Using CUDA"
    CUDA.versioninfo()
    backend = CUDABackend()
    dev = CuArray
elseif AMDGPU.functional()
    @info "Using AMD"
    AMDGPU.versioninfo()
    backend = ROCBackend()
    dev = ROCArray
elseif oneAPI.functional() ## not well supported
    @info "Using oneAPI"
    oneAPI.versioninfo()
    backend = oneBackend()
    dev = oneArray
elseif Metal.functional()
    @info "Using Metal"
    Metal.versioninfo()
    backend = MetalBackend()
    dev = MtlArray
else
    @info "No GPU is available. Using CPU."
    backend = CPU()
    dev = Array
end

include("transform.jl")
include("utils.jl")

# Run benchmarks
@testset "Benchmarks" begin
    s = 10 # number of samples
    e = 1 # number of evals

    # Create an array of input sizes to benchmark for 2D
    sizes_2D = [2^i for i in 3:4]

    # Create DataFrames to store the benchmark results
    df_dt_2D = DataFrame(
        os_info = os_info,
        gpu_info = gpu_info,
        sizes = sizes_2D,
        dt_proposed = Float64[]
    )

    for size in sizes_2D
        f = Float32.(rand([0, 1], size, size))

        if dev != Array
            f_dev = dev(f)
            dt = @benchmark(transform($boolean_indicator($f_dev)); samples=s, evals=e)
            push!(df_dt_2D, [os_info, gpu_info, size, minimum(dt).time])
        end
    end

    # Create an array of input sizes to benchmark for 3D
    sizes_3D = [2^i for i in 0:2]

    # Create DataFrames to store the benchmark results
    df_dt_3D = DataFrame(
        os_info = os_info,
        gpu_info = gpu_info,
        sizes = sizes_3D,
        dt_proposed = Float64[]
    )

    for size in sizes_3D
        f = Float32.(rand([0, 1], size, size, size))

        if dev != Array
            f_dev = dev(f)
            dt = @benchmark(transform($boolean_indicator($f_dev)); samples=s, evals=e)
            push!(df_dt_3D, [os_info, gpu_info, size, minimum(dt).time])
        end
    end

    # Show the dataframes
    @show df_dt_2D
    @show df_dt_3D
end