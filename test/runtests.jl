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

# Benchmark the transform function
include("benchmarks.jl")