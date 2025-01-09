using DistanceTransforms
using Test
using KernelAbstractions
using Random

#= 
To run the tests locally, and still test a GPU backend (e.g. Metal), use the following command:
```
using Pkg
Pkg.test("DistanceTransforms", test_args=["Metal"])
```
=#

AVAILABLE_GPU_BACKENDS = ["CUDA", "AMDGPU", "Metal", "oneAPI"]
TEST_BACKENDS = filter(x->x in [AVAILABLE_GPU_BACKENDS; "CPU"], ARGS)

if isempty(TEST_BACKENDS)
    TEST_BACKENDS = ["CPU"]
    @info "Using $(backend=first(TEST_BACKENDS))"
else
    @info "Using test_args" backend=first(TEST_BACKENDS)
end

USE_GPU = any(AVAILABLE_GPU_BACKENDS .∈ Ref(TEST_BACKENDS))

if "CUDA" in TEST_BACKENDS
    using CUDA
    CUDA.versioninfo()
    backend = CUDABackend()
    dev = CuArray
elseif "AMDGPU" in TEST_BACKENDS
    using AMDGPU
    AMDGPU.versioninfo()
    backend = ROCBackend()
    dev = ROCArray
elseif "Metal" in TEST_BACKENDS
    using Metal
    Metal.versioninfo()
    backend = MetalBackend()
    dev = MtlArray
elseif "oneAPI" in TEST_BACKENDS
    using oneAPI
    oneAPI.versioninfo()
    backend = oneAPIBackend()
    dev = oneArray
else
    using KernelAbstractions: CPU
    @info "CPU using $(Threads.nthreads()) thread(s)" maxlog=1
    backend = CPU()
    dev = Array
end

include("transform.jl")
include("utils.jl")