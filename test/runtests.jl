using DistanceTransforms
using Test

AVAILABLE_GPU_BACKENDS = ["CUDA", "AMDGPU", "Metal", "oneAPI"]
TEST_BACKENDS = filter(x->x in [AVAILABLE_GPU_BACKENDS; "CPU"], ARGS)

if isempty(TEST_BACKENDS)
    using Preferences
    TEST_BACKENDS = [load_preference(KomaMRICore, "test_backend", "CPU")]
    @info "Using [preferences.KomaMRICore]." backend=first(TEST_BACKENDS)
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
    backend = oneBackend()
    dev = oneArray
else
    @info "CPU using $(Threads.nthreads()) thread(s)" maxlog=1
    backend = CPU()
    dev = Array
end

using KernelAbstractions
using Random

include("transform.jl")
include("utils.jl")