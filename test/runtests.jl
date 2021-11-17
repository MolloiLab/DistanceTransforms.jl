include("imports.jl")

include("chamfer.jl")
include("euclidean.jl")
include("squared_euclidean.jl")
include("utils.jl")

@testset "CUDA" begin
    if CUDA.functional()
        include("cuda/runtests.jl")
    else
        @warn "CUDA unavailable, not testing GPU support"
    end
end
