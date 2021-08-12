include("./imports.jl")

@testset ExtendedTestSet "squared_euclidean" begin
    @testset ExtendedTestSet "squared_euclidean" begin
        x = boolean_indicator([0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0])
        dt = Array{Float32}(undef, size(x))
        v = ones(Int64, length(x))
        z = zeros(Float32, length(x) + 1)
        answer = [1, 0, 1, 4, 1, 0, 0, 0, 0, 0, 1]
        @test squared_euclidean(x, dt, v, z) == answer
    end

    @testset ExtendedTestSet "squared_euclidean" begin
        x = boolean_indicator([
            0 1 1 1 0
            1 1 1 1 1
            1 0 0 0 1
            1 0 0 0 1
            1 0 0 0 1
            1 1 1 1 1
            0 1 1 1 0
        ])
        dt = Array{Float32}(undef, size(x))
        v = ones(Int64, size(x))
        z = zeros(Float32, size(x) .+ 1)
        answer = [
            1.0 0.0 0.0 0.0 1.0
            0.0 0.0 0.0 0.0 0.0
            0.0 1.0 1.0 1.0 0.0
            0.0 4.0 4.0 4.0 0.0
            0.0 1.0 1.0 1.0 0.0
            0.0 0.0 0.0 0.0 0.0
            1.0 0.0 0.0 0.0 1.0
        ]
        @test squared_euclidean(x, dt, v, z) == answer
    end

    @testset ExtendedTestSet "squared_euclidean" begin
        x = boolean_indicator([
            0 1 1 1 0
            1 1 1 1 1
            1 0 0 0 1
            1 0 0 0 1
            1 0 0 0 1
            1 1 1 1 1
            0 1 1 1 0
        ])
        dt = Array{Float32}(undef, size(x))
        v = ones(Int64, size(x))
        z = zeros(Float32, size(x) .+ 1)
        answer = [
            1.0 0.0 0.0 0.0 1.0
            0.0 0.0 0.0 0.0 0.0
            0.0 1.0 1.0 1.0 0.0
            0.0 4.0 4.0 4.0 0.0
            0.0 1.0 1.0 1.0 0.0
            0.0 0.0 0.0 0.0 0.0
            1.0 0.0 0.0 0.0 1.0
        ]
        @test squared_euclidean(x, dt, v, z) == answer
    end

    @testset ExtendedTestSet "squared_euclidean!" begin
        x = boolean_indicator([
            0 1 1 1 0
            1 1 1 1 1
            1 0 0 0 1
            1 0 0 0 1
            1 0 0 0 1
            1 1 1 1 1
            0 1 1 1 0
        ])
        v = ones(Int64, size(x))
        z = zeros(Float32, size(x) .+ 1)
        dt = Array{Float32}(undef, size(x))
        nthreads = Threads.nthreads()
        answer = [
            1.0 0.0 0.0 0.0 1.0
            0.0 0.0 0.0 0.0 0.0
            0.0 1.0 1.0 1.0 0.0
            0.0 4.0 4.0 4.0 0.0
            0.0 1.0 1.0 1.0 0.0
            0.0 0.0 0.0 0.0 0.0
            1.0 0.0 0.0 0.0 1.0
        ]
        @test squared_euclidean!(x, dt, v, z, nthreads) == answer
    end

    # # TODO: Figure out how to import CUDA, FLoops, and FoldsCUDA only for testing
    # @testset ExtendedTestSet "squared_euclidean!" begin
    #     x = CuArray(boolean_indicator([
    #         0 1 1 1 0
    #         1 1 1 1 1
    #         1 0 0 0 1
    #         1 0 0 0 1
    #         1 0 0 0 1
    #         1 1 1 1 1
    #         0 1 1 1 0]))
    #     dt = CuArray{Float32}(undef, size(x4))
    #     v = CUDA.ones(Int64, size(x4))
    #     z = CUDA.zeros(Float32, size(x4) .+ 1)
    #     answer = [
    #         1.0 0.0 0.0 0.0 1.0
    #         0.0 0.0 0.0 0.0 0.0
    #         0.0 1.0 1.0 1.0 0.0
    #         0.0 4.0 4.0 4.0 0.0
    #         0.0 1.0 1.0 1.0 0.0
    #         0.0 0.0 0.0 0.0 0.0
    #         1.0 0.0 0.0 0.0 1.0
    #     ]
    #     @test squared_euclidean!(x, dt, v, z, nthreads) == answer
    # end
end
