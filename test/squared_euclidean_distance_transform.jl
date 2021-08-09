include("./imports.jl")

@testset ExtendedTestSet "squared_euclidean_distance_transform" begin
    @testset ExtendedTestSet "squared_euclidean_distance_transform" begin
        x = boolean_indicator([0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0])
        dt = Array{Float32}(undef, size(x))
        v = ones(Int64, length(x))
        z = zeros(Float32, length(x) + 1)
        answer = [1, 0, 1, 4, 1, 0, 0, 0, 0, 0, 1]
        @test squared_euclidean_distance_transform(x, dt, v, z) == answer
    end

    @testset ExtendedTestSet "squared_euclidean_distance_transform" begin
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
        @test squared_euclidean_distance_transform(x, dt, v, z) == answer
    end

    @testset ExtendedTestSet "squared_euclidean_distance_transform" begin
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
        @test squared_euclidean_distance_transform(x, dt, v, z) == answer
    end

    @testset ExtendedTestSet "squared_euclidean_distance_transform" begin
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
        answer = [
            1.0 0.0 0.0 0.0 1.0
            0.0 0.0 0.0 0.0 0.0
            0.0 1.0 1.0 1.0 0.0
            0.0 4.0 4.0 4.0 0.0
            0.0 1.0 1.0 1.0 0.0
            0.0 0.0 0.0 0.0 0.0
            1.0 0.0 0.0 0.0 1.0
        ]
        @test squared_euclidean_distance_transform(x, dt, v, z, Threads.nthreads()) == answer
    end
end
