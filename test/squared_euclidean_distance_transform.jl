include("./imports.jl")

@testset ExtendedTestSet "squared_euclidean_distance_transform" begin
    @testset ExtendedTestSet "squared_euclidean_distance_transform" begin
        x = boolean_indicator([0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0])
        dt = Array{Float32}(undef, size(x))
        answer = [1, 0, 1, 4, 1, 0, 0, 0, 0, 0, 1]
        @test squared_euclidean_distance_transform(x, dt) == answer
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
        answer = [
            1.0 0.0 0.0 0.0 1.0
            0.0 0.0 0.0 0.0 0.0
            0.0 1.0 1.0 1.0 0.0
            0.0 4.0 4.0 4.0 0.0
            0.0 1.0 1.0 1.0 0.0
            0.0 0.0 0.0 0.0 0.0
            1.0 0.0 0.0 0.0 1.0
        ]
        @test squared_euclidean_distance_transform(x, dt) == answer
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
        answer = [
            1.0 0.0 0.0 0.0 1.0
            0.0 0.0 0.0 0.0 0.0
            0.0 1.0 1.0 1.0 0.0
            0.0 4.0 4.0 4.0 0.0
            0.0 1.0 1.0 1.0 0.0
            0.0 0.0 0.0 0.0 0.0
            1.0 0.0 0.0 0.0 1.0
        ]
        @test squared_euclidean_distance_transform(x, dt) == answer
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
        answer = [
            1.0 0.0 0.0 0.0 1.0
            0.0 0.0 0.0 0.0 0.0
            0.0 1.0 1.0 1.0 0.0
            0.0 4.0 4.0 4.0 0.0
            0.0 1.0 1.0 1.0 0.0
            0.0 0.0 0.0 0.0 0.0
            1.0 0.0 0.0 0.0 1.0
        ]
        @test squared_euclidean_distance_transform(x, dt, Threads.nthreads()) == answer
    end
end
