include("./imports.jl")

@testset ExtendedTestSet "squared_euclidean_distance_transform" begin
    @testset ExtendedTestSet "squared_euclidean_distance_transform" begin
        x = boolean_indicator([0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0])
        tfm = SquaredEuclidean(x)
        answer = [1, 0, 1, 4, 1, 0, 0, 0, 0, 0, 1]
        @test transform(x, tfm) == answer
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
        tfm = SquaredEuclidean(x)
        answer = [
            1.0 0.0 0.0 0.0 1.0
            0.0 0.0 0.0 0.0 0.0
            0.0 1.0 1.0 1.0 0.0
            0.0 4.0 4.0 4.0 0.0
            0.0 1.0 1.0 1.0 0.0
            0.0 0.0 0.0 0.0 0.0
            1.0 0.0 0.0 0.0 1.0
        ]
        @test transform(x, tfm) == answer
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
        tfm = SquaredEuclidean(x)
        answer = [
            1.0 0.0 0.0 0.0 1.0
            0.0 0.0 0.0 0.0 0.0
            0.0 1.0 1.0 1.0 0.0
            0.0 4.0 4.0 4.0 0.0
            0.0 1.0 1.0 1.0 0.0
            0.0 0.0 0.0 0.0 0.0
            1.0 0.0 0.0 0.0 1.0
        ]
        @test transform(x, tfm) == answer
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
        tfm = SquaredEuclidean(x)
        answer = [
            1.0 0.0 0.0 0.0 1.0
            0.0 0.0 0.0 0.0 0.0
            0.0 1.0 1.0 1.0 0.0
            0.0 4.0 4.0 4.0 0.0
            0.0 1.0 1.0 1.0 0.0
            0.0 0.0 0.0 0.0 0.0
            1.0 0.0 0.0 0.0 1.0
        ]
        @test transform!(x, tfm, Threads.nthreads()) == answer
    end

    @testset ExtendedTestSet "squared_euclidean_distance_transform" begin
        x = CUDA.CuArray(boolean_indicator([
            0 1 1 1 0
            1 1 1 1 1
            1 0 0 0 1
            1 0 0 0 1
            1 0 0 0 1
            1 1 1 1 1
            0 1 1 1 0
        ]))
        tfm = SquaredEuclidean(x)
        answer = CUDA.CuArray([
            1.0 0.0 0.0 0.0 1.0
            0.0 0.0 0.0 0.0 0.0
            0.0 1.0 1.0 1.0 0.0
            0.0 4.0 4.0 4.0 0.0
            0.0 1.0 1.0 1.0 0.0
            0.0 0.0 0.0 0.0 0.0
            1.0 0.0 0.0 0.0 1.0
        ])
        @test transform!(x, tfm) == answer
    end
end
