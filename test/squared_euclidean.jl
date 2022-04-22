@testset ExtendedTestSet "squared_euclidean" begin
    @testset ExtendedTestSet "squared_euclidean 1D" begin
        x = boolean_indicator([0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0])
        tfm = SquaredEuclidean(x)
        answer = [1, 0, 1, 4, 1, 0, 0, 0, 0, 0, 1]
        @test transform(x, tfm) == answer
    end

    @testset ExtendedTestSet "squared_euclidean 2D" begin
        x = boolean_indicator(
            [
                0 1 1 1 0
                1 1 1 1 1
                1 0 0 0 1
                1 0 0 0 1
                1 0 0 0 1
                1 1 1 1 1
                0 1 1 1 0
            ]
        )
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

    @testset ExtendedTestSet "squared_euclidean 3D" begin
        x1 = [
                0 1 1 1 0
                1 1 1 1 1
                1 0 0 0 1
                1 0 0 0 1
                1 0 0 0 1
                1 1 1 1 1
                0 1 1 1 0
                ]
        
        x = cat(x1, x1, dims=3)
        x = boolean_indicator(x)

        tfm = SquaredEuclidean(x)
        a1 = [
            1.0 0.0 0.0 0.0 1.0
            0.0 0.0 0.0 0.0 0.0
            0.0 1.0 1.0 1.0 0.0
            0.0 4.0 4.0 4.0 0.0
            0.0 1.0 1.0 1.0 0.0
            0.0 0.0 0.0 0.0 0.0
            1.0 0.0 0.0 0.0 1.0
        ]
        answer = cat(a1, a1, dims=3)
        @test transform(x, tfm) == answer
    end

    @testset ExtendedTestSet "squared_euclidean threaded" begin
        x = boolean_indicator(
            [
                0 1 1 1 0
                1 1 1 1 1
                1 0 0 0 1
                1 0 0 0 1
                1 0 0 0 1
                1 1 1 1 1
                0 1 1 1 0
            ]
        )
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
end
