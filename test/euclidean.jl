include("./imports.jl")

@testset ExtendedTestSet "euclidean" begin
    @testset ExtendedTestSet "euclidean" begin
        x = [
            1 1 0 0
            0 1 1 0
            0 1 0 1
            0 1 0 0
        ]
        answer = [
            1.0 1.0 0.0 0.0
            0.0 1.0 1.0 0.0
            0.0 1.0 0.0 1.0
            0.0 1.0 0.0 0.0
        ]
        @test euclidean(x) == answer
    end

    @testset ExtendedTestSet "euclidean" begin
        x = [
            1 1 1 0
            0 1 1 1
            0 1 1 1
            0 1 1 1
        ]
        answer = [
            1.0 1.4142135623730951 1.0 0.0
            0.0 1.0 1.4142135623730951 1.0
            0.0 1.0 2.0 2.0
            0.0 1.0 2.0 3.0
        ]
        @test euclidean(x) ≈ answer
    end
end
