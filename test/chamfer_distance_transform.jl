include("./imports.jl")

@testset ExtendedTestSet "chamfer_distance_transform" begin
    @testset ExtendedTestSet "chamfer_distance_transform" begin
        x = [
            1 1 0 0
            0 1 1 0
            0 1 0 1
            0 1 0 0
        ]
        answer = [
            0 0 3 4
            3 0 0 3
            3 0 3 0
            3 0 3 3
        ]
        tfm = Chamfer(x)
        @test transform(x, tfm) == answer
    end

    @testset ExtendedTestSet "chamfer_distance_transform" begin
        x = [
            1 1 0 0
            0 1 0 0
            0 1 0 0
            0 1 0 0
        ]
        answer = [
            0 0 3 6
            3 0 3 6
            3 0 3 6
            3 0 3 6
        ]
        tfm = Chamfer(x)
        @test transform(x, tfm) == answer
    end

    @testset ExtendedTestSet "chamfer_distance_transform" begin
        x1 = [
            1 1 0 0
            0 1 0 0
            0 1 0 0
            0 1 0 0
        ]
        x2 = [
            1 1 0 0
            0 1 0 0
            0 1 0 0
            0 1 0 0
        ]
        x = cat(x1, x2; dims=3)

        a1 = [
            0 0 3 6
            3 0 3 6
            3 0 3 6
            3 0 3 6
        ]
        a2 = [
            0 0 3 6
            3 0 3 6
            3 0 3 6
            3 0 3 6
        ]
        answer = cat(a1, a2; dims=3)
        tfm = Chamfer(x)
        @test transform(x, tfm) == answer
    end
end
