include("./imports.jl")

@testset ExtendedTestSet "chamfer_distance" begin
    @testset ExtendedTestSet "chamfer_distance" begin
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
        @test chamfer_distance(x) == answer
    end

    @testset ExtendedTestSet "chamfer_distance" begin
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
        @test chamfer_distance(x) == answer
    end
end

@testset ExtendedTestSet "chamfer_distance3D" begin
    @testset ExtendedTestSet "chamfer_distance3D" begin
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
        x3D = cat(x1, x2; dims=3)

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
        @test chamfer_distance3D(x3D) == answer
    end
end
