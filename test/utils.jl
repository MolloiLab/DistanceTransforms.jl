include("./imports.jl")

@testset ExtendedTestSet "euc" begin
    @testset ExtendedTestSet "euc" begin
        A = CartesianIndex(1, 3, 1)
        B = CartesianIndex(1, 3, 1)
        @test euc(A, B) == 0
    end

    @testset ExtendedTestSet "euc" begin
        A = CartesianIndex(1, 3, 1)
        C = CartesianIndex(1, 3, 2)
        @test euc(A, C) == 1.0
    end
end

@testset ExtendedTestSet "find_edges" begin
    @testset ExtendedTestSet "find_edges" begin
        y1 = [1 1 1 0; 1 1 1 0; 1 1 1 0; 1 1 1 0]
        y2 = [1 1 1 0; 1 1 1 0; 1 1 1 0; 1 1 1 0]
        y = cat(y1, y2; dims=3)

        C = CartesianIndex{3}[
            CartesianIndex(1, 3, 1),
            CartesianIndex(2, 3, 1),
            CartesianIndex(3, 3, 1),
            CartesianIndex(4, 3, 1),
            CartesianIndex(1, 3, 2),
            CartesianIndex(2, 3, 2),
            CartesianIndex(3, 3, 2),
            CartesianIndex(4, 3, 2),
        ]
        @test find_edges(y) == C
    end
    @testset ExtendedTestSet "find_edges" begin
        x1 = [1 1 1 0; 1 1 1 0]
        x2 = [1 1 1 0; 1 1 1 0]
        x = cat(x1, x2; dims=3)

        C = CartesianIndex{3}[
            CartesianIndex(1, 3, 1)
            CartesianIndex(2, 3, 1)
            CartesianIndex(1, 3, 2)
            CartesianIndex(2, 3, 2)
        ]
        @test find_edges(x) == C
    end
end

@testset ExtendedTestSet "detect_edges_3D" begin
    @testset ExtendedTestSet "detect_edges_3D" begin
        x1 = [1 1 1 0; 1 1 1 0]
        x2 = [1 1 1 0; 1 1 1 0]
        x3 = [1 1 1 0; 1 1 1 0]
        x = cat(x1, x2, x3; dims=3)

        e1 = [0 0 1 0; 0 0 1 0]
        e2 = [0 0 1 0; 0 0 1 0]
        e3 = [0 0 1 0; 0 0 1 0]
        edge = cat(e1, e2, e3; dims=3)
        @test detect_edges_3D(x, Canny()) ≈ edge
    end
end
