include("./imports.jl")

@testset ExtendedTestSet "detect_edges_3D" begin
    @testset ExtendedTestSet "detect_edges_3D" begin
        x1 = [1 1 1 0; 1 1 1 0]
        x2 = [1 1 1 0; 1 1 1 0]
        x3 = [1 1 1 0; 1 1 1 0]
        x = cat(x1, x2, x3, dims=3)

        e1 = [0 0 1 0; 0 0 1 0]
        e2 = [0 0 1 0; 0 0 1 0]
        e3 = [0 0 1 0; 0 0 1 0]
        edge = cat(e1, e2, e3, dims=3)
    @test detect_edges_3D(x, Canny()) ≈ edge
    end
end