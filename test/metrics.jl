using DistanceTransforms
using Test
using Distances
using Distances: Euclidean
using ImageEdgeDetection

@testset ExtendedTestSet "mean_hausdorff_2D" begin
    @testset ExtendedTestSet "mean_hausdorff_2D" begin
    x = [1 1 1 0; 1 1 1 0; 1 1 1 0; 1 1 1 0]
    y = [1 1 1 0; 1 1 1 0; 1 1 1 0; 1 1 1 0]
    @test mean_hausdorff_2D(x, y, Euclidean(), Canny()) ≈ 0
    end
end

@testset ExtendedTestSet "mean_dice" begin
    @testset ExtendedTestSet "mean_dice" begin
    x = [1 1 1 0; 1 1 1 0; 1 1 1 0; 1 1 1 0]
    y = [1 1 1 0; 1 1 1 0; 1 1 1 0; 1 1 1 0]
    @test mean_dice(x, y) ≈ 1
    end
end