using DistanceTransforms
using Test

@testset ExtendedTestSet "mean_hausdorff" begin
    @testset ExtendedTestSet "mean_hausdorff" begin
    x = [1 1 1 0; 1 1 1 0; 1 1 1 0; 1 1 1 0]
    y = [1 1 1 0; 1 1 1 0; 1 1 1 0; 1 1 1 0]
    @test mean_hausdorff(x, y) ≈ 0
    end
end

@testset ExtendedTestSet "mean_dice" begin
    @testset ExtendedTestSet "mean_dice" begin
    x = [1 1 1 0; 1 1 1 0; 1 1 1 0; 1 1 1 0]
    y = [1 1 1 0; 1 1 1 0; 1 1 1 0; 1 1 1 0]
    @test mean_dice(x, y) ≈ 1
    end
end