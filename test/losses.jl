using DistanceTransforms
using Test

@testset ExtendedTestSet "dice_loss" begin
    @testset ExtendedTestSet "dice_loss" begin
    x = [1 1 1 0; 1 1 1 0; 1 1 1 0; 1 1 1 0]
    y = [1 1 1 0; 1 1 1 0; 1 1 1 0; 1 1 1 0]
    @test dice_loss(x, y) ≈ 0
    end
end