include("./imports.jl")

@testset ExtendedTestSet "dice_metric" begin
    @testset ExtendedTestSet "dice_metric" begin
    x = [1 1 1 0; 1 1 1 0; 1 1 1 0; 1 1 1 0]
    y = [1 1 1 0; 1 1 1 0; 1 1 1 0; 1 1 1 0]
    @test dice_metric(x, y) ≈ 1
    end

    @testset ExtendedTestSet "dice_metric" begin
    x = [0 0 0 1; 0 0 0 1; 0 0 0 1; 0 0 0 1]
    y = [1 1 1 0; 1 1 1 0; 1 1 1 0; 1 1 1 0]
    @test dice_metric(x, y) ≈ 0
    end
end

@testset ExtendedTestSet "mean_hausdorff_2D" begin
    @testset ExtendedTestSet "mean_hausdorff_2D" begin
    x = [1 1 1 0; 1 1 1 0; 1 1 1 0; 1 1 1 0]
    y = [1 1 1 0; 1 1 1 0; 1 1 1 0; 1 1 1 0]
    @test mean_hausdorff_2D(x, y, Euclidean(), Canny()) ≈ 0
    end
end