include("./imports.jl")

@testset ExtendedTestSet "dice_loss" begin
    @testset ExtendedTestSet "dice_loss" begin
    ŷ = [1 1 1 0; 1 1 1 0; 1 1 1 0; 1 1 1 0]
    y = [1 1 1 0; 1 1 1 0; 1 1 1 0; 1 1 1 0]
    @test dice_loss(ŷ, y) == 0.0
    end

    @testset ExtendedTestSet "dice_loss" begin
        y1 = [1 1 1 0; 1 1 1 0; 1 1 1 0; 1 1 1 0]
        y2 = [1 1 1 0; 1 1 1 0; 1 1 1 0; 1 1 1 0]
        y = cat(y1, y2, dims=3)
        
        ŷ1 = [1 1 1 0; 1 1 1 0; 1 1 1 0; 1 1 1 0]
        ŷ2 = [1 1 1 0; 1 1 1 0; 1 1 1 0; 1 1 1 0]
        ŷ = cat(ŷ1, ŷ2, dims=3)
        @test dice_loss(ŷ, y) == 0.0
    end
end

@testset ExtendedTestSet "hd_loss" begin
    @testset ExtendedTestSet "hd_loss" begin
    y1 = [1 1 1 0; 1 1 1 0; 1 1 1 0; 1 1 1 0]
    y2 = [1 1 1 0; 1 1 1 0; 1 1 1 0; 1 1 1 0]
    y = cat(y1, y2, dims=3)
    
    ŷ1 = [1 1 1 0; 1 1 1 0; 1 1 1 0; 1 1 1 0]
    ŷ2 = [1 1 1 0; 1 1 1 0; 1 1 1 0; 1 1 1 0]
    ŷ = cat(ŷ1, ŷ2, dims=3)
    
    ŷ_dtm = compute_dtm(ŷ)
    y_dtm = compute_dtm(y)

    @test hd_loss(ŷ, y, ŷ_dtm, y_dtm) == 0.0
    end
end

@testset ExtendedTestSet "hd_lossP" begin
    @testset ExtendedTestSet "hd_lossP" begin
    y1 = [1 1 1 0; 1 1 1 0; 1 1 1 0; 1 1 1 0]
    y2 = [1 1 1 0; 1 1 1 0; 1 1 1 0; 1 1 1 0]
    y = cat(y1, y2, dims=3)
    
    ŷ1 = [1 1 1 0; 1 1 1 0; 1 1 1 0; 1 1 1 0]
    ŷ2 = [1 1 1 0; 1 1 1 0; 1 1 1 0; 1 1 1 0]
    ŷ = cat(ŷ1, ŷ2, dims=3)
    
    ŷ_dtm = compute_dtm(ŷ)
    y_dtm = compute_dtm(y)

    @test hd_lossP(ŷ, y, ŷ_dtm, y_dtm) == hd_loss(ŷ, y, ŷ_dtm, y_dtm)
    end
end