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
        y = cat(y1, y2; dims=3)

        ŷ1 = [1 1 1 0; 1 1 1 0; 1 1 1 0; 1 1 1 0]
        ŷ2 = [1 1 1 0; 1 1 1 0; 1 1 1 0; 1 1 1 0]
        ŷ = cat(ŷ1, ŷ2; dims=3)
        @test dice_loss(ŷ, y) == 0.0
    end
end

@testset ExtendedTestSet "hd_loss" begin
    @testset ExtendedTestSet "hd_loss" begin
        y1 = [1 1 1 0; 1 1 1 0; 1 1 1 0; 1 1 1 0]
        y2 = [1 1 1 0; 1 1 1 0; 1 1 1 0; 1 1 1 0]
        y = cat(y1, y2; dims=3)

        ŷ1 = [1 1 1 0; 1 1 1 0; 1 1 1 0; 1 1 1 0]
        ŷ2 = [1 1 1 0; 1 1 1 0; 1 1 1 0; 1 1 1 0]
        ŷ = cat(ŷ1, ŷ2; dims=3)

        ŷ_dtm = euclidean_distance_transform(ŷ)
        y_dtm = euclidean_distance_transform(y)

        @test hd_loss(ŷ, y, ŷ_dtm, y_dtm) == 0.0
    end

    @testset ExtendedTestSet "hd_loss" begin
        y1 = [1 1 1 0; 1 1 1 0; 1 1 1 0; 1 1 1 0]
        y2 = [1 1 1 0; 1 1 1 0; 1 1 1 0; 1 1 1 0]
        y = cat(y1, y2; dims=3)

        ŷ1 = [1 1 1 0; 1 1 1 0; 1 1 1 0; 1 1 1 0]
        ŷ2 = [1 1 1 0; 1 1 1 0; 1 1 1 0; 1 1 1 0]
        ŷ = cat(ŷ1, ŷ2; dims=3)

        ŷ_dtm = chamfer_distance_transform3D(ŷ)
        y_dtm = chamfer_distance_transform3D(y)

        @test hd_loss(ŷ, y, ŷ_dtm, y_dtm) == 0.0
    end
end

@testset ExtendedTestSet "hd_lossP" begin
    @testset ExtendedTestSet "hd_lossP" begin
        y1 = [1 1 1 0; 1 1 1 0; 1 1 1 0; 1 1 1 0]
        y2 = [1 1 1 0; 1 1 1 0; 1 1 1 0; 1 1 1 0]
        y = cat(y1, y2; dims=3)
        y = cat(y, y; dims=4)
        y = cat(y, y; dims=5)

        ŷ1 = [1 1 1 0; 1 1 1 0; 1 1 1 0; 1 1 1 0]
        ŷ2 = [1 1 1 0; 1 1 1 0; 1 1 1 0; 1 1 1 0]
        ŷ = cat(ŷ1, ŷ2; dims=3)
        ŷ = cat(ŷ, ŷ; dims=4)
        ŷ = cat(ŷ, ŷ; dims=5)

        ŷ_dtm = euclidean_distance_transform(ŷ)
        y_dtm = euclidean_distance_transform(y)

        @test hd_lossP(ŷ, y, ŷ_dtm, y_dtm) == hd_loss(ŷ, y, ŷ_dtm, y_dtm)
    end
end

@testset ExtendedTestSet "dice_lossP" begin
    @testset ExtendedTestSet "dice_lossP" begin
        y1 = [1 1 1 0; 1 1 1 0; 1 1 1 0; 1 1 1 0]
        y2 = [1 1 1 0; 1 1 1 0; 1 1 1 0; 1 1 1 0]
        y = cat(y1, y2; dims=3)
        y = cat(y, y; dims=4)
        y = cat(y, y; dims=5)

        ŷ1 = [1 1 1 0; 1 1 1 0; 1 1 1 0; 1 1 1 0]
        ŷ2 = [1 1 1 0; 1 1 1 0; 1 1 1 0; 1 1 1 0]
        ŷ = cat(ŷ1, ŷ2; dims=3)
        ŷ = cat(ŷ, ŷ; dims=4)
        ŷ = cat(ŷ, ŷ; dims=5)

        @test dice_lossP(ŷ, y) == dice_loss(ŷ, y)
    end
end
