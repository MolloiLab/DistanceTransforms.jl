@testset ExtendedTestSet "squared euclidean" begin
    @testset ExtendedTestSet "squared euclidean 1D" begin
        f = [1, 1, 0, 0, 0, 1, 1]
        output, v, z = zeros(length(f)), ones(Int32, length(f)), ones(length(f) .+ 1)
        tfm = SquaredEuclidean()
        test = transform(boolean_indicator(f), tfm; output=output, v=v, z=z)
        answer = [0.0, 0.0, 1.0, 4.0, 1.0, 0.0, 0.0]
        @test test == answer
    end
    @testset ExtendedTestSet "squared euclidean 1D" begin
        f = [0, 0, 0, 1]
        output, v, z = zeros(length(f)), ones(Int32, length(f)), ones(length(f) .+ 1)
        tfm = SquaredEuclidean()
        test = transform(boolean_indicator(f), tfm; output=output, v=v, z=z)
        answer = [9.0, 4.0, 1.0, 0.0]
        @test test == answer
    end
    @testset ExtendedTestSet "squared euclidean 1D" begin
        f = [1, 0, 0, 0]
        output, v, z = zeros(length(f)), ones(Int32, length(f)), ones(length(f) .+ 1)
        tfm = SquaredEuclidean()
        test = transform(boolean_indicator(f), tfm; output=output, v=v, z=z)
        answer = [0, 1, 4, 9]
        @test test == answer
    end

    @testset ExtendedTestSet "squared_euclidean 2D" begin
        img = [
            0 1 1 1 0 0 0 1 1
            1 1 1 1 1 0 0 0 1
            1 0 0 0 1 0 0 1 1
            1 0 0 0 1 0 1 1 0
            1 0 0 0 1 1 0 1 0
            1 1 1 1 1 0 0 1 0
            0 1 1 1 0 0 0 0 1
        ]
        output, v, z = zeros(size(img)), ones(Int32, size(img)), ones(size(img) .+ 1)
        tfm = SquaredEuclidean()
        test = transform(boolean_indicator(img), tfm; output=output, v=v, z=z)
        answer = [
            1.0  0.0  0.0  0.0  1.0  2.0  1.0  0.0  0.0
            0.0  0.0  0.0  0.0  0.0  1.0  2.0  1.0  0.0
            0.0  1.0  1.0  1.0  0.0  1.0  1.0  0.0  0.0
            0.0  1.0  4.0  1.0  0.0  1.0  0.0  0.0  1.0
            0.0  1.0  1.0  1.0  0.0  0.0  1.0  0.0  1.0
            0.0  0.0  0.0  0.0  0.0  1.0  1.0  0.0  1.0
            1.0  0.0  0.0  0.0  1.0  2.0  2.0  1.0  0.0
        ]
        @test test == answer
    end
    @testset ExtendedTestSet "squared_euclidean 2D" begin
        img = [
            0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0	0 0 0
            0 0 0 1 1 1 0 0 0 0 0
            0 0 1 0 0 1 0 0 0 0 0
            0 0 1 0 0 1 1 1 0 0 0
            0 0 1 0 0 0 0 1 0 0 0
            0 0 1 0 0 0 0 1 0 0 0
            0 0 0 1 1 1 1 0 0 0 0	
            0 0 0 0 0 0 0 0 0 0 0	
            0 0 0 0 0 0 0 0 0 0 0
        ]
        output, v, z = zeros(size(img)), ones(Int32, size(img)), ones(size(img) .+ 1)
        tfm = SquaredEuclidean()
        test = transform(boolean_indicator(img), tfm; output=output, v=v, z=z)
        answer = [
            18.0  13.0  10.0  9.0  9.0  9.0  10.0  13.0  18.0  25.0  34.0
            13.0   8.0   5.0  4.0  4.0  4.0   5.0   8.0  13.0  20.0  25.0
             8.0   5.0   2.0  1.0  1.0  1.0   2.0   5.0  10.0  13.0  18.0
             5.0   2.0   1.0  0.0  0.0  0.0   1.0   4.0   5.0   8.0  13.0
             4.0   1.0   0.0  1.0  1.0  0.0   1.0   1.0   2.0   5.0  10.0
             4.0   1.0   0.0  1.0  1.0  0.0   0.0   0.0   1.0   4.0   9.0
             4.0   1.0   0.0  1.0  2.0  1.0   1.0   0.0   1.0   4.0   9.0
             4.0   1.0   0.0  1.0  1.0  1.0   1.0   0.0   1.0   4.0   9.0
             5.0   2.0   1.0  0.0  0.0  0.0   0.0   1.0   2.0   5.0  10.0
             8.0   5.0   2.0  1.0  1.0  1.0   1.0   2.0   5.0   8.0  13.0
            13.0   8.0   5.0  4.0  4.0  4.0   4.0   5.0   8.0  13.0  18.0
        ]
        @test test == answer
    end

    @testset ExtendedTestSet "squared_euclidean 3D" begin
        img = [
            0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0	0 0 0
            0 0 0 1 1 1 0 0 0 0 0
            0 0 1 0 0 1 0 0 0 0 0
            0 0 1 0 0 1 1 1 0 0 0
            0 0 1 0 0 0 0 1 0 0 0
            0 0 1 0 0 0 0 1 0 0 0
            0 0 0 1 1 1 1 0 0 0 0	
            0 0 0 0 0 0 0 0 0 0 0	
            0 0 0 0 0 0 0 0 0 0 0
        ]
        img_inv = @. ifelse(img == 0, 1, 0)
        vol = cat(img, img_inv, dims=3)
        container2 = []
        for i in 1:10
            push!(container2, vol)
        end
        vol_inv = cat(container2..., dims=3)
        output, v, z = zeros(size(vol_inv)), ones(Int32, size(vol_inv)), ones(size(vol_inv) .+ 1)
        tfm = SquaredEuclidean()
        test = transform(boolean_indicator(vol_inv), tfm; output=output, v=v, z=z)
        a1 = img_inv
        a2 = img
        ans = cat(a1, a2, dims=3)
        container_a = []
        for i in 1:10
            push!(container_a, ans)
        end
        answer = cat(container_a..., dims=3)
        @test test == answer
    end

    @testset ExtendedTestSet "squared_euclidean 2D in-place" begin
        img = [
            0 1 1 1 0 0 0 1 1
            1 1 1 1 1 0 0 0 1
            1 0 0 0 1 0 0 1 1
            1 0 0 0 1 0 1 1 0
            1 0 0 0 1 1 0 1 0
            1 1 1 1 1 0 0 1 0
            0 1 1 1 0 0 0 0 1
        ]
        output, v, z = zeros(size(img)), ones(Int32, size(img)), ones(size(img) .+ 1)
        tfm = SquaredEuclidean()
        test = transform!(boolean_indicator(img), tfm; output=output, v=v, z=z)
        answer = [
            1.0  0.0  0.0  0.0  1.0  2.0  1.0  0.0  0.0
            0.0  0.0  0.0  0.0  0.0  1.0  2.0  1.0  0.0
            0.0  1.0  1.0  1.0  0.0  1.0  1.0  0.0  0.0
            0.0  1.0  4.0  1.0  0.0  1.0  0.0  0.0  1.0
            0.0  1.0  1.0  1.0  0.0  0.0  1.0  0.0  1.0
            0.0  0.0  0.0  0.0  0.0  1.0  1.0  0.0  1.0
            1.0  0.0  0.0  0.0  1.0  2.0  2.0  1.0  0.0
        ]
        @test test == answer
    end
    # @testset ExtendedTestSet "squared_euclidean 2D in-place" begin
    #     img = [
    #         0 0 0 0 0 0 0 0 0 0 0
    #         0 0 0 0 0 0 0 0 0 0 0
    #         0 0 0 0 0 0 0 0	0 0 0
    #         0 0 0 1 1 1 0 0 0 0 0
    #         0 0 1 0 0 1 0 0 0 0 0
    #         0 0 1 0 0 1 1 1 0 0 0
    #         0 0 1 0 0 0 0 1 0 0 0
    #         0 0 1 0 0 0 0 1 0 0 0
    #         0 0 0 1 1 1 1 0 0 0 0	
    #         0 0 0 0 0 0 0 0 0 0 0	
    #         0 0 0 0 0 0 0 0 0 0 0
    #     ]
    #     output, v, z = zeros(size(img)), ones(Int32, size(img)), ones(size(img) .+ 1)
    #     tfm = SquaredEuclidean()
    #     test = transform!(boolean_indicator(img), tfm; output=output, v=v, z=z)
    #     answer = [
    #         18.0  13.0  10.0  9.0  9.0  9.0  10.0  13.0  18.0  25.0  34.0
    #         13.0   8.0   5.0  4.0  4.0  4.0   5.0   8.0  13.0  20.0  25.0
    #          8.0   5.0   2.0  1.0  1.0  1.0   2.0   5.0  10.0  13.0  18.0
    #          5.0   2.0   1.0  0.0  0.0  0.0   1.0   4.0   5.0   8.0  13.0
    #          4.0   1.0   0.0  1.0  1.0  0.0   1.0   1.0   2.0   5.0  10.0
    #          4.0   1.0   0.0  1.0  1.0  0.0   0.0   0.0   1.0   4.0   9.0
    #          4.0   1.0   0.0  1.0  2.0  1.0   1.0   0.0   1.0   4.0   9.0
    #          4.0   1.0   0.0  1.0  1.0  1.0   1.0   0.0   1.0   4.0   9.0
    #          5.0   2.0   1.0  0.0  0.0  0.0   0.0   1.0   2.0   5.0  10.0
    #          8.0   5.0   2.0  1.0  1.0  1.0   1.0   2.0   5.0   8.0  13.0
    #         13.0   8.0   5.0  4.0  4.0  4.0   4.0   5.0   8.0  13.0  18.0
    #     ]
    #     @test test == answer
    # end

    @testset ExtendedTestSet "squared_euclidean 3D in-place" begin
        img = [
            0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0	0 0 0
            0 0 0 1 1 1 0 0 0 0 0
            0 0 1 0 0 1 0 0 0 0 0
            0 0 1 0 0 1 1 1 0 0 0
            0 0 1 0 0 0 0 1 0 0 0
            0 0 1 0 0 0 0 1 0 0 0
            0 0 0 1 1 1 1 0 0 0 0	
            0 0 0 0 0 0 0 0 0 0 0	
            0 0 0 0 0 0 0 0 0 0 0
        ]
        img_inv = @. ifelse(img == 0, 1, 0)
        vol = cat(img, img_inv, dims=3)
        container2 = []
        for i in 1:10
            push!(container2, vol)
        end
        vol_inv = cat(container2..., dims=3)
        output, v, z = zeros(size(vol_inv)), ones(Int32, size(vol_inv)), ones(size(vol_inv) .+ 1)
        tfm = SquaredEuclidean()
        test = transform!(boolean_indicator(vol_inv), tfm; output=output, v=v, z=z)
        a1 = img_inv
        a2 = img
        ans = cat(a1, a2, dims=3)
        container_a = []
        for i in 1:10
            push!(container_a, ans)
        end
        answer = cat(container_a..., dims=3)
        @test test == answer
    end

    @testset ExtendedTestSet "squared_euclidean 2D multi-threaded" begin
    end

    @testset ExtendedTestSet "squared_euclidean 3D multi-threaded" begin
    end
end
