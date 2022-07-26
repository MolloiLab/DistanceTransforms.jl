@testset ExtendedTestSet "Wenbo" begin
    @testset ExtendedTestSet "Wenbo 1D" begin
        f = [1, 1, 0, 0, 0, 1, 1]
        arg1, arg2, arg3 = zeros(length(f)), 1, 1
        tfm = Wenbo()
        test = transform(boolean_indicator(f), tfm; output=arg1, pointerA=arg2, pointerB=arg3)
        answer = [0.0, 0.0, 1.0, 4.0, 1.0, 0.0, 0.0]
        @test test == answer
    end

    @testset ExtendedTestSet "Wenbo 1D" begin
        f = [0, 0, 0, 1]
        arg1, arg2, arg3 = zeros(length(f)), 1, 1
        tfm = Wenbo()
        test = transform(boolean_indicator(f), tfm; output=arg1, pointerA=arg2, pointerB=arg3)
        answer = [9.0, 4.0, 1.0, 0.0]
        @test test == answer
    end

    @testset ExtendedTestSet "Wenbo 1D" begin
        f = [1, 0, 0, 0]
        arg1, arg2, arg3 = zeros(length(f)), 1, 1
        tfm = Wenbo()
        test = transform(boolean_indicator(f), tfm; output=arg1, pointerA=arg2, pointerB=arg3)
        answer = [0.0, 1.0, 4.0, 9.0]
        @test test == answer
    end

    @testset ExtendedTestSet "Wenbo 2D" begin
        img = [
            0 1 1 1 0 0 0 1 1
            1 1 1 1 1 0 0 0 1
            1 0 0 0 1 0 0 1 1
            1 0 0 0 1 0 1 1 0
            1 0 0 0 1 1 0 1 0
            1 1 1 1 1 0 0 1 0
            0 1 1 1 0 0 0 0 1
        ]
        output, pointerA, pointerB = zeros(size(img)), 1, 1
        tfm = Wenbo()
        test = transform(boolean_indicator(img), tfm; output=output, pointerA=pointerA, pointerB=pointerB)
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

    @testset ExtendedTestSet "Wenbo 2D" begin
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
        output, pointerA, pointerB = zeros(size(img)), 1, 1
        tfm = Wenbo()
        test = transform(boolean_indicator(img), tfm; output=output, pointerA=pointerA, pointerB=pointerB)
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

    @testset ExtendedTestSet "Wenbo 3D" begin
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
        output, pointerA, pointerB = zeros(size(vol_inv)), 1, 1
        tfm = Wenbo()
        test = transform(boolean_indicator(vol_inv), tfm; output=output, pointerA=pointerA, pointerB=pointerB)
        a1 = [
            1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0
            1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0
            1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0
            1.0  1.0  1.0  0.0  0.0  0.0  1.0  1.0  1.0  1.0  1.0
            1.0  1.0  0.0  1.0  1.0  0.0  1.0  1.0  1.0  1.0  1.0
            1.0  1.0  0.0  1.0  1.0  0.0  0.0  0.0  1.0  1.0  1.0
            1.0  1.0  0.0  1.0  1.0  1.0  1.0  0.0  1.0  1.0  1.0
            1.0  1.0  0.0  1.0  1.0  1.0  1.0  0.0  1.0  1.0  1.0
            1.0  1.0  1.0  0.0  0.0  0.0  0.0  1.0  1.0  1.0  1.0
            1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0
            1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0
       ]
       a2 = [
            0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
            0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
            0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
            0.0  0.0  0.0  1.0  1.0  1.0  0.0  0.0  0.0  0.0  0.0
            0.0  0.0  1.0  0.0  0.0  1.0  0.0  0.0  0.0  0.0  0.0
            0.0  0.0  1.0  0.0  0.0  1.0  1.0  1.0  0.0  0.0  0.0
            0.0  0.0  1.0  0.0  0.0  0.0  0.0  1.0  0.0  0.0  0.0
            0.0  0.0  1.0  0.0  0.0  0.0  0.0  1.0  0.0  0.0  0.0
            0.0  0.0  0.0  1.0  1.0  1.0  1.0  0.0  0.0  0.0  0.0
            0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
            0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
       ]
        ans = cat(a1, a2, dims=3)
        container_a = []
        for i in 1:10
            push!(container_a, ans)
        end
        answer = cat(container_a..., dims=3)
        @test test == answer
    end
end
