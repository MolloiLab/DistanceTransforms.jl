using ImageMorphology: distance_transform, feature_transform

test_transform(img) = distance_transform(feature_transform(Bool.(img)))

@testset "transform!" begin
    @testset "1D transform!" begin
        for n in [10, 50, 100]
            for test_idx in 1:20
                img = rand([0.0f0, 1.0f0], n)

                img_bool = boolean_indicator(img)
                output = similar(img, Float32)
                v = ones(Int32, size(img))
                z = ones(Float32, size(img) .+ 1)

                transform!(img_bool, output, v, z)
                img_test = test_transform(img) .^ 2

                @test Array(output) ≈ img_test
            end
        end
    end

    @testset "2D transform!" begin
        for n in [10, 50, 100]
            for test_idx in 1:20
                img = rand([0.0f0, 1.0f0], n, n)

                img_bool = boolean_indicator(img)
                output = similar(img, Float32)
                v = ones(Int32, size(img))
                z = ones(Float32, size(img) .+ 1)

                transform!(img_bool, output, v, z)
                img_test = test_transform(img) .^ 2

                @test Array(output) ≈ img_test
            end

            # non-threaded
            for test_idx in 1:20
                img = rand([0.0f0, 1.0f0], n, n)

                img_bool = boolean_indicator(img)
                output = similar(img, Float32)
                v = ones(Int32, size(img))
                z = ones(Float32, size(img) .+ 1)

                transform!(img_bool, output, v, z; threaded=false)
                img_test = test_transform(img) .^ 2

                @test Array(output) ≈ img_test
            end
        end
    end

    @testset "3D transform!" begin
        for n in [10, 100]
            for test_idx in 1:5
                img = rand([0.0f0, 1.0f0], n, n, n)

                img_bool = boolean_indicator(img)
                output = similar(img, Float32)
                v = ones(Int32, size(img))
                z = ones(Float32, size(img) .+ 1)

                transform!(img_bool, output, v, z)
                img_test = test_transform(img) .^ 2

                @test Array(output) ≈ img_test
            end

            # non-threaded
            for test_idx in 1:5
                img = rand([0.0f0, 1.0f0], n, n, n)

                img_bool = boolean_indicator(img)
                output = similar(img, Float32)
                v = ones(Int32, size(img))
                z = ones(Float32, size(img) .+ 1)

                transform!(img_bool, output, v, z; threaded=false)
                img_test = test_transform(img) .^ 2

                @test Array(output) ≈ img_test
            end
        end
    end

    if dev != Array
        @testset "2D GPU transform!" begin
            for n in [10, 50, 100]
                for test_idx in 1:20
                    img = dev(rand([0.0f0, 1.0f0], n, n))

                    img_bool = boolean_indicator(img)
                    output = similar(img, Float32)
                    v = dev(ones(Int32, size(img)))
                    z = dev(ones(Float32, size(img) .+ 1))

                    transform!(img_bool, output, v, z)
                    img_test = test_transform(Array(img)) .^ 2

                    @test Array(output) ≈ img_test
                end
            end
        end
        @testset "3D GPU transform!" begin
            for n in [10, 50, 100]
                for test_idx in 1:20
                    img = dev(rand([0.0f0, 1.0f0], n, n, n))

                    img_bool = boolean_indicator(img)
                    output = similar(img, Float32)
                    v = dev(ones(Int32, size(img)))
                    z = dev(ones(Float32, size(img) .+ 1))

                    transform!(img_bool, output, v, z)
                    img_test = test_transform(Array(img)) .^ 2

                    @test Array(output) ≈ img_test
                end
            end
        end
    else
        @info "No GPU available, skipping tests"
    end
end

@testset "transform" begin
    @testset "1D transform" begin
        for n in [10, 50, 100]
            for test_idx in 1:20
                img = rand([0.0f0, 1.0f0], n)
                img_bool = boolean_indicator(img)
                output = transform(img_bool)
                img_test = test_transform(img) .^ 2
                @test Array(output) ≈ img_test
            end
        end
    end

    @testset "2D transform" begin
        for n in [10, 50, 100]
            for test_idx in 1:20
                img = rand([0.0f0, 1.0f0], n, n)
                img_bool = boolean_indicator(img)
                output = transform(img_bool)
                img_test = test_transform(img) .^ 2
                @test Array(output) ≈ img_test
            end

            # non-threaded
            for test_idx in 1:20
                img = rand([0.0f0, 1.0f0], n, n)
                img_bool = boolean_indicator(img)
                output = transform(img_bool; threaded=false)
                img_test = test_transform(img) .^ 2
                @test Array(output) ≈ img_test
            end
        end
    end

    @testset "3D transform" begin
        for n in [10, 100]
            for test_idx in 1:5
                img = rand([0.0f0, 1.0f0], n, n, n)
                img_bool = boolean_indicator(img)
                output = transform(img_bool)
                img_test = test_transform(img) .^ 2
                @test Array(output) ≈ img_test
            end

            # non-threaded
            for test_idx in 1:5
                img = rand([0.0f0, 1.0f0], n, n, n)
                img_bool = boolean_indicator(img)
                output = transform(img_bool; threaded=false)
                img_test = test_transform(img) .^ 2
                @test Array(output) ≈ img_test
            end
        end
    end

    if dev != Array
        @testset "2D GPU transform" begin
            for n in [10, 50, 100]
                for test_idx in 1:20
                    img = dev(rand([0.0f0, 1.0f0], n, n))
                    img_bool = boolean_indicator(img)
                    output = transform(img_bool)
                    img_test = test_transform(Array(img)) .^ 2
                    @test Array(output) ≈ img_test
                end
            end
        end

        @testset "3D GPU transform" begin
            for n in [10, 50, 100]
                for test_idx in 1:20
                    img = dev(rand([0.0f0, 1.0f0], n, n, n))
                    img_bool = boolean_indicator(img)
                    output = transform(img_bool)
                    img_test = test_transform(Array(img)) .^ 2
                    @test Array(output) ≈ img_test
                end
            end
        end
    else
        @info "No GPU available, skipping tests"
    end
end
