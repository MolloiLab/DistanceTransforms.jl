using ImageMorphology: distance_transform, feature_transform

test_transform(img) = distance_transform(feature_transform(Bool.(img)))

@testset "1D" begin
	ns = collect(10:10:100)
	for n in ns
		img = rand([0.0f0, 1.0f0], n)
		img_bool = boolean_indicator(img)
		output = transform(img_bool)
		img_test = test_transform(img) .^ 2
		@test Array(output) ≈ img_test
	end
end

@testset "2D" begin
	ns = collect(10:10:100)
	for n in ns
		img = rand([0.0f0, 1.0f0], n, n)
		img_bool = boolean_indicator(img)
		output = transform(img_bool)
		img_test = test_transform(img) .^ 2
		@test Array(output) ≈ img_test
	end
	for n in ns
		img = rand([0.0f0, 1.0f0], n, n)
		img_bool = boolean_indicator(img)
		output = transform(img_bool; threaded = false)
		img_test = test_transform(img) .^ 2
		@test Array(output) ≈ img_test
	end
end

@testset "3D" begin
	ns = collect(10:10:100)
	for n in ns
		img = rand([0.0f0, 1.0f0], n, n, n)
		img_bool = boolean_indicator(img)
		output = transform(img_bool)
		img_test = test_transform(img) .^ 2
		@test Array(output) ≈ img_test
	end
	for n in ns
		img = rand([0.0f0, 1.0f0], n, n, n)
		img_bool = boolean_indicator(img)
		output = transform(img_bool; threaded = false)
		img_test = test_transform(img) .^ 2
		@test Array(output) ≈ img_test
	end
end

@testset "2D GPU" begin
	ns = collect(10:10:100)
	for n in ns
		img = rand([0.0f0, 1.0f0], n, n)
		img_gpu = dev(img)
		output = transform(img_gpu)
		img_test = test_transform(img) .^ 2
		@test Array(output) ≈ img_test
	end
end

@testset "3D GPU" begin
	ns = collect(10:10:100)
	for n in ns
		img = rand([0.0f0, 1.0f0], n, n, n)
		img_gpu = dev(img)
		output = transform(img_gpu)
		img_test = test_transform(img) .^ 2
		@test Array(output) ≈ img_test
	end
end
