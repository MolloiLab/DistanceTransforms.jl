
@testset "Felzenszwalb 1D" begin
	f = [1, 1, 0, 0, 0, 1, 1]
	output, v, z = zeros(length(f)), ones(Int32, length(f)), ones(length(f) .+ 1)
	tfm = Felzenszwalb()
	test = transform(boolean_indicator(f), tfm; output=output, v=v, z=z)
	answer = [0.0, 0.0, 1.0, 4.0, 1.0, 0.0, 0.0]
	@test test == answer

	f = [0, 0, 0, 1]
	output, v, z = zeros(length(f)), ones(Int32, length(f)), ones(length(f) .+ 1)
	tfm = Felzenszwalb()
	test = transform(boolean_indicator(f), tfm; output=output, v=v, z=z)
	answer = [9.0, 4.0, 1.0, 0.0]
	@test test == answer

	f = [1, 0, 0, 0]
	output, v, z = zeros(length(f)), ones(Int32, length(f)), ones(length(f) .+ 1)
	tfm = Felzenszwalb()
	test = transform(boolean_indicator(f), tfm; output=output, v=v, z=z)
	answer = [0, 1, 4, 9]
	@test test == answer
end

@testset "Felzenszwalb 2D" begin
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
	tfm = Felzenszwalb()
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
	tfm = Felzenszwalb()
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

	for i = 1 : 270
		img = Bool.(rand([0, 1], 512, 512))
		output, v, z = zeros(size(img)), ones(Int32, size(img)), ones(size(img) .+ 1)
		tfm = Felzenszwalb()
		test = transform(boolean_indicator(img), tfm; output=output, v=v, z=z)
		answer = round.(transform(img, Maurer()) .^ 2)
		@test test == answer
	end
end


@testset "Felzenszwalb 3D" begin
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
	tfm = Felzenszwalb()
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

	for i = 1 : 55
		img = Bool.(rand([0, 1], 96, 96, 96))
		output, v, z = zeros(size(img)), ones(Int32, size(img)), ones(size(img) .+ 1)
		tfm = Felzenszwalb()
		test = transform(boolean_indicator(img), tfm; output=output, v=v, z=z)
		answer = round.(transform(img, Maurer()) .^ 2)
		@test test == answer
	end
end


@testset "Felzenszwalb 2D multi-threaded" begin
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
	tfm = Felzenszwalb()
	nthreads = Threads.nthreads()
	test = transform(boolean_indicator(img), tfm, nthreads; output=output, v=v, z=z)
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

	img = rand([0, 1], 10, 10)
	output, v, z = zeros(size(img)), ones(Int32, size(img)), ones(size(img) .+ 1)
	tfm = Felzenszwalb()
	nthreads = Threads.nthreads()
	test = transform(boolean_indicator(img), tfm, nthreads; output=output, v=v, z=z)
	answer = round.(transform(img, Maurer()) .^ 2)
	@test test == answer

	nthreads = Threads.nthreads()
	for i = 1 : 500
		img = Bool.(rand([0, 1], 512, 512))
		output, v, z = zeros(size(img)), ones(Int32, size(img)), ones(size(img) .+ 1)
		tfm = Felzenszwalb()
		test = transform(boolean_indicator(img), tfm, nthreads; output=output, v=v, z=z)
		answer = round.(transform(img, Maurer()) .^ 2)
		@test test == answer
	end
end


@testset "Felzenszwalb 3D multi-threaded" begin
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
	tfm = Felzenszwalb()
	nthreads = Threads.nthreads()
	test = transform(boolean_indicator(vol_inv), tfm, nthreads; output=output, v=v, z=z)
	a1 = img_inv
	a2 = img
	ans = cat(a1, a2, dims=3)
	container_a = []
	for i in 1:10
		push!(container_a, ans)
	end
	answer = cat(container_a..., dims=3)
	@test test == answer

	img = rand([0, 1], 10, 10, 10)
	output, v, z = zeros(size(img)), ones(Int32, size(img)), ones(size(img) .+ 1)
	tfm = Felzenszwalb()
	nthreads = Threads.nthreads()
	test = transform(boolean_indicator(img), tfm, nthreads; output=output, v=v, z=z)
	answer = round.(transform(img, Maurer()) .^ 2)
	@test test == answer

	nthreads = Threads.nthreads()
	for i = 1 : 110
		img = Bool.(rand([0, 1], 96, 96, 96))
		output, v, z = zeros(size(img)), ones(Int32, size(img)), ones(size(img) .+ 1)
		tfm = Felzenszwalb()
		test = transform(boolean_indicator(img), tfm, nthreads; output=output, v=v, z=z)
		answer = round.(transform(img, Maurer()) .^ 2)
		@test test == answer
	end
end

if CUDA.has_cuda_gpu()
	@testset "Felzenszwalb 2D GPU" begin
		img = [
			0 1 1 1 0 0 0 1 1
			1 1 1 1 1 0 0 0 1
			1 0 0 0 1 0 0 1 1
			1 0 0 0 1 0 1 1 0
			1 0 0 0 1 1 0 1 0
			1 1 1 1 1 0 0 1 0
			0 1 1 1 0 0 0 0 1
		]
		output, v, z = CUDA.zeros(size(img)), CUDA.ones(Int32, size(img)), CUDA.ones(size(img) .+ 1)
		tfm = Felzenszwalb()
		test = transform(CUDA.CuArray(boolean_indicator(img)), tfm; output=output, v=v, z=z)
		answer = CuArray([
			1.0  0.0  0.0  0.0  1.0  2.0  1.0  0.0  0.0
			0.0  0.0  0.0  0.0  0.0  1.0  2.0  1.0  0.0
			0.0  1.0  1.0  1.0  0.0  1.0  1.0  0.0  0.0
			0.0  1.0  4.0  1.0  0.0  1.0  0.0  0.0  1.0
			0.0  1.0  1.0  1.0  0.0  0.0  1.0  0.0  1.0
			0.0  0.0  0.0  0.0  0.0  1.0  1.0  0.0  1.0
			1.0  0.0  0.0  0.0  1.0  2.0  2.0  1.0  0.0
		])
		@test test == answer

		img = rand([0, 1], 10, 10)
		img2 = copy(img)
		output, v, z = CUDA.zeros(size(img)), CUDA.ones(Int32, size(img)), CUDA.ones(size(img) .+ 1)
		output2, v2, z2 = zeros(size(img2)), ones(Int32, size(img2)), ones(size(img2) .+ 1)
		tfm = Felzenszwalb()
		test = transform(CUDA.CuArray(boolean_indicator(img)), tfm; output=output, v=v, z=z)
		answer = transform(boolean_indicator(img2), tfm; output=output2, v=v2, z=z2)
		@test test == CuArray(answer)
	end
	@testset "Felzenszwalb 3D multi-threaded" begin
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
		vol_inv = CuArray(cat(container2..., dims=3))
		output, v, z = CUDA.zeros(size(vol_inv)), CUDA.ones(Int32, size(vol_inv)), CUDA.ones(size(vol_inv) .+ 1)
		tfm = Felzenszwalb()
		test = transform(boolean_indicator(vol_inv), tfm; output=output, v=v, z=z)
		a1 = img_inv
		a2 = img
		ans = cat(a1, a2, dims=3)
		container_a = []
		for i in 1:10
			push!(container_a, ans)
		end
		answer = cat(container_a..., dims=3)
		@test test == CuArray(answer)

		img = rand([0, 1], 10, 10, 10)
		tfm = Felzenszwalb()
		# test = transform(CUDA.CuArray(boolean_indicator(img)), tfm; output=output, v=v, z=z)
		test = transform(CUDA.CuArray(boolean_indicator(img)), tfm)
		answer = round.(transform(img, Maurer()) .^ 2)
		@test test == CuArray(answer)
	end
else
	@warn "CUDA unavailable, not testing GPU support"
end
