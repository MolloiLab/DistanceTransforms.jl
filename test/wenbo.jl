### A Pluto.jl notebook ###
# v0.19.12

using Markdown
using InteractiveUtils

# ╔═╡ 38a67f7e-3b1a-4a6c-9916-91eab4ad4d63
# ╠═╡ show_logs = false
begin
	using Pkg
	Pkg.activate("..")
	using Revise
	using PlutoUI
	using Test
	using CUDA
	using DistanceTransforms
	using FoldsThreads
end

# ╔═╡ 3b8595ae-b215-4e3b-b615-ff460793f028
TableOfContents()

# ╔═╡ 4a043231-7888-4408-a663-4541e8606245
md"""
# `Wenbo`
"""

# ╔═╡ edc7480a-11ec-4952-8984-af97d3967639
md"""
## Single-Thread
"""

# ╔═╡ ab715087-a707-485b-9463-8d17fe0bcaab
md"""
### 1D
"""

# ╔═╡ f860bf51-4c1f-4b82-a67d-d5294733345f
@testset "wenbo 1D" begin
	f = [1, 1, 0, 0, 0, 1, 1]
	tfm = Wenbo()
	test = transform(f, tfm)
	answer = [0.0, 0.0, 1.0, 4.0, 1.0, 0.0, 0.0]
	@test test == answer
end;

# ╔═╡ f6d1f13b-2b81-4253-ad80-38dd6c11eb93
@testset "wenbo 1D" begin
	f = [0, 0, 0, 1]
	tfm = Wenbo()
	test = transform(f, tfm)
	answer = [9.0, 4.0, 1.0, 0.0]
	@test test == answer
end;

# ╔═╡ 77970fcb-e584-4222-af3c-5b8482bab391
@testset "wenbo 1D" begin
	f = [1, 0, 0, 0]
	tfm = Wenbo()
	test = transform(f, tfm)
	answer = [0, 1, 4, 9]
	@test test == answer
end;

# ╔═╡ eac8edcd-ee45-4824-a3b7-cd0535260cb6
md"""
### 2D
"""

# ╔═╡ eb456533-684f-45e7-9591-7cebf493f63c
@testset "Wenbo 2D" begin
	img = [
		0 1 1 1 0 0 0 1 1
		1 1 1 1 1 0 0 0 1
		1 0 0 0 1 0 0 1 1
		1 0 0 0 1 0 1 1 0
		1 0 0 0 1 1 0 1 0
		1 1 1 1 1 0 0 1 0
		0 1 1 1 0 0 0 0 1
	]
	tfm = Wenbo()
	test = transform(img, tfm)
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
end;

# ╔═╡ 350a21d9-552f-4d0a-a586-a8932fc4e245
@testset "wenbo 2D" begin
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
	tfm = Wenbo()
	test = transform(img, tfm)
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
end;

# ╔═╡ e05ea048-d7bd-48a1-b943-2dc9b9ba5e02
md"""
### 3D
"""

# ╔═╡ b1d3b0d0-5103-4fdf-8d55-16babca9ee21
@testset "wenbo 3D" begin
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
	tfm = Wenbo()
	test = transform(vol_inv, tfm)
	a1 = img_inv
	a2 = img
	ans = cat(a1, a2, dims=3)
	container_a = []
	for i in 1:10
		push!(container_a, ans)
	end
	answer = cat(container_a..., dims=3)
	@test test == answer
end;

# ╔═╡ 8fb44a16-a4cb-4f4c-9a16-f85cacf9e81f
md"""
## Multi-Threaded
"""

# ╔═╡ cdcd1cda-be51-4a2a-a69d-381338fd5bd9
md"""
### 2D
"""

# ╔═╡ ea81cc9e-2ee9-4d62-9288-c081ef9ef0c4
@testset "wenbo 2D multi-threaded" begin
	img = [
		0 1 1 1 0 0 0 1 1
		1 1 1 1 1 0 0 0 1
		1 0 0 0 1 0 0 1 1
		1 0 0 0 1 0 1 1 0
		1 0 0 0 1 1 0 1 0
		1 1 1 1 1 0 0 1 0
		0 1 1 1 0 0 0 0 1
	]
	tfm = Wenbo()
	nthreads = Threads.nthreads()
	test = transform(img, tfm, nthreads)
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
end;

# ╔═╡ 58c3aaee-be8c-42a9-a3f7-35aafc77a308
@testset "wenbo 2D multi-threaded" begin
	img = rand([0, 1], 10, 10)
	tfm = Wenbo()
	nthreads = Threads.nthreads()
	test = transform(img, tfm, nthreads)
	answer = transform(img, tfm)
	@test test == answer
end;

# ╔═╡ 89bf91b7-2a29-48ed-8c0c-b6e46fa5de4e
md"""
### 3D
"""

# ╔═╡ bdef08b3-f15c-4769-ba7d-c3dc4f9a0006
@testset "wenbo 3D multi-threaded" begin
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
	tfm = Wenbo()
	nthreads = Threads.nthreads()
	test = transform(vol_inv, tfm, nthreads)
	a1 = img_inv
	a2 = img
	ans = cat(a1, a2, dims=3)
	container_a = []
	for i in 1:10
		push!(container_a, ans)
	end
	answer = cat(container_a..., dims=3)
	@test test == answer
end;

# ╔═╡ 51117c0a-5f23-4b5d-ab76-81d25cd73a27
@testset "wenbo 3D multi-threaded" begin
	img = rand([0, 1], 10, 10, 10)
	tfm = Wenbo()
	nthreads = Threads.nthreads()
	test = transform(img, tfm, nthreads)
	answer = transform(img, tfm)
	@test test == answer
end;

# ╔═╡ 91899f57-48cc-49e4-b100-116f2326d126
md"""
## GPU
"""

# ╔═╡ 6293532a-198d-49a2-b450-3ab7c7be3227
md"""
### 2D
"""

# ╔═╡ edc19bce-db97-4a0d-a4a9-11e9241434e9
if CUDA.has_cuda_gpu()
	ks = DistanceTransforms.get_GPU_kernels(Wenbo())
	@testset "wenbo 2D GPU" begin
		img = [
			0 1 1 1 0 0 0 1 1
			1 1 1 1 1 0 0 0 1
			1 0 0 0 1 0 0 1 1
			1 0 0 0 1 0 1 1 0
			1 0 0 0 1 1 0 1 0
			1 1 1 1 1 0 0 1 0
			0 1 1 1 0 0 0 0 1
		]
		tfm = Wenbo()
		test = transform(CuArray(img), tfm, ks)
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
	end
else
	@warn "CUDA unavailable, not testing GPU support"
end;

# ╔═╡ 9d3e6bab-d8a5-42d3-862b-135f10887562
if CUDA.has_cuda_gpu()
	@testset "wenbo 2D GPU" begin
		img = rand([0, 1], 10, 10)
		img2 = copy(img)
		tfm = Wenbo()
		test = transform(CUDA.CuArray(img), tfm, ks)
		answer = transform(img2, tfm)
		@test test == CuArray(answer)
	end
else
	@warn "CUDA unavailable, not testing GPU support"

end;

# ╔═╡ 35ff78e2-b1d7-4792-8145-ced70a6ff233
md"""
### 3D
"""

# ╔═╡ f5159fb2-30bc-4828-bb21-d9ad7343fe85
if CUDA.has_cuda_gpu()
	@testset "wenbo 3D GPU" begin
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
		tfm = Wenbo()
		test = transform(vol_inv, tfm, ks)
		a1 = img_inv
		a2 = img
		ans = cat(a1, a2, dims=3)
		container_a = []
		for i in 1:10
			push!(container_a, ans)
		end
		answer = cat(container_a..., dims=3)
		@test test == CuArray(answer)
	end
else
	@warn "CUDA unavailable, not testing GPU support"
end;

# ╔═╡ 4fa62723-c3e2-4afd-911b-e3f2285df74d
if CUDA.has_cuda_gpu()
	@testset "Wenbo 3D GPU" begin
		img = rand([0, 1], 10, 10, 10)
		img2 = copy(img)
		tfm = Wenbo()
		test = transform(CUDA.CuArray(img), tfm, ks)
		answer = transform(img2, tfm)
		@test test == CuArray(answer)
	end
else
	@warn "CUDA unavailable, not testing GPU support"

end;

# ╔═╡ 9755e49a-699d-4ebf-ab28-958007d1a0ab
md"""
## Various Multi-Threading
"""

# ╔═╡ 278053f3-de28-4090-add3-c08257aff74f
md"""
### 2D
"""

# ╔═╡ f87ded07-5767-4a4c-8d00-d9fa5d6914bc
@testset "wenbo 2D FoldsThreads " begin
	img = [
		0 1 1 1 0 0 0 1 1
		1 1 1 1 1 0 0 0 1
		1 0 0 0 1 0 0 1 1
		1 0 0 0 1 0 1 1 0
		1 0 0 0 1 1 0 1 0
		1 1 1 1 1 0 0 1 0
		0 1 1 1 0 0 0 0 1
	]
	tfm = Wenbo()
	ex = DepthFirstEx()
	test = transform(img, tfm, ex)
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
end;

# ╔═╡ c52fd17a-ad54-4317-96ac-a6c7b0da4f99
@testset "wenbo 2D FoldsThreads " begin
	img = rand([0, 1], 10, 10)
	tfm = Wenbo()
	ex = DepthFirstEx()
	test = transform(img, tfm, ex)
	answer = transform(img, tfm)
	@test test == answer
end;

# ╔═╡ 4a17f216-e186-4352-8c18-5fdb345add80
@testset "wenbo 2D FoldsThreads " begin
	img = [
		0 1 1 1 0 0 0 1 1
		1 1 1 1 1 0 0 0 1
		1 0 0 0 1 0 0 1 1
		1 0 0 0 1 0 1 1 0
		1 0 0 0 1 1 0 1 0
		1 1 1 1 1 0 0 1 0
		0 1 1 1 0 0 0 0 1
	]
	tfm = Wenbo()
	ex = NonThreadedEx()
	test = transform(img, tfm, ex)
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
end;

# ╔═╡ b6d801fa-2097-43e7-887e-b5dad19bc3f8
@testset "wenbo 2D FoldsThreads " begin
	img = rand([0, 1], 10, 10)
	tfm = Wenbo()
	ex = NonThreadedEx()
	test = transform(img, tfm, ex)
	answer = transform(img, tfm)
	@test test == answer
end;

# ╔═╡ 07d8dd61-39ff-40e6-886c-410ae651dc39
@testset "wenbo 2D FoldsThreads " begin
	img = [
		0 1 1 1 0 0 0 1 1
		1 1 1 1 1 0 0 0 1
		1 0 0 0 1 0 0 1 1
		1 0 0 0 1 0 1 1 0
		1 0 0 0 1 1 0 1 0
		1 1 1 1 1 0 0 1 0
		0 1 1 1 0 0 0 0 1
	]
	tfm = Wenbo()
	ex = WorkStealingEx()
	test = transform(img, tfm, ex)
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
end;

# ╔═╡ eccdfa48-2c8d-45a3-81b0-55325020f429
@testset "wenbo FoldsThreads" begin
	img = rand([0, 1], 10, 10)
	tfm = Wenbo()
	ex = WorkStealingEx()
	test = transform(img, tfm, ex)
	answer = transform(img, tfm)
	@test test == answer
end;

# ╔═╡ Cell order:
# ╠═38a67f7e-3b1a-4a6c-9916-91eab4ad4d63
# ╠═3b8595ae-b215-4e3b-b615-ff460793f028
# ╟─4a043231-7888-4408-a663-4541e8606245
# ╠═edc7480a-11ec-4952-8984-af97d3967639
# ╟─ab715087-a707-485b-9463-8d17fe0bcaab
# ╠═f860bf51-4c1f-4b82-a67d-d5294733345f
# ╠═f6d1f13b-2b81-4253-ad80-38dd6c11eb93
# ╠═77970fcb-e584-4222-af3c-5b8482bab391
# ╟─eac8edcd-ee45-4824-a3b7-cd0535260cb6
# ╠═eb456533-684f-45e7-9591-7cebf493f63c
# ╠═350a21d9-552f-4d0a-a586-a8932fc4e245
# ╟─e05ea048-d7bd-48a1-b943-2dc9b9ba5e02
# ╠═b1d3b0d0-5103-4fdf-8d55-16babca9ee21
# ╟─8fb44a16-a4cb-4f4c-9a16-f85cacf9e81f
# ╟─cdcd1cda-be51-4a2a-a69d-381338fd5bd9
# ╠═ea81cc9e-2ee9-4d62-9288-c081ef9ef0c4
# ╠═58c3aaee-be8c-42a9-a3f7-35aafc77a308
# ╟─89bf91b7-2a29-48ed-8c0c-b6e46fa5de4e
# ╠═bdef08b3-f15c-4769-ba7d-c3dc4f9a0006
# ╠═51117c0a-5f23-4b5d-ab76-81d25cd73a27
# ╟─91899f57-48cc-49e4-b100-116f2326d126
# ╟─6293532a-198d-49a2-b450-3ab7c7be3227
# ╠═edc19bce-db97-4a0d-a4a9-11e9241434e9
# ╠═9d3e6bab-d8a5-42d3-862b-135f10887562
# ╟─35ff78e2-b1d7-4792-8145-ced70a6ff233
# ╠═f5159fb2-30bc-4828-bb21-d9ad7343fe85
# ╠═4fa62723-c3e2-4afd-911b-e3f2285df74d
# ╟─9755e49a-699d-4ebf-ab28-958007d1a0ab
# ╟─278053f3-de28-4090-add3-c08257aff74f
# ╠═f87ded07-5767-4a4c-8d00-d9fa5d6914bc
# ╠═c52fd17a-ad54-4317-96ac-a6c7b0da4f99
# ╠═4a17f216-e186-4352-8c18-5fdb345add80
# ╠═b6d801fa-2097-43e7-887e-b5dad19bc3f8
# ╠═07d8dd61-39ff-40e6-886c-410ae651dc39
# ╠═eccdfa48-2c8d-45a3-81b0-55325020f429
