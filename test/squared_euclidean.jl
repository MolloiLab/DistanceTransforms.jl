### A Pluto.jl notebook ###
# v0.19.11

using Markdown
using InteractiveUtils

# ╔═╡ cd8bf944-2329-11ed-208f-1b2e91673a5e
# ╠═╡ show_logs = false
begin
	using Pkg
	Pkg.activate("..")
	using Revise
	using PlutoUI
	using Test
	using CUDA
	using DistanceTransforms
end

# ╔═╡ 373a8802-bbac-4ab5-abe4-420ccbb61ea0
TableOfContents()

# ╔═╡ 4a9d1a91-d2ac-4c30-bc2f-f2dd0e27cc43
md"""
# `SquaredEuclidean`
"""

# ╔═╡ 9f2edcc0-4f04-4881-9e83-25f92bc42119
md"""
## Regular
"""

# ╔═╡ 81f78cda-0efa-4a74-bf4c-a39197d9b73f
md"""
### 1D
"""

# ╔═╡ d3f4223e-f697-4a31-8825-66b1fae2b4f5
@testset "squared euclidean 1D" begin
	f = [1, 1, 0, 0, 0, 1, 1]
	output, v, z = zeros(length(f)), ones(Int32, length(f)), ones(length(f) .+ 1)
	tfm = SquaredEuclidean()
	test = transform(boolean_indicator(f), tfm; output=output, v=v, z=z)
	answer = [0.0, 0.0, 1.0, 4.0, 1.0, 0.0, 0.0]
	@test test == answer
end;

# ╔═╡ d48cc7be-b769-4a64-970e-5982105fd382
@testset "squared euclidean 1D" begin
	f = [0, 0, 0, 1]
	output, v, z = zeros(length(f)), ones(Int32, length(f)), ones(length(f) .+ 1)
	tfm = SquaredEuclidean()
	test = transform(boolean_indicator(f), tfm; output=output, v=v, z=z)
	answer = [9.0, 4.0, 1.0, 0.0]
	@test test == answer
end;

# ╔═╡ 4ded097b-5e18-47af-bbbb-aa9fb626a9d1
@testset "squared euclidean 1D" begin
	f = [1, 0, 0, 0]
	output, v, z = zeros(length(f)), ones(Int32, length(f)), ones(length(f) .+ 1)
	tfm = SquaredEuclidean()
	test = transform(boolean_indicator(f), tfm; output=output, v=v, z=z)
	answer = [0, 1, 4, 9]
	@test test == answer
end;

# ╔═╡ 3b1fcab4-ca39-40c4-8cfa-410acfd6b006
md"""
### 2D
"""

# ╔═╡ eafcd9e8-d691-4ae5-9e71-e999f8b6702c
@testset "squared euclidean 2D" begin
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
end;

# ╔═╡ 2f5755cb-e575-4ca9-a8b3-f2fd7846c068
@testset "squared euclidean 2D" begin
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
end;

# ╔═╡ 755aa045-73cd-425c-8057-50e65c342716
md"""
### 3D
"""

# ╔═╡ afc433aa-ecc9-443a-bbd4-55bd48730937
@testset "squared euclidean 3D" begin
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
end;

# ╔═╡ 33980a16-c38d-4ff8-afdc-bfa8b2dcb482
md"""
## In-Place
"""

# ╔═╡ bf72b4f5-5e6e-4e2f-b1da-45e389815a60
md"""
### 2D!
"""

# ╔═╡ fcab8ebd-6799-4ef8-90f8-959af5bfb375
@testset "squared euclidean 2D in-place" begin
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
end;

# ╔═╡ bb7361e0-5a20-4454-87bf-a3cbf568a94b
@testset "squared euclidean 2D in-place" begin
	img = rand([0, 1], 10, 10)
	output, v, z = zeros(size(img)), ones(Int32, size(img)), ones(size(img) .+ 1)
	tfm = SquaredEuclidean()
	test = transform!(boolean_indicator(img), tfm; output=output, v=v, z=z)
	answer = transform(boolean_indicator(img), tfm; output=output, v=v, z=z)
	@test test == answer
end;

# ╔═╡ 17eb24ab-6751-4b43-aa16-e779ae10e676
md"""
### 3D!
"""

# ╔═╡ 4c066e7c-4f1d-4795-bfb7-525b999e2a53
@testset "squared euclidean 3D in-place" begin
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
end;

# ╔═╡ ddfaf97d-052f-40b5-8e02-9cbaa0f3a801
@testset "squared euclidean 3D in-place" begin
	img = rand([0, 1], 10, 10, 10)
	output, v, z = zeros(size(img)), ones(Int32, size(img)), ones(size(img) .+ 1)
	tfm = SquaredEuclidean()
	test = transform!(boolean_indicator(img), tfm; output=output, v=v, z=z)
	answer = transform(boolean_indicator(img), tfm; output=output, v=v, z=z)
	@test test == answer
end;

# ╔═╡ 8efd0196-a621-46e9-9ff3-3f4801820ee4
md"""
## Multi-Threaded
"""

# ╔═╡ d8dc96f7-b61b-4021-8094-bd70a6b66d5e
md"""
### 2D!
"""

# ╔═╡ bf7df452-38b7-4937-a981-70cdd820622a
@testset "squared euclidean 2D multi-threaded" begin
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
	nthreads = Threads.nthreads()
	test = transform!(boolean_indicator(img), tfm, nthreads; output=output, v=v, z=z)
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

# ╔═╡ 91c1a3a4-d2cd-4bd3-8994-2efb501aa3e8
@testset "squared euclidean 2D multi-threaded" begin
	img = rand([0, 1], 10, 10)
	output, v, z = zeros(size(img)), ones(Int32, size(img)), ones(size(img) .+ 1)
	tfm = SquaredEuclidean()
	nthreads = Threads.nthreads()
	test = transform!(boolean_indicator(img), tfm, nthreads; output=output, v=v, z=z)
	answer = transform(boolean_indicator(img), tfm; output=output, v=v, z=z)
	@test test == answer
end;

# ╔═╡ 3b5f2010-48f7-4332-8742-60c71c4a01dd
md"""
### 3D!
"""

# ╔═╡ 0ccd566f-18b7-4f99-93db-8df1379e3cd1
@testset "squared euclidean 3D multi-threaded" begin
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
	nthreads = Threads.nthreads()
	test = transform!(boolean_indicator(vol_inv), tfm, nthreads; output=output, v=v, z=z)
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

# ╔═╡ ad76fe0e-c7f4-4e5e-9a5f-11f597274411
@testset "squared euclidean 3D multi-threaded" begin
	img = rand([0, 1], 10, 10, 10)
	output, v, z = zeros(size(img)), ones(Int32, size(img)), ones(size(img) .+ 1)
	tfm = SquaredEuclidean()
	nthreads = Threads.nthreads()
	test = transform!(boolean_indicator(img), tfm, nthreads; output=output, v=v, z=z)
	answer = transform(boolean_indicator(img), tfm; output=output, v=v, z=z)
	@test test == answer
end;

# ╔═╡ 7393ca78-681c-4fc2-b376-c5279812b12d
md"""
## GPU
"""

# ╔═╡ d1fc47ea-920f-48f2-8fec-bcdfcae5d04c
md"""
### 2D!
"""

# ╔═╡ e7b6805a-6362-49a6-8124-4baa09e44937
if CUDA.functional()
	@testset "squared euclidean 2D GPU" begin
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
		tfm = SquaredEuclidean()
		test = transform!(CUDA.CuArray(boolean_indicator(img)), tfm; output=output, v=v, z=z)
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

# ╔═╡ ab075b7b-4634-491d-b68b-9e3399dd814e
if CUDA.functional()
	@testset "squared euclidean 2D gpu" begin
		img = rand([0, 1], 10, 10)
		img2 = copy(img)
		output, v, z = CUDA.zeros(size(img)), CUDA.ones(Int32, size(img)), CUDA.ones(size(img) .+ 1)
		output2, v2, z2 = zeros(size(img2)), ones(Int32, size(img2)), ones(size(img2) .+ 1)
		tfm = SquaredEuclidean()
		test = transform!(CUDA.CuArray(boolean_indicator(img)), tfm; output=output, v=v, z=z)
		answer = transform(boolean_indicator(img2), tfm; output=output2, v=v2, z=z2)
		@test test == CuArray(answer)
	end
else
	@warn "CUDA unavailable, not testing GPU support"

end;

# ╔═╡ b5ba030e-9d5e-4ce5-b138-f3be7fea0e23
md"""
### 3D!
"""

# ╔═╡ 223acdc8-b5de-4eb1-9c80-19c3281e0dfd
# if CUDA.functional()
# 	@testset "squared euclidean 3D multi-threaded" begin
# 		img = [
# 			0 0 0 0 0 0 0 0 0 0 0
# 			0 0 0 0 0 0 0 0 0 0 0
# 			0 0 0 0 0 0 0 0	0 0 0
# 			0 0 0 1 1 1 0 0 0 0 0
# 			0 0 1 0 0 1 0 0 0 0 0
# 			0 0 1 0 0 1 1 1 0 0 0
# 			0 0 1 0 0 0 0 1 0 0 0
# 			0 0 1 0 0 0 0 1 0 0 0
# 			0 0 0 1 1 1 1 0 0 0 0	
# 			0 0 0 0 0 0 0 0 0 0 0	
# 			0 0 0 0 0 0 0 0 0 0 0
# 		]
# 		img_inv = @. ifelse(img == 0, 1, 0)
# 		vol = cat(img, img_inv, dims=3)
# 		container2 = []
# 		for i in 1:10
# 			push!(container2, vol)
# 		end
# 		vol_inv = CuArray(cat(container2..., dims=3))
# 		output, v, z = CUDA.zeros(size(vol_inv)), CUDA.ones(Int32, size(vol_inv)), CUDA.ones(size(vol_inv) .+ 1)
# 		tfm = SquaredEuclidean()
# 		test = transform_gpu3!(boolean_indicator(vol_inv), tfm; output=output, v=v, z=z)
# 		a1 = img_inv
# 		a2 = img
# 		ans = cat(a1, a2, dims=3)
# 		container_a = []
# 		for i in 1:10
# 			push!(container_a, ans)
# 		end
# 		answer = cat(container_a..., dims=3)
# 		@test test == CuArray(answer)
# 	end
# else
# 	@warn "CUDA"
# end;

# ╔═╡ Cell order:
# ╠═cd8bf944-2329-11ed-208f-1b2e91673a5e
# ╠═373a8802-bbac-4ab5-abe4-420ccbb61ea0
# ╟─4a9d1a91-d2ac-4c30-bc2f-f2dd0e27cc43
# ╟─9f2edcc0-4f04-4881-9e83-25f92bc42119
# ╟─81f78cda-0efa-4a74-bf4c-a39197d9b73f
# ╠═d3f4223e-f697-4a31-8825-66b1fae2b4f5
# ╠═d48cc7be-b769-4a64-970e-5982105fd382
# ╠═4ded097b-5e18-47af-bbbb-aa9fb626a9d1
# ╟─3b1fcab4-ca39-40c4-8cfa-410acfd6b006
# ╠═eafcd9e8-d691-4ae5-9e71-e999f8b6702c
# ╠═2f5755cb-e575-4ca9-a8b3-f2fd7846c068
# ╟─755aa045-73cd-425c-8057-50e65c342716
# ╠═afc433aa-ecc9-443a-bbd4-55bd48730937
# ╟─33980a16-c38d-4ff8-afdc-bfa8b2dcb482
# ╟─bf72b4f5-5e6e-4e2f-b1da-45e389815a60
# ╠═fcab8ebd-6799-4ef8-90f8-959af5bfb375
# ╠═bb7361e0-5a20-4454-87bf-a3cbf568a94b
# ╟─17eb24ab-6751-4b43-aa16-e779ae10e676
# ╠═4c066e7c-4f1d-4795-bfb7-525b999e2a53
# ╠═ddfaf97d-052f-40b5-8e02-9cbaa0f3a801
# ╟─8efd0196-a621-46e9-9ff3-3f4801820ee4
# ╟─d8dc96f7-b61b-4021-8094-bd70a6b66d5e
# ╠═bf7df452-38b7-4937-a981-70cdd820622a
# ╠═91c1a3a4-d2cd-4bd3-8994-2efb501aa3e8
# ╟─3b5f2010-48f7-4332-8742-60c71c4a01dd
# ╠═0ccd566f-18b7-4f99-93db-8df1379e3cd1
# ╠═ad76fe0e-c7f4-4e5e-9a5f-11f597274411
# ╟─7393ca78-681c-4fc2-b376-c5279812b12d
# ╟─d1fc47ea-920f-48f2-8fec-bcdfcae5d04c
# ╠═e7b6805a-6362-49a6-8124-4baa09e44937
# ╠═ab075b7b-4634-491d-b68b-9e3399dd814e
# ╟─b5ba030e-9d5e-4ce5-b138-f3be7fea0e23
# ╠═223acdc8-b5de-4eb1-9c80-19c3281e0dfd
