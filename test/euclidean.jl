### A Pluto.jl notebook ###
# v0.19.11

using Markdown
using InteractiveUtils

# ╔═╡ 1594a983-3c6b-45c1-94c6-21d6ebbc918e
begin
	using Pkg
	Pkg.activate("..")
	using Revise
	using PlutoUI
	using Test
	using CUDA
	using DistanceTransforms
end

# ╔═╡ 559815d5-ef99-4c28-8673-c38336542d14
TableOfContents()

# ╔═╡ 7dd8f4c6-f891-4b25-8431-96117b96e5c0
md"""
# `euclidean`
"""

# ╔═╡ d7eb672d-e687-4934-86eb-2a3fb8b53ea7
md"""
## 1D
"""

# ╔═╡ 15579321-6bdf-424f-82bc-a51dd22a04f5
@testset "euclidean 1D" begin
	x = [1, 1, 0, 0]
	answer = [2, 1, 0, 0]
	@test euclidean(x) == answer
end;

# ╔═╡ 87c4750a-06f3-4446-a563-329bcfbd27c5
@testset "euclidean 1D" begin
	x = [1, 0, 0, 1, 1, 1]
	answer = [1, 0, 0, 1, 2, 3]
	@test euclidean(x) == answer
end;

# ╔═╡ d2aac89d-821e-43de-8bda-0bb626ac6011
md"""
## 2D
"""

# ╔═╡ c0cdf4cf-66d0-4e41-8973-b0d23402edfa
@testset "euclidean 2D" begin
	x = [
		1 1 0 0
		0 1 1 0
		0 1 0 1
		0 1 0 0
	]
	answer = [
		1.0 1.0 0.0 0.0
		0.0 1.0 1.0 0.0
		0.0 1.0 0.0 1.0
		0.0 1.0 0.0 0.0
	]
	@test euclidean(x) == answer
end;

# ╔═╡ 238684cd-9591-4f5f-9df6-5246d160cae2
@testset "euclidean 2D" begin
	x = [
		1 1 1 0
		0 1 1 1
		0 1 1 1
		0 1 1 1
	]
	answer = [
		1.0 1.4142135623730951 1.0 0.0
		0.0 1.0 1.4142135623730951 1.0
		0.0 1.0 2.0 2.0
		0.0 1.0 2.0 3.0
	]
	@test euclidean(x) ≈ answer
end;

# ╔═╡ 36c8ad7f-d6d9-4bb2-82de-212e8d66f268
@testset "euclidean 2D" begin
	x = Bool.([
		1 1 0 0
		0 1 1 0
		0 1 0 1
		0 1 0 0
	])
	answer = [
		1.0 1.0 0.0 0.0
		0.0 1.0 1.0 0.0
		0.0 1.0 0.0 1.0
		0.0 1.0 0.0 0.0
	]
	@test euclidean(x) == answer
end;

# ╔═╡ 1569c3c8-eb3c-4453-b404-ce0f3b21d0e1
md"""
## 3D
"""

# ╔═╡ fdb8e7b5-b721-4c68-9603-8244f0562052
@testset "euclidean 3D" begin
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
	test = euclidean(vol_inv)
	a1 = img
	a2 = img_inv
	ans = cat(a1, a2, dims=3)
	container_a = []
	for i in 1:10
		push!(container_a, ans)
	end
	answer = cat(container_a..., dims=3)
	@test test == answer
end;

# ╔═╡ Cell order:
# ╠═1594a983-3c6b-45c1-94c6-21d6ebbc918e
# ╠═559815d5-ef99-4c28-8673-c38336542d14
# ╟─7dd8f4c6-f891-4b25-8431-96117b96e5c0
# ╟─d7eb672d-e687-4934-86eb-2a3fb8b53ea7
# ╠═15579321-6bdf-424f-82bc-a51dd22a04f5
# ╠═87c4750a-06f3-4446-a563-329bcfbd27c5
# ╟─d2aac89d-821e-43de-8bda-0bb626ac6011
# ╠═c0cdf4cf-66d0-4e41-8973-b0d23402edfa
# ╠═238684cd-9591-4f5f-9df6-5246d160cae2
# ╠═36c8ad7f-d6d9-4bb2-82de-212e8d66f268
# ╟─1569c3c8-eb3c-4453-b404-ce0f3b21d0e1
# ╠═fdb8e7b5-b721-4c68-9603-8244f0562052
