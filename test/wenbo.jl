### A Pluto.jl notebook ###
# v0.19.11

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
end

# ╔═╡ 3b8595ae-b215-4e3b-b615-ff460793f028
TableOfContents()

# ╔═╡ 4a043231-7888-4408-a663-4541e8606245
md"""
# `Wenbo`
"""

# ╔═╡ edc7480a-11ec-4952-8984-af97d3967639
md"""
## Regular
"""

# ╔═╡ ab715087-a707-485b-9463-8d17fe0bcaab
md"""
### 1D
"""

# ╔═╡ f860bf51-4c1f-4b82-a67d-d5294733345f
@testset "wenbo 1D" begin
	f = [1, 1, 0, 0, 0, 1, 1]
	output, pointerA, pointerB = zeros(length(f)), 1, 1
	tfm = Wenbo()
	test = transform(boolean_indicator(f), tfm; output=output, pointerA=pointerA, pointerB=pointerB)
	answer = [0.0, 0.0, 1.0, 4.0, 1.0, 0.0, 0.0]
	@test test == answer
end;

# ╔═╡ f6d1f13b-2b81-4253-ad80-38dd6c11eb93
@testset "wenbo 1D" begin
	f = [0, 0, 0, 1]
	output, pointerA, pointerB = zeros(length(f)), 1, 1
	tfm = Wenbo()
	test = transform(boolean_indicator(f), tfm; output=output, pointerA=pointerA, pointerB=pointerB)
	answer = [9.0, 4.0, 1.0, 0.0]
	@test test == answer
end;

# ╔═╡ 77970fcb-e584-4222-af3c-5b8482bab391
@testset "wenbo 1D" begin
	f = [1, 0, 0, 0]
	output, pointerA, pointerB = zeros(length(f)), 1, 1
	tfm = Wenbo()
	test = transform(boolean_indicator(f), tfm; output=output, pointerA=pointerA, pointerB=pointerB)
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
	output, pointerA, pointerB = zeros(size(vol_inv)), 1, 1
	tfm = Wenbo()
	test = transform(boolean_indicator(vol_inv), tfm; output=output, pointerA=pointerA, pointerB=pointerB)
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

# ╔═╡ Cell order:
# ╠═38a67f7e-3b1a-4a6c-9916-91eab4ad4d63
# ╠═3b8595ae-b215-4e3b-b615-ff460793f028
# ╟─4a043231-7888-4408-a663-4541e8606245
# ╟─edc7480a-11ec-4952-8984-af97d3967639
# ╟─ab715087-a707-485b-9463-8d17fe0bcaab
# ╠═f860bf51-4c1f-4b82-a67d-d5294733345f
# ╠═f6d1f13b-2b81-4253-ad80-38dd6c11eb93
# ╠═77970fcb-e584-4222-af3c-5b8482bab391
# ╟─eac8edcd-ee45-4824-a3b7-cd0535260cb6
# ╠═eb456533-684f-45e7-9591-7cebf493f63c
# ╠═350a21d9-552f-4d0a-a586-a8932fc4e245
# ╟─e05ea048-d7bd-48a1-b943-2dc9b9ba5e02
# ╠═b1d3b0d0-5103-4fdf-8d55-16babca9ee21
