### A Pluto.jl notebook ###
# v0.19.22

using Markdown
using InteractiveUtils

# ╔═╡ a91769e4-a43b-41d2-8525-1daa923e8949
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

# ╔═╡ d8d774e1-882f-491d-b383-a7b2c314cdc9
TableOfContents()

# ╔═╡ 5cadc26e-698f-4cbf-8bf5-09b83dbd9447
md"""
# `Borgefors`
"""

# ╔═╡ 38754b7d-d2e7-4e44-a150-7efab14d585b
md"""
## Regular
"""

# ╔═╡ 0378331e-7e8d-4543-9f38-9af51da24314
md"""
### 2D
"""

# ╔═╡ 2e8f1750-7fc0-4d21-aa25-d00bc67bc048
@testset "Borgefors 2D" begin
	x = [
		1 1 0 0
		0 1 1 0
		0 1 0 1
		0 1 0 0
	]
	dt = zeros(Float32, size(x))
	tfm = Borgefors()
	answer = [
		0 0 3 4
		3 0 0 3
		3 0 3 0
		3 0 3 3
	]
	@test transform(x, dt, tfm) == answer
end;

# ╔═╡ b268d818-90b5-4382-a330-b8bc9e18b914
@testset "Borgefors 2D" begin
	x = [
		1 1 0 0
		0 1 0 0
		0 1 0 0
		0 1 0 0
	]
	dt = zeros(Float32, size(x))
	tfm = Borgefors()
	answer = [
		0 0 3 6
		3 0 3 6
		3 0 3 6
		3 0 3 6
	]
	@test transform(x, dt, tfm) == answer
end;

# ╔═╡ 6c78fadf-7cab-445f-b77e-4803dd06a887
md"""
### 3D
"""

# ╔═╡ dde0dc70-4422-4698-92bd-803347c04cda
@testset "Borgefors 3D" begin
	x1 = [
		1 1 0 0
		0 1 0 0
		0 1 0 0
		0 1 0 0
	]
	x = cat(x1, x1; dims=3)
	dt = zeros(Float32, size(x))
	tfm = Borgefors()
	a1 = [
		0 0 3 6
		3 0 3 6
		3 0 3 6
		3 0 3 6
	]
	answer = cat(a1, a1; dims=3)
	@test transform(x, dt, tfm) == answer
end;

# ╔═╡ Cell order:
# ╠═a91769e4-a43b-41d2-8525-1daa923e8949
# ╠═d8d774e1-882f-491d-b383-a7b2c314cdc9
# ╟─5cadc26e-698f-4cbf-8bf5-09b83dbd9447
# ╟─38754b7d-d2e7-4e44-a150-7efab14d585b
# ╟─0378331e-7e8d-4543-9f38-9af51da24314
# ╠═2e8f1750-7fc0-4d21-aa25-d00bc67bc048
# ╠═b268d818-90b5-4382-a330-b8bc9e18b914
# ╟─6c78fadf-7cab-445f-b77e-4803dd06a887
# ╠═dde0dc70-4422-4698-92bd-803347c04cda
