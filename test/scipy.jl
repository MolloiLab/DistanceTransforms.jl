### A Pluto.jl notebook ###
# v0.19.22

using Markdown
using InteractiveUtils

# ╔═╡ ffeb49bc-b3b1-11ed-28ed-0561ebf8eacc
# ╠═╡ show_logs = false
begin
	using Pkg
	Pkg.activate("..")
	using Revise
	using PlutoUI
	using Test
	using DistanceTransforms
end

# ╔═╡ 01619f53-766d-4bd5-b6fe-db186e4c663b
TableOfContents()

# ╔═╡ a8ec2e49-31ba-48a6-b666-9998ac853642
md"""
# `Scipy`
"""

# ╔═╡ eccb5bf2-4127-437d-904b-c9481b534542
md"""
## 1D
"""

# ╔═╡ 8cbd0d80-66a1-4dc5-a12d-7f9d79c40be7
@testset "Scipy 1D" begin
	x = rand(Bool, 10)
	x_inv = .!x
	test = transform(x, Scipy())
	answer = transform(x_inv, Maurer())
	@test test == answer
end;

# ╔═╡ a35001dc-53c7-442c-8b03-cce1434c64fd
md"""
## 2D
"""

# ╔═╡ bf0f3e40-094b-48dc-b6c3-2a7209a64e71
@testset "Scipy 2D" begin
	x = rand(Bool, 10, 10)
	x_inv = .!x
	test = transform(x, Scipy())
	answer = transform(x_inv, Maurer())
	@test test ≈ answer
end

# ╔═╡ c60e9ab8-d9bb-48e5-84da-c8ca7a21329d
md"""
## 3D
"""

# ╔═╡ 96b87f1c-0328-4bc5-b246-7182ca14ffdc
@testset "Scipy 3D" begin
	x = rand(Bool, 10, 10, 10)
	x_inv = .!x
	test = transform(x, Scipy())
	answer = transform(x_inv, Maurer())
	@test test ≈ answer
end

# ╔═╡ Cell order:
# ╠═ffeb49bc-b3b1-11ed-28ed-0561ebf8eacc
# ╠═01619f53-766d-4bd5-b6fe-db186e4c663b
# ╟─a8ec2e49-31ba-48a6-b666-9998ac853642
# ╟─eccb5bf2-4127-437d-904b-c9481b534542
# ╠═8cbd0d80-66a1-4dc5-a12d-7f9d79c40be7
# ╟─a35001dc-53c7-442c-8b03-cce1434c64fd
# ╠═bf0f3e40-094b-48dc-b6c3-2a7209a64e71
# ╟─c60e9ab8-d9bb-48e5-84da-c8ca7a21329d
# ╠═96b87f1c-0328-4bc5-b246-7182ca14ffdc
