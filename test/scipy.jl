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

# ╔═╡ 061664e3-83b4-424a-bca6-a4e8c4e93d1e
begin
	x = [
		1 1 0 0
		0 1 1 0
		0 1 0 1
		0 1 0 0
	]
	test = transform(x, Scipy())
end

# ╔═╡ 8cbd0d80-66a1-4dc5-a12d-7f9d79c40be7
# @testset "Scipy 1D" begin
# 	x = [1, 1, 0, 0]
# 	test = transform(x, Scipy())
# 	answer = transform(x, Maurer())
# 	@test test == answer
# end;

# ╔═╡ Cell order:
# ╠═ffeb49bc-b3b1-11ed-28ed-0561ebf8eacc
# ╠═01619f53-766d-4bd5-b6fe-db186e4c663b
# ╟─a8ec2e49-31ba-48a6-b666-9998ac853642
# ╟─eccb5bf2-4127-437d-904b-c9481b534542
# ╠═061664e3-83b4-424a-bca6-a4e8c4e93d1e
# ╠═8cbd0d80-66a1-4dc5-a12d-7f9d79c40be7
