### A Pluto.jl notebook ###
# v0.19.11

using Markdown
using InteractiveUtils

# ╔═╡ dcdd2c68-3ec4-11ed-2e84-77926655aa11
# ╠═╡ show_logs = false
begin
	using Pkg
	Pkg.activate("..")
	using Revise
	using PlutoUI
	using Test
	using CUDA
	using DistanceTransforms
	using LoopVectorization
end

# ╔═╡ 6ca83ba8-2e67-4b8f-9ed9-c903a867dc75
TableOfContents()

# ╔═╡ b3c5c757-2237-41d8-bf2e-48da4878aeca
@testset "boolean_indicator" begin
	f = rand([0, 1], 100, 100)
	f_bool = Bool.(f)

	@test boolean_indicator(f) == boolean_indicator(f_bool)
end

# ╔═╡ Cell order:
# ╠═dcdd2c68-3ec4-11ed-2e84-77926655aa11
# ╠═6ca83ba8-2e67-4b8f-9ed9-c903a867dc75
# ╠═b3c5c757-2237-41d8-bf2e-48da4878aeca
