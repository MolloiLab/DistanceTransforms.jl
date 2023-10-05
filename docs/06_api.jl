### A Pluto.jl notebook ###
# v0.19.11

#> [frontmatter]
#> title = "API"
#> category = "API"

using Markdown
using InteractiveUtils

# ╔═╡ 64fe85d7-be32-400f-a883-2c67813f9273
# ╠═╡ show_logs = false
begin
	let
		using Pkg
		Pkg.activate(mktempdir())
		Pkg.Registry.update()
		Pkg.add("PlutoUI")
		Pkg.add(url="https://github.com/Dale-Black/DistanceTransforms.jl")
	end
	
	using PlutoUI
	using DistanceTransforms
end

# ╔═╡ 4ab83df3-2b5a-4499-8f0e-e050e2d734c2
TableOfContents()

# ╔═╡ a86cb4f5-146c-4ef0-81c8-89b4eea9f6bf
md"""
## DistanceTransform
"""

# ╔═╡ 275dbfb2-9fc6-4070-ab08-cb72df2e03c1
@doc DistanceTransform

# ╔═╡ 61563bfb-6dcf-4c04-b26e-6b3db787ac36
md"""
## Borgefors
"""

# ╔═╡ a4642582-2fa3-40c8-9487-b80bd6669b7a
@doc Borgefors

# ╔═╡ 45876212-a0c4-48f6-851a-96c1bfed6130
md"""
## Felzenszwalb
"""

# ╔═╡ 70b218bf-7bc9-4812-9a9c-b330a9e90841
@doc Felzenszwalb

# ╔═╡ e81cd30d-9c5a-4b62-9963-680163028dde
md"""
## Wenbo
"""

# ╔═╡ 64b552f7-30a8-4dae-8b2f-ec17bc89542c
@doc Wenbo

# ╔═╡ bbd04e76-a70a-4b5d-a751-91e1ea0505e6
md"""
## boolean_indicator
"""

# ╔═╡ 59ec7e38-76e2-4a45-9701-f82eae268205
@doc boolean_indicator

# ╔═╡ 4a2d7b81-0837-47b4-abc7-135549036dbb
md"""
## Maurer
"""

# ╔═╡ 675a8ae4-9152-4e77-80eb-b39dc97e04b0
@doc Maurer

# ╔═╡ c3ab2951-b998-4b4f-afbc-20dcacd4bf18
md"""
## transform
"""

# ╔═╡ 00802ec0-a9bf-4930-ba89-834437e9c0af
@doc transform

# ╔═╡ e6d7c77d-a946-44f8-97c6-4f493aa76446
md"""
## transform!
"""

# ╔═╡ cf9ad397-0093-4a94-9f41-0f0b122542ae
@doc transform!

# ╔═╡ 53129038-71d5-48e0-857f-6431e1d7998e
md"""
#### Helper
"""

# ╔═╡ fd04d338-31db-4cbb-8044-ce1738400796
begin
	exported_names = []
	nms = names(DistanceTransforms)
	for i in 1:length(nms)
		nm = getfield(@__MODULE__, nms[i])
		push!(exported_names, nm)
	end
	idx = findall(x->x==DistanceTransforms, exported_names)
	deleteat!(exported_names, idx)
end

# ╔═╡ e70bb49b-f340-446c-afad-fae2ee292e4b
exported_names

# ╔═╡ Cell order:
# ╠═64fe85d7-be32-400f-a883-2c67813f9273
# ╠═4ab83df3-2b5a-4499-8f0e-e050e2d734c2
# ╠═e70bb49b-f340-446c-afad-fae2ee292e4b
# ╟─a86cb4f5-146c-4ef0-81c8-89b4eea9f6bf
# ╟─275dbfb2-9fc6-4070-ab08-cb72df2e03c1
# ╟─61563bfb-6dcf-4c04-b26e-6b3db787ac36
# ╟─a4642582-2fa3-40c8-9487-b80bd6669b7a
# ╟─45876212-a0c4-48f6-851a-96c1bfed6130
# ╟─70b218bf-7bc9-4812-9a9c-b330a9e90841
# ╟─4a2d7b81-0837-47b4-abc7-135549036dbb
# ╟─675a8ae4-9152-4e77-80eb-b39dc97e04b0
# ╟─e81cd30d-9c5a-4b62-9963-680163028dde
# ╟─64b552f7-30a8-4dae-8b2f-ec17bc89542c
# ╟─bbd04e76-a70a-4b5d-a751-91e1ea0505e6
# ╟─59ec7e38-76e2-4a45-9701-f82eae268205
# ╟─c3ab2951-b998-4b4f-afbc-20dcacd4bf18
# ╟─00802ec0-a9bf-4930-ba89-834437e9c0af
# ╟─e6d7c77d-a946-44f8-97c6-4f493aa76446
# ╟─cf9ad397-0093-4a94-9f41-0f0b122542ae
# ╟─53129038-71d5-48e0-857f-6431e1d7998e
# ╟─fd04d338-31db-4cbb-8044-ce1738400796
