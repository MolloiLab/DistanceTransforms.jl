### A Pluto.jl notebook ###
# v0.19.26

#> [frontmatter]
#> title = "Getting Started"
#> category = "Tutorials"

using Markdown
using InteractiveUtils

# ╔═╡ 26b9d680-1004-4440-a392-16c178230066
# ╠═╡ show_logs = false
begin
	let
		using Pkg
		Pkg.activate(mktempdir())
		Pkg.Registry.update()
		Pkg.add("PlutoUI")
		Pkg.add("CairoMakie")
		Pkg.add("BenchmarkTools")
		Pkg.add(url="https://github.com/Dale-Black/DistanceTransforms.jl")
	end
	
	using PlutoUI
	using CairoMakie
	using BenchmarkTools
	using DistanceTransforms
end

# ╔═╡ 32a932f4-cade-434f-a489-282bb909a04c
md"""
## Import packages
First, let's import the most up-to-date version of DistanceTransforms.jl, which can be found on the main/master branch of the [GitHub repository](https://github.com/Dale-Black/DistanceTransforms.jl). 
Because we are using the unregistered version (most recent) we will need to `Pkg.add` this explicitly, without using Pluto's built-in package manager. Be aware, this can take a long time, especially if this is the first time being downloaded. Future work on this package will focus on improving this.
To help with the formatting of this documentation we will also add [PlutoUI.jl](https://github.com/JuliaPluto/PlutoUI.jl).
Lastly, to visualize results and time the functions were going to add [Makie.jl](https://github.com/JuliaPlots/Makie.jl) and [BenchmarkTools.jl](https://github.com/JuliaCI/BenchmarkTools.jl).
"""

# ╔═╡ c579c92a-3d1c-4213-82c5-cca54b5c0144
TableOfContents()

# ╔═╡ 1b06308d-4084-4be8-9758-8052277e5ac1
md"""
## Quick start
Distance transforms are an important part of many computer vision-related tasks. Let's create some sample data and see how DistanceTransforms.jl can be used to apply efficient distance transform operations on arrays in Julia.
"""

# ╔═╡ 97aad592-8362-48ce-87da-5ab3e8eb6f95
md"""
The quintessential distance transform operation in DistanceTransforms.jl is just a wrapper function combining the `feature_transform` and `distance_transform` functions from the excellent [ImageMorphology](https://github.com/JuliaImages/ImageMorphology.jl) package. 
To utilize this in DistanceTransforms.jl all one must do is call `transform(x, Maurer())`, where `x` is any boolean or integer array of ones and zeros and `Maurer()` is what specifies that this transform operation is using the one described in Maurer, et. al which was written in ImageMorphology.jl
"""

# ╔═╡ 862747e5-e88f-47ab-bc97-f5a456738b22
array1 = [
	0 1 1 0
	0 0 0 0
	1 1 0 0
]

# ╔═╡ b9ba3bde-8779-4f3c-a3fb-ef157e121632
transform(array1, Maurer())

# ╔═╡ b2d06446-991c-4311-993f-87ddef5e8828
md"""
As you can see, each element in the array is either zero (corresponding to background), which remains zero, or is one (corresponding to foreground) that then gets replaced by the Euclidean distance to the nearest zero. 
We can see this easily using Makie.jl.
"""

# ╔═╡ d3498bb0-6f8f-470a-b1de-b1760229cc1c
heatmap(array1, colormap=:grays)

# ╔═╡ ec3a90f1-f2d7-4d5a-8785-476073ee0079
heatmap(transform(array1, Maurer()); colormap=:grays)

# ╔═╡ 01752ec8-9917-4254-8c92-daed39aeea39
md"""
## `transform`s
The rest of the DistanceTransform.jl library is built around a common `transform` interface. Users can apply various distance transform algorithms to arrays all under one unified interface.
Let's examine two different options:
"""

# ╔═╡ be063896-415a-4178-8362-10073feadadf
md"""
### `Borgefors`
This algorithm is based on the 3-4 chamfer distance transform, as described by [Gunilla Borgefors](https://studentportalen.uu.se/uusp-filearea-tool/download.action?nodeId=214320&toolAttachmentId=64777)
To get started we need to initialize the type of algorithm or `tfm` we want to use, then we run this through the `transform` function. We can again use `array1` for this task.
"""

# ╔═╡ 37ed41d1-793d-4ec3-99e7-0b0a387b76f3
tfm = Borgefors()

# ╔═╡ d9857001-e4e4-4bc9-b211-a7ed21fd685f
dt = zeros(size(array1))

# ╔═╡ 90cef7f0-5437-4bc7-b0cc-78c89030351f
borgefors_transform = transform(array1, dt, tfm)

# ╔═╡ c7c26a02-bad1-43b5-a119-3f8c4e53180c
md"""
Notice that the distance computed to the nearest backgroun `0` element is no longer a Euclidean distance, but instead follows the rules of the 3-4 Chamfer distance transform.
"""

# ╔═╡ bf546f09-7604-4f11-8676-34a7ac571780
md"""
### `Felzenszwalb`
This algorithm is based on the squared Euclidean distance transform, as described by [Felzenszwalb and
Huttenlocher] (DOI: 10.4086/toc.2012.v008a019)
We will use a similar approach to get started, with one extra step; the array must either be designated as a boolean indicator (meaning zeros correspond to background and ones correspond to foreground) or ignored. Most of the time the array should be considered a boolean indicator:
"""

# ╔═╡ 77de39d7-5d30-410f-baa5-460ae2d8a4a3
array2 = [
	0 1 1 0 1
	0 0 0 1 0
	1 1 0 0 0
]

# ╔═╡ 7c557d32-2bb8-47a2-8216-e18755d258c7
array2_bool = boolean_indicator(array2)

# ╔═╡ 00a4b3ed-ca0a-4c67-b400-cb9830dbbff2
tfm2 = Felzenszwalb()

# ╔═╡ d6b06775-2ded-4e15-9616-f3e4653d1396
sq_euc_transform = transform(array2_bool, tfm2)

# ╔═╡ 5095c3e1-a873-4cec-988d-fb6bfa8a0aaa
md"""
It should be clear to see that the `transform` returns the Euclidean distance, just squared. To find the true Euclidean distance, all one must do to get the true Euclidean distance take the square root of each element. In many cases, this is not necessary and can add time which is why it is left optional.
"""

# ╔═╡ 7597fa37-b406-4fca-8eaa-1627b1eb835d
euc_transform = sqrt.(sq_euc_transform)

# ╔═╡ 46a38900-bf33-4f5c-a6e3-14f961403185
euc_transform ≈ transform(array2, Maurer())

# ╔═╡ Cell order:
# ╟─32a932f4-cade-434f-a489-282bb909a04c
# ╠═26b9d680-1004-4440-a392-16c178230066
# ╠═c579c92a-3d1c-4213-82c5-cca54b5c0144
# ╟─1b06308d-4084-4be8-9758-8052277e5ac1
# ╟─97aad592-8362-48ce-87da-5ab3e8eb6f95
# ╠═862747e5-e88f-47ab-bc97-f5a456738b22
# ╠═b9ba3bde-8779-4f3c-a3fb-ef157e121632
# ╟─b2d06446-991c-4311-993f-87ddef5e8828
# ╠═d3498bb0-6f8f-470a-b1de-b1760229cc1c
# ╠═ec3a90f1-f2d7-4d5a-8785-476073ee0079
# ╟─01752ec8-9917-4254-8c92-daed39aeea39
# ╟─be063896-415a-4178-8362-10073feadadf
# ╠═37ed41d1-793d-4ec3-99e7-0b0a387b76f3
# ╠═d9857001-e4e4-4bc9-b211-a7ed21fd685f
# ╠═90cef7f0-5437-4bc7-b0cc-78c89030351f
# ╟─c7c26a02-bad1-43b5-a119-3f8c4e53180c
# ╟─bf546f09-7604-4f11-8676-34a7ac571780
# ╠═77de39d7-5d30-410f-baa5-460ae2d8a4a3
# ╠═7c557d32-2bb8-47a2-8216-e18755d258c7
# ╠═00a4b3ed-ca0a-4c67-b400-cb9830dbbff2
# ╠═d6b06775-2ded-4e15-9616-f3e4653d1396
# ╟─5095c3e1-a873-4cec-988d-fb6bfa8a0aaa
# ╠═7597fa37-b406-4fca-8eaa-1627b1eb835d
# ╠═46a38900-bf33-4f5c-a6e3-14f961403185
