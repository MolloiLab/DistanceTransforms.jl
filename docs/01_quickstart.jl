### A Pluto.jl notebook ###
# v0.17.2

using Markdown
using InteractiveUtils

# ╔═╡ 22c33fe6-4efd-11ec-351c-25a9d73ec50a
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

# ╔═╡ 0095c4d9-e9d0-4769-83ff-fd7f4346be8f
md"""
## Import packages
First, let's import the most up-to-date version of DistanceTransforms.jl, which can be found on the main/master branch of the [GitHub repository](https://github.com/Dale-Black/DistanceTransforms.jl). 

Because we are using the unregistered version (most recent) we will need to `Pkg.add` this explicitly, without using Pluto's built-in package manager. Be aware, this can take a long time, especially if this is the first time being downloaded. Future work on this package will focus on improving this.

To help with the formatting of this documentation we will also add [PlutoUI.jl](https://github.com/JuliaPluto/PlutoUI.jl).

Lastly, to visualize results and time the functions were going to add [Makie.jl](https://github.com/JuliaPlots/Makie.jl) and [BenchmarkTools.jl](https://github.com/JuliaCI/BenchmarkTools.jl).
"""

# ╔═╡ 7f2b1c79-d6dd-41f8-850e-32a18ba1b2a4
TableOfContents()

# ╔═╡ d70e8f0c-9052-47a4-bac7-af8362afda2c
md"""
## Quick start
Distance transforms are an important part of many computer vision-related tasks. Let's create some sample data and see how DistanceTransforms.jl can be used to apply efficient distance transform operations on arrays in Julia.
"""

# ╔═╡ 6c01894f-442a-497c-9519-dfc18b63be06
md"""
The quintessential distance transform operation in DistanceTransforms.jl is just a wrapper function combining the `feature_transform` and `distance_transform` functions from the excellent [ImageMorphology](https://github.com/JuliaImages/ImageMorphology.jl) package. 

To utilize this in DistanceTransforms.jl all one must do is call `euclidean(x)`, where `x` is any boolean or integer array of ones and zeros
"""

# ╔═╡ 023529de-8f6b-4339-89a6-23d8b8fb881f
array1 = [
	1 0 0 1
	1 1 1 1
	0 0 1 1
]

# ╔═╡ cc17f05a-6e29-4a81-a45f-2dcfdedf2e88
euclidean(array1)

# ╔═╡ 0ad7423a-f2ff-417a-bfd1-e400a994e88c
md"""
As you can see, each element in the array is either zero, which remains zero, or is one that then gets replaced by the Euclidean distance to the nearest zero. 

We can see this easily using Makie.jl.
"""

# ╔═╡ fde6da34-cd4d-4d97-aeb5-5a745cdffa13
heatmap(array1, colormap=:grays)

# ╔═╡ f629fb0b-530e-4a06-bfb0-81c27ae12301
heatmap(euclidean(array1); colormap=:grays)

# ╔═╡ 66ee23f9-9a36-427f-b062-306e513eea69
md"""
## `transform`s
The rest of the DistanceTransform.jl library is built around a common `transform` interface. Users can apply various distance transform algorithms to arrays all under one unified interface.

Let's examine two different options:
"""

# ╔═╡ 5fee8340-03ba-49a1-b0f6-975c7624e684
md"""
### `Chamfer`
This algorithm is based on the 3-4 Chamfer distance transform, as described by [Gunilla Borgefors](https://studentportalen.uu.se/uusp-filearea-tool/download.action?nodeId=214320&toolAttachmentId=64777)

To get started we need to initialize the type of algorithm or `tfm` we want to use, then we run this through the `transform` function. We can again use `array1` for this task.
"""

# ╔═╡ 22fa734b-73c0-4ccf-a646-af03e309c973
tfm = Chamfer(array1)

# ╔═╡ c7ba037e-5f85-44ec-8701-787f30cbfae3
chamfer_transform = transform(array1, tfm)

# ╔═╡ 04e19c33-992c-483b-9e4a-e728d1a507b9
md"""
Notice that the distance computed to the nearest backgroun `0` element is no longer a Euclidean distance, but instead follows the rules of the 3-4 Chamfer distance transform.
"""

# ╔═╡ 25c49c9f-4ad4-4d6d-8993-c6e6e6fd9303
md"""
### `SquaredEuclidean`
This algorithm is based on the squared Euclidean distance transform, as described by [Felzenszwalb and
Huttenlocher] (DOI: 10.4086/toc.2012.v008a019)

We will use a similar approach to get started, with one extra step; the array must either be designated as a boolean indicator (meaning zeros correspond to background and ones correspond to foreground) or ignored. Most of the time the array should be considered a boolean indicator:
"""

# ╔═╡ bacc8e48-40c6-4116-8f39-431fbebf065e
array2 = [
	0 1 1 0 1
	0 0 0 1 0
	1 1 0 0 0
]

# ╔═╡ 33e4022e-e7ff-467e-9f42-ed79b5fd2650
array2_bool = boolean_indicator(array2)

# ╔═╡ a4bd1fc3-ce53-41e9-a3e5-d4b767feafcc
tfm2 = SquaredEuclidean(array2_bool)

# ╔═╡ 0c625365-32ef-4261-a7b6-4c5002f0c464
sq_euc_transform = transform(array2_bool, tfm2)

# ╔═╡ 0c1e2872-f9a5-4697-b537-deac7f959a43
md"""
Notice that the initialization of the `SquaredEuclidean` transformation algorithm returns three separate arrays. This can be understood better by looking through the source code, but simply it's essential for utilizing this transformation algorithm in a similar way while taking advantage of multi-threading and/or GPU accelerated hardware.

It should also be clear to see that the `transform` returns the Euclidean distance, just squared. To find the true Euclidean distance, all one must do is return the take the square root of each element. In many cases, this is not necessary and can add time which is why it is left as optional.
"""

# ╔═╡ 926dadf5-38a6-4e77-bec9-0ac76005eb87
euc_transform = sqrt.(sq_euc_transform)

# ╔═╡ Cell order:
# ╟─0095c4d9-e9d0-4769-83ff-fd7f4346be8f
# ╠═22c33fe6-4efd-11ec-351c-25a9d73ec50a
# ╠═7f2b1c79-d6dd-41f8-850e-32a18ba1b2a4
# ╟─d70e8f0c-9052-47a4-bac7-af8362afda2c
# ╟─6c01894f-442a-497c-9519-dfc18b63be06
# ╠═023529de-8f6b-4339-89a6-23d8b8fb881f
# ╠═cc17f05a-6e29-4a81-a45f-2dcfdedf2e88
# ╟─0ad7423a-f2ff-417a-bfd1-e400a994e88c
# ╠═fde6da34-cd4d-4d97-aeb5-5a745cdffa13
# ╠═f629fb0b-530e-4a06-bfb0-81c27ae12301
# ╟─66ee23f9-9a36-427f-b062-306e513eea69
# ╟─5fee8340-03ba-49a1-b0f6-975c7624e684
# ╠═22fa734b-73c0-4ccf-a646-af03e309c973
# ╠═c7ba037e-5f85-44ec-8701-787f30cbfae3
# ╟─04e19c33-992c-483b-9e4a-e728d1a507b9
# ╟─25c49c9f-4ad4-4d6d-8993-c6e6e6fd9303
# ╠═bacc8e48-40c6-4116-8f39-431fbebf065e
# ╠═33e4022e-e7ff-467e-9f42-ed79b5fd2650
# ╠═a4bd1fc3-ce53-41e9-a3e5-d4b767feafcc
# ╠═0c625365-32ef-4261-a7b6-4c5002f0c464
# ╟─0c1e2872-f9a5-4697-b537-deac7f959a43
# ╠═926dadf5-38a6-4e77-bec9-0ac76005eb87
