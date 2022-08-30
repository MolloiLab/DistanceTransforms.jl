### A Pluto.jl notebook ###
# v0.19.11

using Markdown
using InteractiveUtils

# ╔═╡ 66295dce-28aa-11ed-3bd9-750a958297ff
# ╠═╡ show_logs = false
begin
	using Pkg
	Pkg.activate(".")
	using Revise
	using PlutoUI
	using DistanceTransforms
	using CairoMakie
	using BenchmarkTools
end

# ╔═╡ ec34f34b-ce2f-4326-92a4-97c8a5f82543
md"""
## Import packages
First, let's import the most up-to-date version of DistanceTransforms.jl, which can be found on the main/master branch of the [GitHub repository](https://github.com/Dale-Black/DistanceTransforms.jl). 

Because we are using the unregistered version (most recent) we will need to `Pkg.add` this explicitly, without using Pluto's built-in package manager. Be aware, this can take a long time, especially if this is the first time being downloaded. Future work on this package will focus on improving this.

To help with the formatting of this documentation we will also add [PlutoUI.jl](https://github.com/JuliaPluto/PlutoUI.jl).

Lastly, to visualize results and time the functions were going to add [Makie.jl](https://github.com/JuliaPlots/Makie.jl) and [BenchmarkTools.jl](https://github.com/JuliaCI/BenchmarkTools.jl).
"""

# ╔═╡ 3e512f2e-0f38-4845-ab6a-37798b696ea3
TableOfContents()

# ╔═╡ ac674e28-31e0-4656-ba96-1f5d20c04c84
md"""
## Quick start
Distance transforms are an important part of many computer vision-related tasks. Let's create some sample data and see how DistanceTransforms.jl can be used to apply efficient distance transform operations on arrays in Julia.
"""

# ╔═╡ c400debb-e5c7-4101-b2b7-d6141aa0c437
md"""
The quintessential distance transform operation in DistanceTransforms.jl is just a wrapper function combining the `feature_transform` and `distance_transform` functions from the excellent [ImageMorphology](https://github.com/JuliaImages/ImageMorphology.jl) package. 

To utilize this in DistanceTransforms.jl all one must do is call `euclidean(x)`, where `x` is any boolean or integer array of ones and zeros
"""

# ╔═╡ 4e70becd-c4f1-439b-9f1e-26b81cc88ad8
array1 = [
	1 0 0 1
	1 1 1 1
	0 0 1 1
]

# ╔═╡ 980dbf98-f0b7-415d-8804-38423911c607
euclidean(array1)

# ╔═╡ 8e375233-5b18-43e7-ad61-c8a9b506952a
md"""
As you can see, each element in the array is either zero (corresponding to background), which remains zero, or is one (corresponding to foreground) that then gets replaced by the Euclidean distance to the nearest zero. 

We can see this easily using Makie.jl.
"""

# ╔═╡ 2695e161-0794-4f11-a244-807027e577ae
heatmap(array1, colormap=:grays)

# ╔═╡ 3a2376e8-b643-45cd-8f25-e0a4ab5f6bda
heatmap(euclidean(array1); colormap=:grays)

# ╔═╡ 5a0f39a5-a32f-455e-9c5f-1f199b60879c
md"""
## `transform`s
The rest of the DistanceTransform.jl library is built around a common `transform` interface. Users can apply various distance transform algorithms to arrays all under one unified interface.

Let's examine two different options:
"""

# ╔═╡ c1ed1336-a137-4964-8aa2-b9ba0853199b
md"""
### `Chamfer`
This algorithm is based on the 3-4 Chamfer distance transform, as described by [Gunilla Borgefors](https://studentportalen.uu.se/uusp-filearea-tool/download.action?nodeId=214320&toolAttachmentId=64777)

To get started we need to initialize the type of algorithm or `tfm` we want to use, then we run this through the `transform` function. We can again use `array1` for this task.
"""

# ╔═╡ 646cae43-1900-4d6a-981e-e219d7dbbeaa
tfm = Chamfer()

# ╔═╡ ef628d0b-0b1e-46a9-9bc6-646590f4a34e
dt = zeros(size(array1))

# ╔═╡ 1a71b8a3-a596-4fc1-a499-57f32c9c41c5
chamfer_transform = transform(array1, dt, tfm)

# ╔═╡ 5340a247-22fb-46aa-ab7f-6a025e42717f
md"""
Notice that the distance computed to the nearest backgroun `0` element is no longer a Euclidean distance, but instead follows the rules of the 3-4 Chamfer distance transform.
"""

# ╔═╡ 8bd885a9-39bf-4658-bd8c-ba344ef46970
md"""
### `SquaredEuclidean`
This algorithm is based on the squared Euclidean distance transform, as described by [Felzenszwalb and
Huttenlocher] (DOI: 10.4086/toc.2012.v008a019)

We will use a similar approach to get started, with one extra step; the array must either be designated as a boolean indicator (meaning zeros correspond to background and ones correspond to foreground) or ignored. Most of the time the array should be considered a boolean indicator:
"""

# ╔═╡ 73ec8426-f413-41a3-9dd5-73292915cdf4
array2 = [
	0 1 1 0 1
	0 0 0 1 0
	1 1 0 0 0
]

# ╔═╡ f3ee1ac5-855e-4900-b74f-238292eac00d
array2_bool = boolean_indicator(array2)

# ╔═╡ 7dace88b-971e-4712-a895-dff5403f2ce0
tfm2 = SquaredEuclidean()

# ╔═╡ da6b2cf7-4e3d-486a-84f2-4b0cc8cb65da
sq_euc_transform = transform(array2_bool, tfm2)

# ╔═╡ 3c27b9ac-dc4c-4a9d-9dfb-6cfc253bdcce
md"""
It should be clear to see that the `transform` returns the Euclidean distance, just squared. To find the true Euclidean distance, all one must do to get the true Euclidean distance take the square root of each element. In many cases, this is not necessary and can add time which is why it is left optional.
"""

# ╔═╡ 3c59dfa6-bb29-4c6c-8fbc-ff845786c0b9
euc_transform = sqrt.(sq_euc_transform)

# ╔═╡ 5fa0662b-ae8e-4c7b-a2f9-284a11f4ab50
euc_transform ≈ euclidean(array2)

# ╔═╡ Cell order:
# ╠═66295dce-28aa-11ed-3bd9-750a958297ff
# ╟─ec34f34b-ce2f-4326-92a4-97c8a5f82543
# ╠═3e512f2e-0f38-4845-ab6a-37798b696ea3
# ╟─ac674e28-31e0-4656-ba96-1f5d20c04c84
# ╟─c400debb-e5c7-4101-b2b7-d6141aa0c437
# ╠═4e70becd-c4f1-439b-9f1e-26b81cc88ad8
# ╠═980dbf98-f0b7-415d-8804-38423911c607
# ╟─8e375233-5b18-43e7-ad61-c8a9b506952a
# ╠═2695e161-0794-4f11-a244-807027e577ae
# ╠═3a2376e8-b643-45cd-8f25-e0a4ab5f6bda
# ╟─5a0f39a5-a32f-455e-9c5f-1f199b60879c
# ╟─c1ed1336-a137-4964-8aa2-b9ba0853199b
# ╠═646cae43-1900-4d6a-981e-e219d7dbbeaa
# ╠═ef628d0b-0b1e-46a9-9bc6-646590f4a34e
# ╠═1a71b8a3-a596-4fc1-a499-57f32c9c41c5
# ╟─5340a247-22fb-46aa-ab7f-6a025e42717f
# ╟─8bd885a9-39bf-4658-bd8c-ba344ef46970
# ╠═73ec8426-f413-41a3-9dd5-73292915cdf4
# ╠═f3ee1ac5-855e-4900-b74f-238292eac00d
# ╠═7dace88b-971e-4712-a895-dff5403f2ce0
# ╠═da6b2cf7-4e3d-486a-84f2-4b0cc8cb65da
# ╟─3c27b9ac-dc4c-4a9d-9dfb-6cfc253bdcce
# ╠═3c59dfa6-bb29-4c6c-8fbc-ff845786c0b9
# ╠═5fa0662b-ae8e-4c7b-a2f9-284a11f4ab50
