### A Pluto.jl notebook ###
# v0.19.32

#> [frontmatter]
#> title = "Getting Started"
#> category = "Quick Start"

using Markdown
using InteractiveUtils

# ╔═╡ 26b9d680-1004-4440-a392-16c178230066
# ╠═╡ show_logs = false
begin
	using Pkg
	Pkg.activate(".")
	Pkg.instantiate()
	
	using PlutoUI, CairoMakie, BenchmarkTools, Images, TestImages, ImageMorphology
	using DistanceTransforms
end

# ╔═╡ c72ed485-d53d-4586-9b9c-e8a4d95f5bf1
md"""
# Getting Started
"""

# ╔═╡ 32a932f4-cade-434f-a489-282bb909a04c
md"""
## Import Packages
First, let's import the most up-to-date version of DistanceTransforms.jl, which can be found on the main/master branch of the [GitHub repository](https://github.com/Dale-Black/DistanceTransforms.jl). 
Because we are using the unregistered version (most recent) we will need to `Pkg.add` this explicitly, without using Pluto's built-in package manager. Be aware, this can take a long time, especially if this is the first time being downloaded. Future work on this package will focus on improving this.
To help with the formatting of this documentation we will also add [PlutoUI.jl](https://github.com/JuliaPluto/PlutoUI.jl).
Lastly, to visualize results and time the functions were going to add [Makie.jl](https://github.com/JuliaPlots/Makie.jl) and [BenchmarkTools.jl](https://github.com/JuliaCI/BenchmarkTools.jl).
"""

# ╔═╡ bb5f71db-9999-4104-9b1a-03a32cea035e
md"""
!!! info "Local Usage"
	If you are running this notebook locally, you will want to install the necessary packages like so:
	
	```julia
	begin
		using Pkg; Pkg.activate(temp = true)
		Pkg.add(["PlutoUI", "CairoMakie", "BenchmarkTools", "Images", "TestImages"])
		Pkg.add(url = "https://github.com/Dale-Black/DistanceTransforms.jl")
	
		using PlutoUI, CairoMakie, BenchmarkTools, Images, TestImages
		using DistanceTransforms
	end
	```
"""

# ╔═╡ c579c92a-3d1c-4213-82c5-cca54b5c0144
TableOfContents()

# ╔═╡ 1b06308d-4084-4be8-9758-8052277e5ac1
md"""
## Introduction
Distance transforms are an important part of many computer vision-related tasks. Let's create some sample data and see how DistanceTransforms.jl can be used to apply efficient distance transform operations on arrays in Julia.
"""

# ╔═╡ 97aad592-8362-48ce-87da-5ab3e8eb6f95
md"""
The quintessential distance transform operation in DistanceTransforms.jl is just `transform`. This takes an array of 0s and 1s and creates a new array, where every background element (0) is replaced with a number that corresponds to the distance to the nearest foreground element. Let's see this in action:
"""

# ╔═╡ 862747e5-e88f-47ab-bc97-f5a456738b22
arr = rand([0f0, 1f0], 20, 20)

# ╔═╡ b9ba3bde-8779-4f3c-a3fb-ef157e121632
transform(boolean_indicator(arr))

# ╔═╡ b2d06446-991c-4311-993f-87ddef5e8828
md"""
As you can see, each element in the array is either zero (corresponding to background), which remains zero, or is one (corresponding to foreground) that then gets replaced by the squared Euclidean distance to the nearest zero. 
We can see this easily using Makie.jl.
"""

# ╔═╡ d3498bb0-6f8f-470a-b1de-b1760229cc1c
heatmap(arr, colormap=:grays)

# ╔═╡ ec3a90f1-f2d7-4d5a-8785-476073ee0079
heatmap(transform(boolean_indicator(arr)); colormap=:grays)

# ╔═╡ cd89f08c-44b4-43f6-9313-b4737725b39a
md"""
## Intermediate Usage

Now, let's load an example image from the Images.jl library and see how a distane transform can be used (and visualized) on the sample image.
"""

# ╔═╡ 158ef85a-28ef-4756-8120-53450f2ef7cd
img = load(download("http://docs.opencv.org/3.1.0/water_coins.jpg"));

# ╔═╡ cf258b9d-e6bf-4128-b669-c8a5b91ecdef
img_bw = Gray.(img) .> 0.5; # theshold image

# ╔═╡ 361835a4-eda2-4eed-a249-4c6cce212662
img_tfm = transform(boolean_indicator(img_bw));

# ╔═╡ 80cbd0a5-5be1-4e7b-b54e-007f4ad0fb7d
let
	f = Figure(resolution = (1000, 1000))
	ax = CairoMakie.Axis(
		f[1:2, 1],
		title = "Original Image",
	)
	heatmap!(rotr90(img); colormap = :grays)
	hidedecorations!(ax)

	ax = CairoMakie.Axis(
		f[3:4, 1],
		title = "Segmented Image",
	)
	heatmap!(rotr90(img_bw); colormap = :grays)
	hidedecorations!(ax)

	ax = CairoMakie.Axis(
		f[2:3, 2],
		title = "Distance Transformed Image",
	)
	heatmap!(rotr90(img_tfm); colormap = :grays)
	hidedecorations!(ax)

	f
end

# ╔═╡ 2e06c9b5-1632-48ee-a045-bcdada5e8f91
md"""
## Helpful Information
"""

# ╔═╡ bf546f09-7604-4f11-8676-34a7ac571780
md"""
!!! info
	This algorithm is based on the squared Euclidean distance transform, as described by [Felzenszwalb and
	Huttenlocher] (DOI: 10.4086/toc.2012.v008a019).

	We will use a similar approach to get started, with one extra step; the array must first be passed through the `boolean_indicator()` function. This transforms the 0s and 1s to `Inf`s and `0`s. Specifically `0 => 1f10` and `1 => 0f0`.
	
	*Aside: `0f0` is the `Float32` version of `0.0`*


	It should be clear to see that the transform returns the Euclidean distance, just squared. To find the true Euclidean distance, all one must do to get the true Euclidean distance take the square root of each element. In many cases, this is not necessary and can add time, which is why it is left optional.
"""

# ╔═╡ 77de39d7-5d30-410f-baa5-460ae2d8a4a3
array2 = [
	0 1 1 0 1
	0 0 0 1 0
	1 1 0 0 0
]

# ╔═╡ 7c557d32-2bb8-47a2-8216-e18755d258c7
array2_bool = boolean_indicator(array2)

# ╔═╡ d6b06775-2ded-4e15-9616-f3e4653d1396
sq_euc_transform = transform(array2_bool)

# ╔═╡ 679125bf-af0e-4046-889e-7d58738b4ccc
md"""
!!! info
	It should be clear to see that the transform returns the Euclidean distance, just squared. To find the true Euclidean distance, all one must do to get the true Euclidean distance take the square root of each element. In many cases, this is not necessary and can add time, which is why it is left optional.
"""

# ╔═╡ 7597fa37-b406-4fca-8eaa-1627b1eb835d
euc_transform = sqrt.(sq_euc_transform)

# ╔═╡ 5c1a062d-d878-4540-b493-35498b36cb17
md"""
!!! info
	Now, we can also perform a distance transform operation using the ImageMorphology.jl. Let's do that now and show that the results are equivalent when using ImageMorphology's `distance_transform(feature_transform(...))` or DistanceTransform's `transform(feature_transform(...))`. Later, we will see some benefits to using DistanceTransforms.jl approach.
"""

# ╔═╡ e49f34f6-2b22-42c4-9c16-e94f8ee2030a
euc_transform2 = distance_transform(feature_transform(Bool.(array2)))

# ╔═╡ 26f4ca31-2e74-4231-9f94-c8bbbbd2fce7
isapprox(euc_transform2, euc_transform; rtol = 1e-2)

# ╔═╡ d167e25b-b1b1-46ef-a1cb-fd0856c2c685
md"""
!!! info
	The following notebooks will showcase some of the usefulness of DistanceTransforms.jl. Keep reading to see how this can be useful in deep learning and the like.
"""

# ╔═╡ Cell order:
# ╟─c72ed485-d53d-4586-9b9c-e8a4d95f5bf1
# ╟─32a932f4-cade-434f-a489-282bb909a04c
# ╠═26b9d680-1004-4440-a392-16c178230066
# ╟─bb5f71db-9999-4104-9b1a-03a32cea035e
# ╠═c579c92a-3d1c-4213-82c5-cca54b5c0144
# ╟─1b06308d-4084-4be8-9758-8052277e5ac1
# ╟─97aad592-8362-48ce-87da-5ab3e8eb6f95
# ╠═862747e5-e88f-47ab-bc97-f5a456738b22
# ╠═b9ba3bde-8779-4f3c-a3fb-ef157e121632
# ╟─b2d06446-991c-4311-993f-87ddef5e8828
# ╟─d3498bb0-6f8f-470a-b1de-b1760229cc1c
# ╟─ec3a90f1-f2d7-4d5a-8785-476073ee0079
# ╟─cd89f08c-44b4-43f6-9313-b4737725b39a
# ╠═158ef85a-28ef-4756-8120-53450f2ef7cd
# ╠═cf258b9d-e6bf-4128-b669-c8a5b91ecdef
# ╠═361835a4-eda2-4eed-a249-4c6cce212662
# ╟─80cbd0a5-5be1-4e7b-b54e-007f4ad0fb7d
# ╟─2e06c9b5-1632-48ee-a045-bcdada5e8f91
# ╟─bf546f09-7604-4f11-8676-34a7ac571780
# ╠═77de39d7-5d30-410f-baa5-460ae2d8a4a3
# ╠═7c557d32-2bb8-47a2-8216-e18755d258c7
# ╠═d6b06775-2ded-4e15-9616-f3e4653d1396
# ╟─679125bf-af0e-4046-889e-7d58738b4ccc
# ╠═7597fa37-b406-4fca-8eaa-1627b1eb835d
# ╟─5c1a062d-d878-4540-b493-35498b36cb17
# ╠═e49f34f6-2b22-42c4-9c16-e94f8ee2030a
# ╠═26f4ca31-2e74-4231-9f94-c8bbbbd2fce7
# ╟─d167e25b-b1b1-46ef-a1cb-fd0856c2c685
