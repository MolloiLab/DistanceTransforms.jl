### A Pluto.jl notebook ###
# v0.19.11

using Markdown
using InteractiveUtils

# ╔═╡ d25a627e-23ee-11ed-1688-ef6ca75e19e4
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

# ╔═╡ f26d651f-6abd-4d25-a1b1-89ac9b30a316
TableOfContents()

# ╔═╡ 15083759-bf1a-4f94-8e29-4e04feb9a89f
md"""
# `euclidean`

```julia
euclidean(img)
euclidean(img::BitArray)
```

Wrapper function for `ImageMorphology.feature_transform` and `ImageMorphology.distance_transform`. Applies a true Euclidean distance transform to the array elements and returns an array with spatial information embedded in the elements.

Arguments
- img: N-dimensional array to be transformed based on location to the nearest background (0) pixel

Citation
- 'A Linear Time Algorithm for Computing Exact Euclidean Distance Transforms of Binary Images in Arbitrary Dimensions' [Maurer et al., 2003] (DOI: 10.1109/TPAMI.2003.1177156)
"""

# ╔═╡ 5db2a6bd-8cf5-41ed-8cfd-bc96b6563d24
euclidean(img) = distance_transform(feature_transform(Bool.(img)))

# ╔═╡ 44e7843c-fc9b-4368-b455-7ffded09ffcc
euclidean(img::BitArray) = distance_transform(feature_transform(img))

# ╔═╡ Cell order:
# ╠═d25a627e-23ee-11ed-1688-ef6ca75e19e4
# ╠═f26d651f-6abd-4d25-a1b1-89ac9b30a316
# ╟─15083759-bf1a-4f94-8e29-4e04feb9a89f
# ╠═5db2a6bd-8cf5-41ed-8cfd-bc96b6563d24
# ╠═44e7843c-fc9b-4368-b455-7ffded09ffcc
