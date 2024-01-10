### A Pluto.jl notebook ###
# v0.19.36

#> [frontmatter]
#> title = "Getting Started"
#> category = "Quick Start"

using Markdown
using InteractiveUtils

# ╔═╡ 26b9d680-1004-4440-a392-16c178230066
# ╠═╡ show_logs = false
using Pkg; Pkg.activate("."); Pkg.instantiate()

# ╔═╡ 7cfeb5d8-ceb6-499f-982f-e91fa89a2a3c
using PlutoUI: TableOfContents

# ╔═╡ 1a062240-c8a9-4187-b02c-9d1ffdeddf83
using CairoMakie: Figure, Axis, heatmap, heatmap!, hidedecorations!

# ╔═╡ 0ebe93c9-8278-40e7-b51a-6ea8f37431a8
using Images: Gray, load

# ╔═╡ 73978188-b49d-455c-986c-bb149236745d
using ImageMorphology: distance_transform, feature_transform

# ╔═╡ 59fcb1fc-c439-4027-b443-2ddae5a577fd
using DistanceTransforms: boolean_indicator, transform

# ╔═╡ 86243827-7f59-4230-bd3d-3d3edb6b2958
md"""
# DistanceTransforms.jl

## Why This Library?

| | DistanceTransforms.jl | [ImageMorphology.jl](https://github.com/JuliaImages/ImageMorphology.jl/blob/master/src/feature_transform.jl) | [SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.distance_transform_edt.html) |
|-----------------------|:---------------------:|:---------------:|:---------------:|
| Fast Distance Transform     | ✅✅ | ✅ | ✅ |
| CPU Single-Threaded Support | ✅ | ✅ | ✅ |
| CPU Multi-Threaded Support  | ✅ | ✅ | ❌ |
| NVIDIA/CUDA Support         | ✅ | ❌ | ❌ |
| AMD/ROCM Support            | ✅ | ❌ | ❌ |
| Apple/Metal Support         | ✅ | ❌ | ❌ |
| Comprehensive Documentation | ✅✅ | ❌ | ✅ |

## Set up
To get started, we first need to set up our Julia environment. This includes activating the current project environment, ensuring all necessary packages are installed and up-to-date. We will then load the essential libraries required for this exploration. These libraries not only include DistanceTransforms.jl but also additional packages for visualization, benchmarking, and handling GPU functionality.

For local use, install the necessary packages as follows:
```julia
begin
    using Pkg; Pkg.activate(temp = true)
    Pkg.add(["PlutoUI", "CairoMakie", "Images", "ImageMorphology"])
    Pkg.add(url = "https://github.com/Dale-Black/DistanceTransforms.jl")
end
```
"""

# ╔═╡ c579c92a-3d1c-4213-82c5-cca54b5c0144
TableOfContents()

# ╔═╡ 903bf2fa-de7b-420f-a6ff-381ea6312287
md"""
# Introduction
Distance transforms play a crucial role in many computer vision tasks. This section will demonstrate how DistanceTransforms.jl facilitates efficient distance transform operations on arrays in Julia.

## Basic Usage
The primary function in DistanceTransforms.jl is `transform`. This function processes an array of 0s and 1s, converting each background element (0) into a value representing its squared Euclidean distance to the nearest foreground element (1). 

## Example
Let's begin with a basic example:
"""

# ╔═╡ 862747e5-e88f-47ab-bc97-f5a456738b22
arr = rand([0f0, 1f0], 10, 10)

# ╔═╡ b9ba3bde-8779-4f3c-a3fb-ef157e121632
transform(boolean_indicator(arr))

# ╔═╡ 73cfc148-5af4-4ad7-a8ff-7520c891564b
md"""
## Visualization
Visualization aids in understanding the effects of the distance transform. We use Makie.jl for this purpose:
"""

# ╔═╡ d3498bb0-6f8f-470a-b1de-b1760229cc1c
heatmap(arr, colormap=:grays)

# ╔═╡ ec3a90f1-f2d7-4d5a-8785-476073ee0079
heatmap(transform(boolean_indicator(arr)); colormap=:grays)

# ╔═╡ cd89f08c-44b4-43f6-9313-b4737725b39a
md"""
# Intermediate Usage

We load an example image from the Images.jl library to demonstrate a distance transform applied to a real-world scenario.
"""

# ╔═╡ 158ef85a-28ef-4756-8120-53450f2ef7cd
img = load(Base.download("http://docs.opencv.org/3.1.0/water_coins.jpg"));

# ╔═╡ cf258b9d-e6bf-4128-b669-c8a5b91ecdef
img_bw = Gray.(img) .> 0.5; # theshold image

# ╔═╡ 361835a4-eda2-4eed-a249-4c6cce212662
img_tfm = transform(boolean_indicator(img_bw));

# ╔═╡ 6dbbdb71-5d86-4613-8900-95da2f40143e
md"""
## Visualization
"""

# ╔═╡ 80cbd0a5-5be1-4e7b-b54e-007f4ad0fb7d
let
	f = Figure(size = (700, 600))
	ax = Axis(
		f[1:2, 1],
		title = "Original Image",
	)
	heatmap!(rotr90(img); colormap = :grays)
	hidedecorations!(ax)

	ax = Axis(
		f[3:4, 1],
		title = "Segmented Image",
	)
	heatmap!(rotr90(img_bw); colormap = :grays)
	hidedecorations!(ax)

	ax = Axis(
		f[2:3, 2],
		title = "Distance Transformed Image",
	)
	heatmap!(rotr90(img_tfm); colormap = :grays)
	hidedecorations!(ax)

	f
end

# ╔═╡ 24ff9beb-81c0-4c5c-865e-4446aa4df435
md"""
# Helpful Information

## About the Algorithm
DistanceTransforms.jl implements sophisticated algorithms for both CPU and GPU environments, ensuring efficient computation regardless of the platform.

**CPU**: On the CPU, DistanceTransforms.jl employs the squared Euclidean distance transform algorithm, a method well-documented and respected in computational geometry. This approach, detailed by [Felzenszwalb and Huttenlocher](https://theoryofcomputing.org/articles/v008a019/), is known for its accuracy and efficiency in distance calculations.

**GPU**: For GPU computations, DistanceTransforms.jl uses a custom algorithm, optimized for performance across various GPU architectures. This is facilitated by KernelAbstractions.jl, a Julia package designed for writing hardware-agnostic Julia code. This ensures that DistanceTransforms.jl can leverage the power of GPUs from different vendors, including NVIDIA (CUDA), AMD (ROCM), and Apple (Metal).

**Simplified Interface via Multiple Dispatch**: One of the key features of DistanceTransforms.jl is its simplicity for the end user. Thanks to Julia's multiple dispatch system, the complexity of choosing between CPU and GPU algorithms is abstracted away. Users need only call `transform(boolean_indicator(...))`, and the library intelligently dispatches the appropriate method based on the input's characteristics. This simplification means users can focus on their tasks without worrying about the underlying computational details.

## Note on Euclidean Distance
The library, by default, returns the squared Euclidean distance, as it is often sufficient for many applications and more computationally efficient. However, for cases where the true Euclidean distance is needed, users can easily obtain it by taking the square root of each element in the transformed array. This flexibility allows for a balance between computational efficiency and the specific needs of the application.
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

# ╔═╡ 7597fa37-b406-4fca-8eaa-1627b1eb835d
euc_transform = sqrt.(sq_euc_transform)

# ╔═╡ 239e0053-a16e-4651-b9a0-e163e1bc3892
md"""
## Comparative Example
We will now compare the functionality of DistanceTransforms.jl with ImageMorphology.jl:
"""

# ╔═╡ e49f34f6-2b22-42c4-9c16-e94f8ee2030a
euc_transform2 = distance_transform(feature_transform(Bool.(array2)))

# ╔═╡ 26f4ca31-2e74-4231-9f94-c8bbbbd2fce7
isapprox(euc_transform2, euc_transform; rtol = 1e-2)

# ╔═╡ Cell order:
# ╟─86243827-7f59-4230-bd3d-3d3edb6b2958
# ╠═26b9d680-1004-4440-a392-16c178230066
# ╠═7cfeb5d8-ceb6-499f-982f-e91fa89a2a3c
# ╠═1a062240-c8a9-4187-b02c-9d1ffdeddf83
# ╠═0ebe93c9-8278-40e7-b51a-6ea8f37431a8
# ╠═73978188-b49d-455c-986c-bb149236745d
# ╠═59fcb1fc-c439-4027-b443-2ddae5a577fd
# ╠═c579c92a-3d1c-4213-82c5-cca54b5c0144
# ╟─903bf2fa-de7b-420f-a6ff-381ea6312287
# ╠═862747e5-e88f-47ab-bc97-f5a456738b22
# ╠═b9ba3bde-8779-4f3c-a3fb-ef157e121632
# ╟─73cfc148-5af4-4ad7-a8ff-7520c891564b
# ╟─d3498bb0-6f8f-470a-b1de-b1760229cc1c
# ╟─ec3a90f1-f2d7-4d5a-8785-476073ee0079
# ╟─cd89f08c-44b4-43f6-9313-b4737725b39a
# ╠═158ef85a-28ef-4756-8120-53450f2ef7cd
# ╠═cf258b9d-e6bf-4128-b669-c8a5b91ecdef
# ╠═361835a4-eda2-4eed-a249-4c6cce212662
# ╟─6dbbdb71-5d86-4613-8900-95da2f40143e
# ╟─80cbd0a5-5be1-4e7b-b54e-007f4ad0fb7d
# ╟─24ff9beb-81c0-4c5c-865e-4446aa4df435
# ╠═77de39d7-5d30-410f-baa5-460ae2d8a4a3
# ╠═7c557d32-2bb8-47a2-8216-e18755d258c7
# ╠═d6b06775-2ded-4e15-9616-f3e4653d1396
# ╠═7597fa37-b406-4fca-8eaa-1627b1eb835d
# ╟─239e0053-a16e-4651-b9a0-e163e1bc3892
# ╠═e49f34f6-2b22-42c4-9c16-e94f8ee2030a
# ╠═26f4ca31-2e74-4231-9f94-c8bbbbd2fce7
