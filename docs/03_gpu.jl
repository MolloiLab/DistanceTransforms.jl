### A Pluto.jl notebook ###
# v0.19.11

#> [frontmatter]
#> title = "GPU"
#> category = "Tutorials"

using Markdown
using InteractiveUtils

# ╔═╡ 22cec81f-c503-43de-8297-ec0591e4e6c5
# ╠═╡ show_logs = false
begin
	let
		using Pkg
		Pkg.activate(mktempdir())
		Pkg.Registry.update()
		Pkg.add("PlutoUI")
		Pkg.add("CairoMakie")
		Pkg.add("BenchmarkTools")
		Pkg.add("CUDA")
		Pkg.add(url="https://github.com/Dale-Black/DistanceTransforms.jl")
	end
	
	using PlutoUI
	using CairoMakie
	using BenchmarkTools
	using CUDA
	using DistanceTransforms
end

# ╔═╡ ed7fea36-60ca-47e0-9046-bf2565fdd41d
md"""
## Import packages
"""

# ╔═╡ afeb0802-60c1-40e0-a9e5-b13ad6af27bc
TableOfContents()

# ╔═╡ b370e3a1-37ec-48aa-86e4-d053bf9bce3b
md"""
## GPU enabled distance transforms
One of the advantages of a Julia-based distance transform library is how accessible something like GPU programming is for complex algorithms. The `SquaredEuclidean` distance transform is highly parallelizable and is set up to take advantage of the [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl) library and GPU hardware.
Unfortunately, [glassnotebook.io](https://glassnotebook.io) does not yet support GPU-compatible static exports, so this documentation, as it stands currently, only shows how to set up a GPU-based distance transform, without any execution.
The code is copy-pastable, though, for anyone interested in testing it out on their own GPU-enabled computer.
"""

# ╔═╡ b885869b-985e-4d73-acd9-b381e6272540
if has_cuda_gpu()
	array = CUDA.CuArray(
		boolean_indicator(
			[
				0 1 1 1 0
				1 1 1 1 1
				1 0 0 0 1
				1 0 0 0 1
				1 0 0 0 1
				1 1 1 1 1
				0 1 1 1 0
			]
		)
	)
	
	dt = CuArray{Float32}(undef, size(array))
	v = CUDA.ones(Int64, size(array))
	z = CUDA.zeros(Float32, size(array) .+ 1)
	tfm = Felzenszwalb()
	
	transform!(array, tfm; output=dt, v=v, z=z)
else
	array = boolean_indicator(
		[
				0 1 1 1 0
				1 1 1 1 1
				1 0 0 0 1
				1 0 0 0 1
				1 0 0 0 1
				1 1 1 1 1
				0 1 1 1 0
		])
	
	dt = Array{Float32}(undef, size(array))
	v = ones(Int64, size(array))
	z = zeros(Float32, size(array) .+ 1)
	tfm = Felzenszwalb()
	
	transform!(array, tfm; output=dt, v=v, z=z)
end

# ╔═╡ 8488fa03-8ad4-4b81-8877-4ec75018a84f
md"""
## GPU-enabled distance transform use cases
Distance transforms are ubiquitous in the fields of computer vision and image processing. From object recognition and path planning to deep learning and image segmentation, distance transforms are increasingly useful. DistanceTransforms.jl was created to give developers a simple, consistent API for implementing various distance transforms and to give end-users a seamless way to utilize distance transforms for various tasks, especially GPU-related tasks.
One such example can be seen in the next tutorial on using distance transforms within various [loss functions](https://glassnotebook.io/nb/119)
"""

# ╔═╡ Cell order:
# ╟─ed7fea36-60ca-47e0-9046-bf2565fdd41d
# ╠═22cec81f-c503-43de-8297-ec0591e4e6c5
# ╠═afeb0802-60c1-40e0-a9e5-b13ad6af27bc
# ╟─b370e3a1-37ec-48aa-86e4-d053bf9bce3b
# ╠═b885869b-985e-4d73-acd9-b381e6272540
# ╟─8488fa03-8ad4-4b81-8877-4ec75018a84f
