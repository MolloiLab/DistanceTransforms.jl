### A Pluto.jl notebook ###
# v0.19.11

using Markdown
using InteractiveUtils

# ╔═╡ 5a9c65be-a98c-4eeb-bc3d-c98f8163e563
# ╠═╡ show_logs = false
begin
	using Pkg
	Pkg.activate(".")
	using Revise
	using PlutoUI
	using DistanceTransforms
	using CairoMakie
	using BenchmarkTools
	using CUDA
end

# ╔═╡ e3f4e1de-8786-4d64-96a9-cf8e58a56506
md"""
## Import packages
"""

# ╔═╡ bd75cf9e-f2ea-4a68-87bc-06cfb4b47f53
TableOfContents()

# ╔═╡ ccea0e93-78ec-436a-86ae-2de3f3186eb1
md"""
## GPU enabled distance transforms
One of the advantages of a Julia-based distance transform library is how accessible something like GPU programming is for complex algorithms. The `SquaredEuclidean` distance transform is highly parallelizable and is set up to take advantage of the [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl) library and GPU hardware.

Unfortunately, [glassnotebook.io](https://glassnotebook.io) does not yet support GPU-compatible static exports, so this documentation, as it stands currently, only shows how to set up a GPU-based distance transform, without any execution.

The code is copy-pastable, though, for anyone interested in testing it out on their own GPU-enabled computer.
"""

# ╔═╡ 88adf4ef-08c0-4550-b729-3eb9b2c9c3db
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
	tfm = DistanceTransforms.SquaredEuclidean()
	
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
	tfm = DistanceTransforms.SquaredEuclidean()
	
	transform!(array, tfm; output=dt, v=v, z=z)
end

# ╔═╡ 16ca7193-4485-4259-a69d-88369a1d351d
md"""
## GPU-enabled distance transform use cases

Distance transforms are ubiquitous in the fields of computer vision and image processing. From object recognition and path planning to deep learning and image segmentation, distance transforms are increasingly useful. DistanceTransforms.jl was created to give developers a simple, consistent API for implementing various distance transforms and to give end-users a seamless way to utilize distance transforms for various tasks, especially GPU-related tasks.

One such example can be seen in the next tutorial on using distance transforms within various [loss functions](link...)
"""

# ╔═╡ Cell order:
# ╠═5a9c65be-a98c-4eeb-bc3d-c98f8163e563
# ╟─e3f4e1de-8786-4d64-96a9-cf8e58a56506
# ╠═bd75cf9e-f2ea-4a68-87bc-06cfb4b47f53
# ╟─ccea0e93-78ec-436a-86ae-2de3f3186eb1
# ╠═88adf4ef-08c0-4550-b729-3eb9b2c9c3db
# ╟─16ca7193-4485-4259-a69d-88369a1d351d
