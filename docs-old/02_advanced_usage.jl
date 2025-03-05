### A Pluto.jl notebook ###
# v0.19.36

#> [frontmatter]
#> title = "Advanced Usage"

using Markdown
using InteractiveUtils

# ╔═╡ e7b64851-7108-4c14-9710-c991d5b7021a
# ╠═╡ show_logs = false
using Pkg; Pkg.activate("."); Pkg.instantiate()

# ╔═╡ a3966df9-1107-4f2b-9fb0-704ad7dbe26a
using PlutoUI: TableOfContents

# ╔═╡ ab982cb6-d37c-4d27-a695-d3d8378ab32b
using DistanceTransforms: transform, boolean_indicator

# ╔═╡ c79a8a06-262c-4733-aa31-a0e9a896fafb
using CairoMakie: Figure, Axis, Label, Legend, scatterlines!

# ╔═╡ 1a7c3ed0-7453-4f53-ac5a-99d0b3f84a85
# ╠═╡ show_logs = false
using DistanceTransformsPy: pytransform

# ╔═╡ fe295739-2a9f-46a5-92b9-19e5b5b5dc50
using ImageMorphology: distance_transform, feature_transform

# ╔═╡ b4125c21-43cf-42a5-b685-bc47b8b1f8b8
using BenchmarkTools

# ╔═╡ 53545152-a6ca-4f13-8275-95a8a04017c0
using KernelAbstractions

# ╔═╡ 197c25db-3cc1-4b65-bf49-f0f27694bcc3
# ╠═╡ show_logs = false
using CUDA, AMDGPU, Metal

# ╔═╡ c80da3e8-ff14-4f8c-9a67-b1601395ff40
using FileIO: save, load

# ╔═╡ 481bdd21-744e-424e-992b-b3e19c4c06f7
using ImageShow

# ╔═╡ dec905ec-5ca9-4ef9-b0a6-2a8d31f3f5c7
md"""
# Introduction

Welcome to this advanced exploration of the DistanceTransforms.jl library. In this notebook, we delve into the sophisticated capabilities of DistanceTransforms.jl, focusing on its multi-threading and GPU acceleration features. These advanced functionalities are crucial for enhancing performance in complex Julia applications, particularly in the fields of image processing and computer vision. 

We will benchmark these features to demonstrate the significant performance gains they offer. This guide is tailored for users who are already acquainted with the basic aspects of DistanceTransforms.jl and are looking to exploit its full potential in demanding computational tasks.

## Setup

To get started, we first need to set up our Julia environment. This includes activating the current project environment, ensuring all necessary packages are installed and up-to-date. We will then load the essential libraries required for this exploration. These libraries not only include DistanceTransforms.jl but also additional packages for visualization, benchmarking, and handling GPU functionality.

With our environment ready, we can now proceed to explore the advanced features of DistanceTransforms.jl.

"""

# ╔═╡ 0e6b49a2-7590-4800-871e-6a2b0019a9f3
TableOfContents()

# ╔═╡ 9dd1c30d-3472-492f-a06c-84777f1aabf4
md"""
# Multi-threading

DistanceTransforms.jl efficiently utilizes multi-threading, particularly in its Felzenszwalb distance transform algorithm. This parallelization significantly enhances performance, especially for large data sets and high-resolution images.

## Multi-threading Benefits
With Julia's multi-threading support, the library can concurrently process distance transform operations, automatically optimizing performance based on the system's capabilities. This leads to faster execution times and increased efficiency in computations, reducing the complexity for the users.

## Practical Advantages
This feature is highly beneficial in scenarios requiring rapid processing, such as real-time computer vision applications and large-scale image analysis. It allows users to focus more on application development rather than on managing computational workloads.

Next, we will examine benchmarks demonstrating these performance improvements in various use cases.

"""

# ╔═╡ 9ff0aadf-80b1-4cfd-b3a1-33fbcbc3fb1a
x = boolean_indicator(rand([0f0, 1f0], 100, 100));

# ╔═╡ a6a3ce15-251c-4edf-8a85-d80cec6f4b3f
single_threaded = @benchmark transform($x; threaded = false)

# ╔═╡ a0785853-9091-4fb8-a23c-536032edee74
multi_threaded = @benchmark transform($x; threaded = true)

# ╔═╡ b8daf192-a46e-4b49-b09b-04f793c3b8cc
md"""
# GPU Acceleration

DistanceTransforms.jl extends its performance capabilities by embracing GPU acceleration. This section explores how GPU support is integrated into the library, offering substantial performance enhancements, particularly for large-scale computations.

One of the features of DistanceTransforms.jl is its GPU compatibility, achieved through Julia's multiple dispatch. This means that the same `transform` function used for CPU computations automatically adapts to leverage GPU resources when available.

Multiple dispatch in Julia allows the `transform` function to intelligently determine the computing resource to utilize, based on the input array's type. When a GPU-compatible array type (like `CUDA.CuArray`) is passed, DistanceTransforms.jl automatically dispatches a GPU-optimized version of the algorithm.

## GPU Execution Example
Below is an illustrative example of how GPU acceleration can be employed in DistanceTransforms.jl:

```julia
x_gpu = CUDA.CuArray(boolean_indicator(rand([0, 1], 100, 100)))

# The `transform` function recognizes the GPU array and uses GPU for computations
gpu_transformed = transform(x_gpu)
```

In this example, the `transform` function identifies `x_gpu` as a GPU array and thus, invokes the GPU-accelerated version of the distance transform algorithm.

## Advantages of GPU
The use of GPU acceleration in DistanceTransforms.jl is particularly advantageous for handling large data sets where the parallel computing capabilities of GPUs can be fully utilized. This results in significantly reduced computation times and increased efficiency, making DistanceTransforms.jl a suitable choice for high-performance computing tasks.

In the following sections, we'll delve into benchmark comparisons to demonstrate the efficacy of GPU acceleration in DistanceTransforms.jl.

"""

# ╔═╡ 1f1867f0-05a4-4071-8f7d-9e238a8c940f
md"""
# Benchmarks
"""

# ╔═╡ 595dbaea-a238-44e5-b22a-57c423c1f6ac
if CUDA.functional()
	@info "Using CUDA"
	CUDA.versioninfo()
	backend = CUDABackend()
	dev = CuArray
elseif AMDGPU.functional()
	@info "Using AMD"
	AMDGPU.versioninfo()
	backend = ROCBackend()
	dev = ROCArray
elseif Metal.functional()
	@info "Using Metal"
	Metal.versioninfo()
	backend = MetalBackend()
	dev = MtlArray
else
    @info "No GPU is available. Using CPU."
	backend = CPU()
	dev = Array
end

# ╔═╡ be5b4e09-2a6d-472d-9790-623ae3e99bb8
Threads.nthreads()

# ╔═╡ c1e55d37-d0c9-4da2-8841-89a714e50c21
begin
	range_size_2D = range(4, 1000, 4)
	range_size_3D = range(4, 100, 4)
end

# ╔═╡ 90142aaa-8a1f-4221-aadf-ad858b05f716
md"""
## 2D
"""

# ╔═╡ f0ca70f4-ab9b-48ae-b701-cf4ac47b7bf4
# ╠═╡ disabled = true
#=╠═╡
begin
	sizes = Float64[]
	
	dt_scipy = Float64[]
	dt_scipy_std = Float64[]

	dt_maurer = Float64[]
	dt_maurer_std = Float64[]

	dt_fenz = Float64[]
	dt_fenz_std = Float64[]

	dt_proposed = Float64[]
	dt_proposed_std = Float64[]
	
	for _n in range_size_2D
		n = round(Int, _n)
		@info n
		push!(sizes, n^2)
		f = Float32.(rand([0, 1], n, n))
		
		# Python (Scipy)
		dt = @benchmark pytransform($f)
		push!(dt_scipy, BenchmarkTools.minimum(dt).time)
		push!(dt_scipy_std, BenchmarkTools.std(dt).time)

		# Maurer (ImageMorphology.jl)
		bool_f = Bool.(f)
		dt = @benchmark distance_transform($feature_transform($bool_f))
		push!(dt_maurer, BenchmarkTools.minimum(dt).time) # ns
		push!(dt_maurer_std, BenchmarkTools.std(dt).time)
		
		# Felzenszwalb (DistanceTransforms.jl)
		dt = @benchmark transform($boolean_indicator($f))
		push!(dt_fenz, BenchmarkTools.minimum(dt).time)
		push!(dt_fenz_std, BenchmarkTools.std(dt).time)

		# Proposed-GPU (DistanceTransforms.jl)
		if dev != Array
			f_gpu = dev(f)
			dt = @benchmark transform($boolean_indicator($f_gpu))
			push!(dt_proposed, BenchmarkTools.minimum(dt).time)
			push!(dt_proposed_std, BenchmarkTools.std(dt).time)
		end
	end
end
  ╠═╡ =#

# ╔═╡ 9d94d434-67c7-41fa-b8fc-3c2ce76c22d0
md"""
## 3D
"""

# ╔═╡ cbf39927-96e7-48f0-9812-7a24c3ce5cb4
# ╠═╡ disabled = true
#=╠═╡
begin
	sizes_3D = Float64[]
	
	dt_scipy_3D = Float64[]
	dt_scipy_std_3D = Float64[]

	dt_maurer_3D = Float64[]
	dt_maurer_std_3D = Float64[]

	dt_fenz_3D = Float64[]
	dt_fenz_std_3D = Float64[]

	dt_proposed_3D = Float64[]
	dt_proposed_std_3D = Float64[]
	
	for _n in range_size_3D
		n = round(Int, _n)
		@info n
		push!(sizes_3D, n^3)
		f = Float32.(rand([0, 1], n, n, n))
		
		# Python (Scipy)
		dt = @benchmark pytransform($f)
		push!(dt_scipy_3D, BenchmarkTools.minimum(dt).time)
		push!(dt_scipy_std_3D, BenchmarkTools.std(dt).time)

		# Maurer (ImageMorphology.jl)
		bool_f = Bool.(f)
		dt = @benchmark distance_transform($feature_transform($bool_f))
		push!(dt_maurer_3D, BenchmarkTools.minimum(dt).time) # ns
		push!(dt_maurer_std_3D, BenchmarkTools.std(dt).time)
		
		# Felzenszwalb (DistanceTransforms.jl)
		dt = @benchmark transform($boolean_indicator($f))
		push!(dt_fenz_3D, BenchmarkTools.minimum(dt).time)
		push!(dt_fenz_std_3D, BenchmarkTools.std(dt).time)

		# Proposed-GPU (DistanceTransforms.jl)
		if dev != Array
			f_gpu = dev(f)
			dt = @benchmark transform($boolean_indicator($f_gpu))
			push!(dt_proposed_3D, BenchmarkTools.minimum(dt).time)
			push!(dt_proposed_std_3D, BenchmarkTools.std(dt).time)
		end
	end
end
  ╠═╡ =#

# ╔═╡ 969ddeaa-8ffc-4f7d-8aa6-47b201a8376d
md"""
## Results
"""

# ╔═╡ 104ff068-ad07-4f94-8f3b-18c3660353c1
#=╠═╡
let
	f = Figure()
	ax = Axis(
		f[1, 1],
		title="2D"
	)

	scatterlines!(dt_scipy, label="Python")
	scatterlines!(dt_maurer, label="Maurer")
	scatterlines!(dt_fenz, label="Felzenszwalb")
	scatterlines!(dt_proposed, label="GPU")

	ax = Axis(
		f[2, 1],
		title="3D"
	)
	scatterlines!(dt_scipy_3D, label="Python")
	scatterlines!(dt_maurer_3D, label="Maurer")
	scatterlines!(dt_fenz_3D, label="Felzenszwalb")
	scatterlines!(dt_proposed_3D, label="GPU")

	f[1:2, 2] = Legend(f, ax, "Distance Transforms", framevisible = false)

	Label(f[0, 1:2]; text="Distance Transforms", fontsize=30)

	# save("gpu_fig.png", f)
	f
end
  ╠═╡ =#

# ╔═╡ 9a5245de-2d04-4d0b-abab-011e2bd5a980
md"""
## GPU Results

Since Glass Notebook does not yet support GPU exports and to save resources, I saved a version of the benchmark results that include the GPU accelerated version (METAL) from my M1 Macbook Air which can be seen below.

If you want to run these benchmarks on your own hardware, just simply run this notebook and enable all of the above cells
"""

# ╔═╡ 29834554-0f91-4542-95e4-afdb085a74b8
load("gpu_fig.png")

# ╔═╡ Cell order:
# ╟─dec905ec-5ca9-4ef9-b0a6-2a8d31f3f5c7
# ╠═e7b64851-7108-4c14-9710-c991d5b7021a
# ╠═a3966df9-1107-4f2b-9fb0-704ad7dbe26a
# ╠═ab982cb6-d37c-4d27-a695-d3d8378ab32b
# ╠═c79a8a06-262c-4733-aa31-a0e9a896fafb
# ╠═1a7c3ed0-7453-4f53-ac5a-99d0b3f84a85
# ╠═fe295739-2a9f-46a5-92b9-19e5b5b5dc50
# ╠═b4125c21-43cf-42a5-b685-bc47b8b1f8b8
# ╠═53545152-a6ca-4f13-8275-95a8a04017c0
# ╠═197c25db-3cc1-4b65-bf49-f0f27694bcc3
# ╠═c80da3e8-ff14-4f8c-9a67-b1601395ff40
# ╠═481bdd21-744e-424e-992b-b3e19c4c06f7
# ╠═0e6b49a2-7590-4800-871e-6a2b0019a9f3
# ╟─9dd1c30d-3472-492f-a06c-84777f1aabf4
# ╠═9ff0aadf-80b1-4cfd-b3a1-33fbcbc3fb1a
# ╠═a6a3ce15-251c-4edf-8a85-d80cec6f4b3f
# ╠═a0785853-9091-4fb8-a23c-536032edee74
# ╟─b8daf192-a46e-4b49-b09b-04f793c3b8cc
# ╟─1f1867f0-05a4-4071-8f7d-9e238a8c940f
# ╠═595dbaea-a238-44e5-b22a-57c423c1f6ac
# ╠═be5b4e09-2a6d-472d-9790-623ae3e99bb8
# ╠═c1e55d37-d0c9-4da2-8841-89a714e50c21
# ╟─90142aaa-8a1f-4221-aadf-ad858b05f716
# ╠═f0ca70f4-ab9b-48ae-b701-cf4ac47b7bf4
# ╟─9d94d434-67c7-41fa-b8fc-3c2ce76c22d0
# ╠═cbf39927-96e7-48f0-9812-7a24c3ce5cb4
# ╟─969ddeaa-8ffc-4f7d-8aa6-47b201a8376d
# ╟─104ff068-ad07-4f94-8f3b-18c3660353c1
# ╟─9a5245de-2d04-4d0b-abab-011e2bd5a980
# ╟─29834554-0f91-4542-95e4-afdb085a74b8
