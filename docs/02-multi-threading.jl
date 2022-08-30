### A Pluto.jl notebook ###
# v0.19.11

using Markdown
using InteractiveUtils

# ╔═╡ b1f28e84-74ed-4f97-bd1c-306a6cb185ec
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

# ╔═╡ ee77bccd-4626-447e-9c1d-8f7dd2b5b87f
md"""
## Import packages
"""

# ╔═╡ 9fabfce6-d8c1-4c31-86e9-44a118477698
TableOfContents()

# ╔═╡ 89e50304-8ba6-44b3-a879-06af10d78fb8
md"""
## Multi-threaded distance transforms
One of the advantages of a Julia-based distance transform library is how accessible something like multi-threading is for complex algorithms. The `SquaredEuclidean` distance transform is highly parallelizable and is set up to take advantage of multi-threaded hardware.
"""

# ╔═╡ 291550e0-c602-4494-abb7-dcb65f30d376
threads = Threads.nthreads()

# ╔═╡ acd9ea9b-3209-48ed-89ca-8cb5f5b5c252
md"""
### 2D
"""

# ╔═╡ 931b8b00-0ca4-45ce-9d5a-31dfb2e16966
array = [
	1 0 1 1 1
	0 0 1 0 1
	1 0 1 0 1
	1 1 0 1 1
]

# ╔═╡ f2a7720a-5656-4a64-90f8-9d12b4a96d8f
bool_array = boolean_indicator(array)

# ╔═╡ 10cb17d2-569c-4603-86cf-bd64ad15608d
tfm = SquaredEuclidean()

# ╔═╡ d0669dd6-fbe1-43a6-98c1-b4fec93f26c2
sq_euc_transform = transform!(bool_array, tfm, threads)

# ╔═╡ 40f59857-1b21-467a-b4f7-f92bc44771e9
md"""
### 3D
"""

# ╔═╡ a4f56343-124e-44d6-a058-f3b092454728
array3D = rand([0, 1], 5, 5, 3)

# ╔═╡ efa13dcd-5175-466e-8045-6275ebffb8b2
sq_euc_transform3D = transform!(boolean_indicator(array3D), tfm, threads)

# ╔═╡ 88681701-2488-4275-b598-066225921256
md"""
## Timing
Let's compare the time difference between the regular `SquaredEuclidean` distance transform algorithm to the multi-threaded algorithm
"""

# ╔═╡ 9057ee19-8a39-4712-bb94-bc951219a732
begin
	sedt_mean = []
	sedt_std = []
	
	sedtP_mean = []
	sedtP_std = []

	sizes = []
	for n in 1:50:200
		_size = n*n
		append!(sizes, _size)
		
		# Squared Euclidean DT
		x2 = boolean_indicator(rand([0, 1], n, n))
		sedt = @benchmark transform($x2, $tfm)
		
		append!(sedt_mean, BenchmarkTools.mean(sedt).time)
		append!(sedt_std, BenchmarkTools.std(sedt).time)
		
		# Squared Euclidean DT threaded
		x3 = boolean_indicator(rand([0, 1], n, n))
		sedtP = @benchmark transform!($x3, $tfm, $threads)
		
		append!(sedtP_mean, BenchmarkTools.mean(sedtP).time)
		append!(sedtP_std, BenchmarkTools.std(sedtP).time)
	end
end

# ╔═╡ 16356cde-1788-4dc3-89b4-9e8ab3177b7b
let
    f = Figure()
    ax1 = Axis(f[1, 1])
	ax1.xlabel = "Number of Elements"
    ax1.ylabel = "Time (ns)"
    ax1.title = "Distance Transforms (2D)"

	
    sc1 = scatter!(ax1, Float64.(sizes), Float64.(sedt_mean))
    # errorbars!(ax1, Float64.(sizes), Float64.(sedt_mean), Float64.(sedt_std))

	sc2 = scatter!(ax1, Float64.(sizes), Float64.(sedtP_mean))
    # errorbars!(ax1, Float64.(sizes), Float64.(sedtP_mean), Float64.(sedtP_std))

	f[1, 2] = Legend(
        f,
        [sc1, sc2],
        ["Squared Euclidean","Squared Euclidean Threaded"];
        framevisible=false,
    )
	f
end

# ╔═╡ 5b8980c6-3aa6-4376-9734-21364613fab2
md"""
As you can see, for small arrays, the multi-threading functionality doesn't offer any improvements in speed (in fact, it limits it). But as the array size increases, the speed improvements also increase by order(s) of magnitude.

One might wonder why, if multi-threading is so beneficial, GPUs wouldn't be utilized. For most libraries, many would answer that question with a demonstration of the added complexity when going from regular programming to GPU programming and why it's not practical, but DistanceTransforms.jl is a pure Julia package. This means that GPUs are readily available with minimal code change, so the next [tutorial](link ...) will showcase this exact use case and why both end-users and developers might benefit from this.
"""

# ╔═╡ Cell order:
# ╠═b1f28e84-74ed-4f97-bd1c-306a6cb185ec
# ╟─ee77bccd-4626-447e-9c1d-8f7dd2b5b87f
# ╠═9fabfce6-d8c1-4c31-86e9-44a118477698
# ╟─89e50304-8ba6-44b3-a879-06af10d78fb8
# ╠═291550e0-c602-4494-abb7-dcb65f30d376
# ╟─acd9ea9b-3209-48ed-89ca-8cb5f5b5c252
# ╠═931b8b00-0ca4-45ce-9d5a-31dfb2e16966
# ╠═f2a7720a-5656-4a64-90f8-9d12b4a96d8f
# ╠═10cb17d2-569c-4603-86cf-bd64ad15608d
# ╠═d0669dd6-fbe1-43a6-98c1-b4fec93f26c2
# ╟─40f59857-1b21-467a-b4f7-f92bc44771e9
# ╠═a4f56343-124e-44d6-a058-f3b092454728
# ╠═efa13dcd-5175-466e-8045-6275ebffb8b2
# ╟─88681701-2488-4275-b598-066225921256
# ╠═9057ee19-8a39-4712-bb94-bc951219a732
# ╠═16356cde-1788-4dc3-89b4-9e8ab3177b7b
# ╟─5b8980c6-3aa6-4376-9734-21364613fab2
