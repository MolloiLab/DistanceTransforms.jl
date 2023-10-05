### A Pluto.jl notebook ###
# v0.19.11

#> [frontmatter]
#> title = "Multi-threading"
#> category = "Tutorials"

using Markdown
using InteractiveUtils

# ╔═╡ 0469b379-82a6-4d53-9b52-262c04c7d6ba
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

# ╔═╡ a77f43f6-4fb4-11ec-2a50-3181f56e24e0
md"""
## Import packages
"""

# ╔═╡ e3c3666c-7686-4aab-95ad-bc4f41b3319b
TableOfContents()

# ╔═╡ ede39bd3-1e20-4426-a518-abe2f574aa91
md"""
## Multi-threaded distance transforms
One of the advantages of a Julia-based distance transform library is how accessible something like multi-threading is for complex algorithms. The `Felzenszwalb` distance transform is highly parallelizable and is set up to take advantage of multi-threaded hardware.
"""

# ╔═╡ 735aa1eb-2ce8-4ae2-8181-76056aa560de
threads = Threads.nthreads()

# ╔═╡ 5104bf97-e5bf-463e-a68a-cf1a9a7f77e8
md"""
### 2D
"""

# ╔═╡ c7914c8a-5c30-482d-9776-1121c1268b81
array = [
	1 0 1 1 1
	0 0 1 0 1
	1 0 1 0 1
	1 1 0 1 1
]

# ╔═╡ c8bd961f-14d7-4724-92f0-3d8cb24b239d
bool_array = boolean_indicator(array)

# ╔═╡ a0b300ff-2835-4440-a34d-cd1d3e5ed346
tfm = Felzenszwalb()

# ╔═╡ e38b8b56-a2cb-46a8-b5c8-9577a846ae1f
sq_euc_transform = transform!(bool_array, tfm, threads)

# ╔═╡ 6001d3d3-a067-4df3-a613-873fb4168c11
md"""
### 3D
"""

# ╔═╡ ae31df46-a7c8-41f3-b8e3-d5eeb1ba317c
array3D = rand([0, 1], 5, 5, 3)

# ╔═╡ 86c7fa96-b05d-42c8-b86b-331993df95fc
sq_euc_transform3D = transform!(boolean_indicator(array3D), tfm, threads)

# ╔═╡ b7651361-7332-4215-84dd-5cf55c61768d
md"""
## Timing
Let's compare the time difference between the regular `SquaredEuclidean` distance transform algorithm to the multi-threaded algorithm
"""

# ╔═╡ 95410494-f9f4-43e3-a5aa-6f0a22f767ef
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

# ╔═╡ 4d7c4515-1dd6-4ce3-b8f8-371066a99ed7
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
        ["Felzenszwalb","Felzenszwalb Threaded"];
        framevisible=false,
    )
	f
end

# ╔═╡ 5a1a3cb3-2073-45fb-9fd5-69bebbdd20fd
md"""
As you can see, for small arrays, the multi-threading functionality doesn't offer any improvements in speed (in fact, it limits it). But as the array size increases, the speed improvements also increase by order(s) of magnitude.
One might wonder why, if multi-threading is so beneficial, GPUs wouldn't be utilized. For most libraries, many would answer that question with a demonstration of the added complexity when going from regular programming to GPU programming and why it's not practical, but DistanceTransforms.jl is a pure Julia package. This means that GPUs are readily available with minimal code change, so the next [tutorial](https://glassnotebook.io/nb/118) will showcase this exact use case and why both end-users and developers might benefit from this.
"""

# ╔═╡ Cell order:
# ╟─a77f43f6-4fb4-11ec-2a50-3181f56e24e0
# ╠═0469b379-82a6-4d53-9b52-262c04c7d6ba
# ╠═e3c3666c-7686-4aab-95ad-bc4f41b3319b
# ╟─ede39bd3-1e20-4426-a518-abe2f574aa91
# ╠═735aa1eb-2ce8-4ae2-8181-76056aa560de
# ╟─5104bf97-e5bf-463e-a68a-cf1a9a7f77e8
# ╠═c7914c8a-5c30-482d-9776-1121c1268b81
# ╠═c8bd961f-14d7-4724-92f0-3d8cb24b239d
# ╠═a0b300ff-2835-4440-a34d-cd1d3e5ed346
# ╠═e38b8b56-a2cb-46a8-b5c8-9577a846ae1f
# ╟─6001d3d3-a067-4df3-a613-873fb4168c11
# ╠═ae31df46-a7c8-41f3-b8e3-d5eeb1ba317c
# ╠═86c7fa96-b05d-42c8-b86b-331993df95fc
# ╟─b7651361-7332-4215-84dd-5cf55c61768d
# ╠═95410494-f9f4-43e3-a5aa-6f0a22f767ef
# ╟─4d7c4515-1dd6-4ce3-b8f8-371066a99ed7
# ╟─5a1a3cb3-2073-45fb-9fd5-69bebbdd20fd
