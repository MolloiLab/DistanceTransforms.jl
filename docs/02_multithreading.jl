### A Pluto.jl notebook ###
# v0.17.2

using Markdown
using InteractiveUtils

# ╔═╡ 0469b379-82a6-4d53-9b52-262c04c7d6ba
begin
	let
		using Pkg
		Pkg.activate(mktempdir())
		Pkg.Registry.update()
		Pkg.add("PlutoUI")
		Pkg.add("Plots")
		Pkg.add("BenchmarkTools")
		Pkg.add(url="https://github.com/Dale-Black/DistanceTransforms.jl")
	end
	
	using PlutoUI
	using Plots
	using BenchmarkTools
	using DistanceTransforms
end

# ╔═╡ a77f43f6-4fb4-11ec-2a50-3181f56e24e0
md"""
## Import packages
"""

# ╔═╡ e3c3666c-7686-4aab-95ad-bc4f41b3319b
TableOfContents()

# ╔═╡ e99a10ab-1ab5-48d0-9342-9a724b750538
md"""
## Multi-threaded distance transforms
One of the advantages of a Julia-based distance transform library is how accessible something like multi-threading is for complex algorithms. The `SquaredEuclidean` distance transform is highly parallelizable and is set up to take advantage of multi-threaded hardware.
"""

# ╔═╡ 03a5c392-c134-42fa-b81c-3e3a3a7b11b4
threads = Threads.nthreads()

# ╔═╡ 3f7ba10d-d7f8-47cb-a879-1582f78e7828
array = [
	1 0 1 1 1
	0 0 1 0 1
	1 0 1 0 1
	1 1 0 1 1
]

# ╔═╡ 4a34664a-93e4-4835-ab9c-75dd10b9c483
bool_array = boolean_indicator(array)

# ╔═╡ 2d44897c-8430-4d35-8e26-4e2b03fba411
tfm = SquaredEuclidean(bool_array)

# ╔═╡ e85546b9-03f0-4a1c-abd4-cd7a86d3fe80
sq_euc_transform = transform!(bool_array, tfm, threads)

# ╔═╡ 4d35a592-5b22-44ef-8823-9025ce054643
md"""
## Timing
Let's compare the time difference between the regular `SquaredEuclidean` distance transform algorithm to the multi-threaded algorithm
"""

# ╔═╡ 390323c4-25a3-4048-b5e8-9ff99422c304
begin
	sedt_mean = []
	sedt_std = []
	
	sedtP_mean = []
	sedtP_std = []
	
	for n in 1:50:200
		
		# SEDT
		x2 = boolean_indicator(rand([0, 1], n, n))
		tfm2 = SquaredEuclidean(x2)
		sedt = @benchmark transform($x2, $tfm2)
		
		push!(sedt_mean, BenchmarkTools.mean(sedt).time)
		push!(sedt_std, BenchmarkTools.std(sedt).time)
		
		# SEDT threaded
		x3 = boolean_indicator(rand([0, 1], n, n))
		tfm3 = SquaredEuclidean(x3)
		nthreads = Threads.nthreads()
		sedtP = @benchmark transform!($x3, $tfm3, $nthreads)
		
		push!(sedtP_mean, BenchmarkTools.mean(sedtP).time)
		push!(sedtP_std, BenchmarkTools.std(sedtP).time)
	end
end

# ╔═╡ 9cbb098b-e5b4-45c1-8eeb-d8b3a91d565b
begin
	x = collect(1:length(sedt_mean))
	x_new = ones(Int32, size(x))
	for i in 2:length(x)
		x_new[i] = x[i] * (200^2)
	end
end

# ╔═╡ 6ccf4414-c76a-475d-8091-2aa03f268382
begin
	Plots.scatter(
		x_new,
		sedt_mean, 
		label="squared euclidean DT",
		xlabel = "Array size (elements)",
		ylabel = "Time (ns)"
		)
	Plots.scatter!(x_new, sedtP_mean, label="squared euclidean DT threaded")
end

# ╔═╡ b88bea1c-73bd-40e5-8966-3e5af6c3a026
md"""
As you can see, for small arrays the multi-threading functionality doesn't offer any improvements in speed (in fact it limits it) but as the array size increases the speed improvements also increase by order(s) of magnitude.
"""

# ╔═╡ Cell order:
# ╟─a77f43f6-4fb4-11ec-2a50-3181f56e24e0
# ╠═0469b379-82a6-4d53-9b52-262c04c7d6ba
# ╠═e3c3666c-7686-4aab-95ad-bc4f41b3319b
# ╟─e99a10ab-1ab5-48d0-9342-9a724b750538
# ╠═03a5c392-c134-42fa-b81c-3e3a3a7b11b4
# ╠═3f7ba10d-d7f8-47cb-a879-1582f78e7828
# ╠═4a34664a-93e4-4835-ab9c-75dd10b9c483
# ╠═2d44897c-8430-4d35-8e26-4e2b03fba411
# ╠═e85546b9-03f0-4a1c-abd4-cd7a86d3fe80
# ╟─4d35a592-5b22-44ef-8823-9025ce054643
# ╠═390323c4-25a3-4048-b5e8-9ff99422c304
# ╠═9cbb098b-e5b4-45c1-8eeb-d8b3a91d565b
# ╠═6ccf4414-c76a-475d-8091-2aa03f268382
# ╟─b88bea1c-73bd-40e5-8966-3e5af6c3a026
