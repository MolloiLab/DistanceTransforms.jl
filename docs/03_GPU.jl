### A Pluto.jl notebook ###
# v0.17.1

using Markdown
using InteractiveUtils

# ╔═╡ 27924817-b0dc-4985-9404-5b399ef3d4c7
begin
	let
		using Pkg
		Pkg.activate(mktempdir())
		Pkg.Registry.update()
		Pkg.add("PlutoUI")
		Pkg.add("Plots")
		Pkg.add("CUDA")
		Pkg.add("BenchmarkTools")
		Pkg.add(url="https://github.com/Dale-Black/DistanceTransforms.jl")
	end
	
	using PlutoUI
	using Plots
	using CUDA
	using BenchmarkTools
	using DistanceTransforms
end

# ╔═╡ 66ee6566-8f02-4d35-aba7-4b7f2495cc47
begin
using Markdown
using InteractiveUtils
end

# ╔═╡ 2c9952e9-c4fe-438b-882f-d58737e9ab28
md"""
## Import packages
"""

# ╔═╡ e76c58b0-66c3-4947-99a9-39b863293169
TableOfContents()

# ╔═╡ 78392f7b-df42-4b4a-9414-5c4ea574e33c
md"""
## Using GPU

We have already seen that the "Distance Transform" library in Julia has options that allow for multi-threading. This can be further exploited by using GPUs. Using GPUs allows for parallel processing: this means we can process multiple threads simultaneously thus speeding up the computational speed significantly.  
"""

# ╔═╡ c69dae8b-6736-42a8-9694-c6f138f9c3fd
md"""
We will be using the same threading process from the previous tutorial while implementing ideas from the CUDA library. 
"""

# ╔═╡ e94dd940-6897-477a-8286-8be9e946b2eb
threads = Threads.nthreads()

# ╔═╡ 94a842d9-811b-4980-8bdd-a85e939f1589
array = [
	1 0 1 1 1
	0 0 1 0 1
	1 0 1 0 1
	1 1 0 1 1
]

# ╔═╡ 65439bc5-2952-4544-8226-faddc13abfa6
bool_array = boolean_indicator(array)

# ╔═╡ 3b43a310-e979-4965-8d5d-917d01949076
tfm = SquaredEuclidean(bool_array)

# ╔═╡ 976ec881-db52-409a-b952-b7cb4e2f8d52
sq_euc_transform = transform!(bool_array, tfm, threads)

# ╔═╡ 2e0384a3-1771-4f9c-9a6e-44d5a5905009
md"""
## Timing
Let's compare the time difference between the `SquaredEuclidean` distance transforms, both threaded and otherwise, that are run on the CPU against the algorithms run on the GPU. 
"""

# ╔═╡ ba54dedf-c4ca-4a7e-aee3-1733813cd7a7
begin
	sedt_mean = []
	sedt_std = []
	
	sedtP_mean = []
	sedtP_std = []

	sedt_gpu_mean = []
	sedt_gpu_std = []
	
	sedtP_gpu_mean = []
	sedtP_gpu_std = []
	
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
				
		# SEDT with GPU
		x3 = boolean_indicator(rand([0, 1], n, n)) |> gpu
		tfm3 = SquaredEuclidean(x3)
		sedt_gpu = @benchmark transform($x3, $tfm3)
		
		push!(sedt_gpu_mean, BenchmarkTools.mean(sedt_gpu).time)
		push!(sedt_gpu_std, BenchmarkTools.std(sedt_gpu).time)
		
		# SEDT threaded
		x4 = boolean_indicator(rand([0, 1], n, n)) |> gpu
		tfm4 = SquaredEuclidean(x4)
		nthreads = Threads.nthreads()
		sedtP = @benchmark transform!($x4, $tfm4, $nthreads)
		
		push!(sedtP_gpu_mean, BenchmarkTools.mean(sedtP_gpu).time)
		push!(sedtP_gpu_std, BenchmarkTools.std(sedtP_gpu).time)
	end
end

# ╔═╡ 9449105b-6464-479e-b0d5-c831f205e68b
begin
	x = collect(1:length(sedt_mean))
	x_new = ones(Int32, size(x))
	for i in 2:length(x)
		x_new[i] = x[i] * (200^2)
	end
end

# ╔═╡ cb2889c7-9438-4e35-9a3a-77ac43e992d4
begin
	Plots.scatter(
		x_new,
		sedt_mean, 
		label="squared euclidean DT",
		xlabel = "Array size (elements)",
		ylabel = "Time (ns)"
		)
	Plots.scatter!(x_new, sedtP_mean, label="squared euclidean DT threaded")
	Plots.scatter!(x_new, sedt_gpu_mean, label="squared euclidean DT with GPU")
	Plots.scatter!(x_new, sedtP_gpu_mean, label="squared euclidean DT threaded with GPU")
end

# ╔═╡ b3b58e17-e64c-4567-957c-eb09f6bd3945
md"""
As the array size increases the speed improvements also increase by orders of magnitude. (Need to complete after getting results.)
"""

# ╔═╡ Cell order:
# ╠═66ee6566-8f02-4d35-aba7-4b7f2495cc47
# ╠═2c9952e9-c4fe-438b-882f-d58737e9ab28
# ╠═27924817-b0dc-4985-9404-5b399ef3d4c7
# ╠═e76c58b0-66c3-4947-99a9-39b863293169
# ╠═78392f7b-df42-4b4a-9414-5c4ea574e33c
# ╠═c69dae8b-6736-42a8-9694-c6f138f9c3fd
# ╠═e94dd940-6897-477a-8286-8be9e946b2eb
# ╠═94a842d9-811b-4980-8bdd-a85e939f1589
# ╠═65439bc5-2952-4544-8226-faddc13abfa6
# ╠═3b43a310-e979-4965-8d5d-917d01949076
# ╠═976ec881-db52-409a-b952-b7cb4e2f8d52
# ╠═2e0384a3-1771-4f9c-9a6e-44d5a5905009
# ╠═ba54dedf-c4ca-4a7e-aee3-1733813cd7a7
# ╠═9449105b-6464-479e-b0d5-c831f205e68b
# ╠═cb2889c7-9438-4e35-9a3a-77ac43e992d4
# ╠═b3b58e17-e64c-4567-957c-eb09f6bd3945
