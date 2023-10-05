### A Pluto.jl notebook ###
# v0.19.11

#> [frontmatter]
#> title = "Benchmarks"
#> category = "Benchmarks"

using Markdown
using InteractiveUtils

# ╔═╡ 39846dfe-2bcf-11ed-1ec9-11cd75a2608e
# ╠═╡ show_logs = false
begin
	let
		using Pkg
		Pkg.activate(mktempdir())
		Pkg.Registry.update()
		Pkg.add("PlutoUI")
		Pkg.add("BenchmarkTools")
		Pkg.add("CairoMakie")
		Pkg.add("CUDA")
		Pkg.add("FoldsThreads")
		Pkg.add(url="https://github.com/Dale-Black/DistanceTransforms.jl")
	end
	
	using PlutoUI
	using DistanceTransforms
	using BenchmarkTools
	using CairoMakie
	using CUDA
	using FoldsThreads
end

# ╔═╡ 4cd108b6-ec51-446a-b8e1-c52078a9e13d
TableOfContents()

# ╔═╡ 6f771ea0-dcf7-4528-84d5-e29da66de753
md"""
## 2D Benchmarks
"""

# ╔═╡ 32029509-72ed-4f84-9c87-08d5046633f7
num_range = range(1, 200, 4)

# ╔═╡ 967518da-7718-4f11-a5c2-8b65b9d5f8a3
threads = Threads.nthreads()

# ╔═╡ 75b7c7c5-baa0-43cd-a6c4-d69a6fa75b6a
begin
	edt_mean_2D = []
	edt_std_2D = []
	
	sedt_mean_2D = []
	sedt_std_2D = []
	
	sedt_inplace_mean_2D = []
	sedt_inplace_std_2D = []

	sedt_threaded_mean_2D = []
	sedt_threaded_std_2D = []

	sedt_threaded_mean_2D_depth = []
	sedt_threaded_std_2D_depth = []

	sedt_threaded_mean_2D_nonthread = []
	sedt_threaded_std_2D_nonthread = []

	sedt_threaded_mean_2D_worksteal = []
	sedt_threaded_std_2D_worksteal = []

	sedt_gpu_mean = []
	sedt_gpu_std = []

	sizes_2D = []
	
	for n in num_range
		n = Int(round(n))
		@info n
		_size = n^2
		append!(sizes_2D, _size)
		
		# EDT
		f = Bool.(rand([0, 1], n, n))
		edt = @benchmark transform($f, $Maurer())
		
		append!(edt_mean_2D, BenchmarkTools.mean(edt).time)
		append!(edt_std_2D, BenchmarkTools.std(edt).time)
		
		# SEDT
		f = Bool.(rand([0, 1], n, n))
		b_f = boolean_indicator(f)
		tfm = Felzenszwalb()
		sedt = @benchmark DistanceTransforms.transform($b_f, $tfm)
		
		append!(sedt_mean_2D, BenchmarkTools.mean(sedt).time)
		append!(sedt_std_2D, BenchmarkTools.std(sedt).time)
		
		# SEDT In-Place
		f = Bool.(rand([0, 1], n, n))
		b_f = boolean_indicator(f)
		sedt_inplace = @benchmark DistanceTransforms.transform!($b_f, $tfm)
		
		append!(sedt_inplace_mean_2D, BenchmarkTools.mean(sedt_inplace).time)
		append!(sedt_inplace_std_2D, BenchmarkTools.std(sedt_inplace).time)
		
		# SEDT Threaded
		f = Bool.(rand([0, 1], n, n))
		b_f = boolean_indicator(f)
		sedt_threaded = @benchmark DistanceTransforms.transform!($b_f, $tfm, $threads)
		
		append!(sedt_threaded_mean_2D, BenchmarkTools.mean(sedt_threaded).time)
		append!(sedt_threaded_std_2D, BenchmarkTools.std(sedt_threaded).time)

		# SEDT DepthFirst()
		f = Bool.(rand([0, 1], n, n))
		b_f = boolean_indicator(f)
		output, v, z = zeros(size(f)), ones(Int32, size(f)), ones(size(f) .+ 1)
		ex = DepthFirstEx()
		sedt_threaded_depth = @benchmark DistanceTransforms.transform!($b_f, $tfm, $ex; output=$output, v=$v, z=$z)
		
		append!(sedt_threaded_mean_2D_depth, BenchmarkTools.mean(sedt_threaded_depth).time)
		append!(sedt_threaded_std_2D_depth, BenchmarkTools.std(sedt_threaded_depth).time)

		# SEDT NonThreadedEx()
		f = Bool.(rand([0, 1], n, n))
		b_f = boolean_indicator(f)
		output, v, z = zeros(size(f)), ones(Int32, size(f)), ones(size(f) .+ 1)
		ex = NonThreadedEx()
		sedt_threaded_nonthread = @benchmark DistanceTransforms.transform!($b_f, $tfm, $ex; output=$output, v=$v, z=$z)
		
		append!(sedt_threaded_mean_2D_nonthread, BenchmarkTools.mean(sedt_threaded_nonthread).time)
		append!(sedt_threaded_std_2D_nonthread, BenchmarkTools.std(sedt_threaded_nonthread).time)

		# SEDT WorkStealingEx()
		f = Bool.(rand([0, 1], n, n))
		b_f = boolean_indicator(f)
		output, v, z = zeros(size(f)), ones(Int32, size(f)), ones(size(f) .+ 1)
		ex = WorkStealingEx()
		sedt_threaded_worksteal = @benchmark DistanceTransforms.transform!($b_f, $tfm, $ex; output=$output, v=$v, z=$z)
		
		append!(sedt_threaded_mean_2D_worksteal, BenchmarkTools.mean(sedt_threaded_worksteal).time)
		append!(sedt_threaded_std_2D_worksteal, BenchmarkTools.std(sedt_threaded_worksteal).time)

		if has_cuda_gpu()
			# SEDT GPU
			f = Bool.(rand([0, 1], n, n))
			b_f = CuArray(boolean_indicator(f))
			output, v, z = CUDA.zeros(size(f)), CUDA.ones(Int32, size(f)), CUDA.ones(size(f) .+ 1)
			sedt_gpu = @benchmark DistanceTransforms.transform!($b_f, $tfm; output=$output, v=$v, z=$z)
			
			append!(sedt_gpu_mean, BenchmarkTools.mean(sedt_gpu).time)
			append!(sedt_gpu_std, BenchmarkTools.std(sedt_gpu).time)
		end
	end
end


# ╔═╡ deb92c24-d289-41e2-b013-533dc636bfe9
let 
    f = Figure()
    ax1 = Axis(f[1, 1])
	ax1.xlabel = "Number of Elements"
    ax1.ylabel = "Time (ns)"
    ax1.title = "Distance Transforms (2D)"
	
    sc1 = scatter!(ax1, Float64.(sizes_2D), Float64.(edt_mean_2D))
	sc2 = scatter!(ax1, Float64.(sizes_2D), Float64.(sedt_mean_2D))
	sc3 = scatter!(ax1, Float64.(sizes_2D), Float64.(sedt_inplace_mean_2D))
	sc4 = scatter!(ax1, Float64.(sizes_2D), Float64.(sedt_threaded_mean_2D))
	sc5 = scatter!(ax1, Float64.(sizes_2D), Float64.(sedt_threaded_mean_2D_depth))
	sc6 = scatter!(ax1, Float64.(sizes_2D), Float64.(sedt_threaded_mean_2D_nonthread))
	sc7 = scatter!(ax1, Float64.(sizes_2D), Float64.(sedt_threaded_mean_2D_worksteal))

	if has_cuda_gpu()
		scPGU = scatter!(ax1, Float64.(sizes_2D), Float64.(sedt_gpu_mean))
		f[1, 2] = Legend(
	        f,
	        [sc1, sc2, sc3, sc4, sc5, sc6, sc7],
	        ["Maurer", "Felzenszwalb", "Felzenszwalb In-Place", "Felzenszwalb Threaded", "Felzenszwalb DepthFirstEx", "Felzenszwalb NonThreadedEx", "Felzenszwalb WorkStealingEx", "Felzenszwalb GPU"];
	        framevisible=false,
	    )
	else
			f[1, 2] = Legend(
	        f,
	        [sc1, sc2, sc3, sc4, sc5, sc6, sc7],
	        ["Maurer", "Felzenszwalb", "Felzenszwalb In-Place", "Felzenszwalb Threaded", "Felzenszwalb DepthFirstEx", "Felzenszwalb NonThreadedEx", "Felzenszwalb WorkStealingEx"];
	        framevisible=false,
	    )
	end
	f
end

# ╔═╡ 169156d8-2188-44eb-8cdb-97f6ed582cce
md"""
## 3D Benchmarks
"""

# ╔═╡ 5a93c54e-7afe-4fa9-8923-7f9e127932ea
begin
	edt_mean_3D = []
	edt_std_3D = []
	
	sedt_mean_3D = []
	sedt_std_3D = []
	
	sedt_inplace_mean_3D = []
	sedt_inplace_std_3D = []

	sedt_threaded_mean_3D = []
	sedt_threaded_std_3D = []

	sedt_threaded_mean_3D_depth = []
	sedt_threaded_std_3D_depth = []

	sedt_threaded_mean_3D_nonthread = []
	sedt_threaded_std_3D_nonthread = []

	sedt_threaded_mean_3D_worksteal = []
	sedt_threaded_std_3D_worksteal = []

	sedt_gpu_mean_3D = []
	sedt_gpu_std_3D = []

	sizes_3D = []
	
	for n in num_range
		n = Int(round(n))
		@info n
		_size = n^3
		append!(sizes_3D, _size)
		
		# EDT
		f = Bool.(rand([0, 1], n, n, n))
		edt = @benchmark transform($f, $Maurer())
		
		append!(edt_mean_3D, BenchmarkTools.mean(edt).time)
		append!(edt_std_3D, BenchmarkTools.std(edt).time)
		
		# SEDT
		f = Bool.(rand([0, 1], n, n, n))
		b_f = boolean_indicator(f)
		tfm = Felzenszwalb()
		sedt = @benchmark DistanceTransforms.transform($b_f, $tfm)
		
		append!(sedt_mean_3D, BenchmarkTools.mean(sedt).time)
		append!(sedt_std_3D, BenchmarkTools.std(sedt).time)
		
		# SEDT In-Place
		f = Bool.(rand([0, 1], n, n, n))
		b_f = boolean_indicator(f)
		sedt_inplace = @benchmark DistanceTransforms.transform!($b_f, $tfm)
		
		append!(sedt_inplace_mean_3D, BenchmarkTools.mean(sedt_inplace).time)
		append!(sedt_inplace_std_3D, BenchmarkTools.std(sedt_inplace).time)
		
		# SEDT Threaded
		f = Bool.(rand([0, 1], n, n, n))
		b_f = boolean_indicator(f)
		sedt_threaded = @benchmark DistanceTransforms.transform!($b_f, $tfm, $threads)
		
		append!(sedt_threaded_mean_3D, BenchmarkTools.mean(sedt_threaded).time)
		append!(sedt_threaded_std_3D, BenchmarkTools.std(sedt_threaded).time)

		# SEDT DepthFirst()
		f = Bool.(rand([0, 1], n, n, n))
		b_f = boolean_indicator(f)
		output, v, z = zeros(size(f)), ones(Int32, size(f)), ones(size(f) .+ 1)
		ex = DepthFirstEx()
		sedt_threaded_depth = @benchmark DistanceTransforms.transform!($b_f, $tfm, $ex; output=$output, v=$v, z=$z)
		
		append!(sedt_threaded_mean_3D_depth, BenchmarkTools.mean(sedt_threaded_depth).time)
		append!(sedt_threaded_std_3D_depth, BenchmarkTools.std(sedt_threaded_depth).time)

		# SEDT NonThreadedEx()
		f = Bool.(rand([0, 1], n, n, n))
		b_f = boolean_indicator(f)
		output, v, z = zeros(size(f)), ones(Int32, size(f)), ones(size(f) .+ 1)
		ex = NonThreadedEx()
		sedt_threaded_nonthread = @benchmark DistanceTransforms.transform!($b_f, $tfm, $ex; output=$output, v=$v, z=$z)
		
		append!(sedt_threaded_mean_3D_nonthread, BenchmarkTools.mean(sedt_threaded_nonthread).time)
		append!(sedt_threaded_std_3D_nonthread, BenchmarkTools.std(sedt_threaded_nonthread).time)

		# SEDT WorkStealingEx()
		f = Bool.(rand([0, 1], n, n, n))
		b_f = boolean_indicator(f)
		output, v, z = zeros(size(f)), ones(Int32, size(f)), ones(size(f) .+ 1)
		ex = WorkStealingEx()
		sedt_threaded_worksteal = @benchmark DistanceTransforms.transform!($b_f, $tfm, $ex; output=$output, v=$v, z=$z)
		
		append!(sedt_threaded_mean_3D_worksteal, BenchmarkTools.mean(sedt_threaded_worksteal).time)
		append!(sedt_threaded_std_3D_worksteal, BenchmarkTools.std(sedt_threaded_worksteal).time)

		if has_cuda_gpu()
			# SEDT GPU
			f = Bool.(rand([0, 1], n, n, n))
			b_f = CuArray(boolean_indicator(f))
			output, v, z = CUDA.zeros(size(f)), CUDA.ones(Int32, size(f)), CUDA.ones(size(f) .+ 1)
			sedt_gpu_3D = @benchmark DistanceTransforms.transform!($b_f, $tfm; output=$output, v=$v, z=$z)
			
			append!(sedt_gpu_mean_3D, BenchmarkTools.mean(sedt_gpu_3D).time)
			append!(sedt_gpu_std_3D, BenchmarkTools.std(sedt_gpu_3D).time)
		end
	end
end


# ╔═╡ 52e4d183-a83c-4864-a282-60dc841402b8
let 
    f = Figure()
    ax1 = Axis(f[1, 1])
	ax1.xlabel = "Number of Elements"
    ax1.ylabel = "Time (ns)"
    ax1.title = "Distance Transforms (3D)"
	
    sc1 = scatter!(ax1, Float64.(sizes_3D), Float64.(edt_mean_3D))
	sc2 = scatter!(ax1, Float64.(sizes_3D), Float64.(sedt_mean_3D))
	sc3 = scatter!(ax1, Float64.(sizes_3D), Float64.(sedt_inplace_mean_3D))
	sc4 = scatter!(ax1, Float64.(sizes_3D), Float64.(sedt_threaded_mean_3D))
	sc5 = scatter!(ax1, Float64.(sizes_3D), Float64.(sedt_threaded_mean_3D_depth))
	sc6 = scatter!(ax1, Float64.(sizes_3D), Float64.(sedt_threaded_mean_3D_nonthread))
	sc7 = scatter!(ax1, Float64.(sizes_3D), Float64.(sedt_threaded_mean_3D_worksteal))

	if has_cuda_gpu()
		scPGU = scatter!(ax1, Float64.(sizes_3D), Float64.(sedt_gpu_mean))
		f[1, 2] = Legend(
	        f,
	        [sc1, sc2, sc3, sc4, sc5, sc6, sc7],
	        ["Maurer", "Felzenszwalb", "Felzenszwalb In-Place", "Felzenszwalb Threaded", "Felzenszwalb DepthFirstEx", "Felzenszwalb NonThreadedEx", "Felzenszwalb WorkStealingEx", "Felzenszwalb GPU"];
	        framevisible=false,
	    )
	else
			f[1, 2] = Legend(
	        f,
	        [sc1, sc2, sc3, sc4, sc5, sc6, sc7],
	        ["Maurer", "Felzenszwalb", "Felzenszwalb In-Place", "Felzenszwalb Threaded", "Felzenszwalb DepthFirstEx", "Felzenszwalb NonThreadedEx", "Felzenszwalb WorkStealingEx"];
	        framevisible=false,
	    )
	end
	f
end

# ╔═╡ Cell order:
# ╠═39846dfe-2bcf-11ed-1ec9-11cd75a2608e
# ╠═4cd108b6-ec51-446a-b8e1-c52078a9e13d
# ╟─6f771ea0-dcf7-4528-84d5-e29da66de753
# ╠═32029509-72ed-4f84-9c87-08d5046633f7
# ╠═967518da-7718-4f11-a5c2-8b65b9d5f8a3
# ╠═75b7c7c5-baa0-43cd-a6c4-d69a6fa75b6a
# ╟─deb92c24-d289-41e2-b013-533dc636bfe9
# ╟─169156d8-2188-44eb-8cdb-97f6ed582cce
# ╠═5a93c54e-7afe-4fa9-8923-7f9e127932ea
# ╟─52e4d183-a83c-4864-a282-60dc841402b8
