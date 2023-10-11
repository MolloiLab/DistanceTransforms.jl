### A Pluto.jl notebook ###
# v0.19.26

#> [frontmatter]
#> title = "Benchmarks"
#> category = "Benchmarks"

using Markdown
using InteractiveUtils

# ╔═╡ 39846dfe-2bcf-11ed-1ec9-11cd75a2608e
# ╠═╡ show_logs = false
begin
	using Pkg
	Pkg.activate(".")
	Pkg.instantiate()
	
	using PlutoUI
	using DistanceTransforms
	using BenchmarkTools
	using CairoMakie
	using CUDA
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
	edt_mean_2D = Float32[]
	edt_std_2D = Float32[]
	
	sedt_mean_2D = Float32[]
	sedt_std_2D = Float32[]

	sedt_threaded_mean_2D = Float32[]
	sedt_threaded_std_2D = Float32[]
	
	sedt_gpu_mean = Float32[]
	sedt_gpu_std = Float32[]

	sizes_2D = Float32[]
	
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
		sedt = @benchmark transform($b_f, $tfm)
		
		append!(sedt_mean_2D, BenchmarkTools.mean(sedt).time)
		append!(sedt_std_2D, BenchmarkTools.std(sedt).time)
		
		# SEDT Threaded
		f = Bool.(rand([0, 1], n, n))
		b_f = boolean_indicator(f)
		sedt_threaded = @benchmark transform($b_f, $tfm, $threads)
		
		append!(sedt_threaded_mean_2D, BenchmarkTools.mean(sedt_threaded).time)
		append!(sedt_threaded_std_2D, BenchmarkTools.std(sedt_threaded).time)

		if has_cuda_gpu()
			# SEDT GPU
			f = Bool.(rand([0, 1], n, n))
			b_f = CuArray(boolean_indicator(f))
			output, v, z = CUDA.zeros(size(f)), CUDA.ones(Int32, size(f)), CUDA.ones(size(f) .+ 1)
			sedt_gpu = @benchmark transform($b_f, $tfm; output=$output, v=$v, z=$z)
			
			append!(sedt_gpu_mean, BenchmarkTools.mean(sedt_gpu).time)
			append!(sedt_gpu_std, BenchmarkTools.std(sedt_gpu).time)
		end
	end
end


# ╔═╡ deb92c24-d289-41e2-b013-533dc636bfe9
let 
    f = Figure()
    ax = Axis(
		f[1, 1],
		title = "Distance Transforms (2D)",
		xlabel = "Number of Elements",
		ylabel = "Time (ns)"
	)
	
    scatterlines!(sizes_2D, edt_mean_2D, label = "Maurer")
	scatterlines!(sizes_2D, sedt_mean_2D, label = "Felzenszwalb")
	scatterlines!(sizes_2D, sedt_threaded_mean_2D, label = "Felzenszwalb Threaded")

	if has_cuda_gpu()
		scatter!(sizes_2D, sedt_gpu_mean, label = "Felzenszwalb GPU")
	end

	axislegend(ax; position = :lt)
	
	f
end

# ╔═╡ 169156d8-2188-44eb-8cdb-97f6ed582cce
md"""
## 3D Benchmarks
"""

# ╔═╡ 63223fd0-b173-4f3b-ba80-12cbf52fe035
num_range2 = range(1, 100, 4)

# ╔═╡ 5a93c54e-7afe-4fa9-8923-7f9e127932ea
begin
	edt_mean_3D = Float32[]
	edt_std_3D = Float32[]
	
	sedt_mean_3D = Float32[]
	sedt_std_3D = Float32[]

	sedt_threaded_mean_3D = Float32[]
	sedt_threaded_std_3D = Float32[]

	sedt_gpu_mean_3D = Float32[]
	sedt_gpu_std_3D = Float32[]

	sizes_3D = Float32[]
	
	for n in num_range2
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
		sedt = @benchmark transform($b_f, $tfm)
		
		append!(sedt_mean_3D, BenchmarkTools.mean(sedt).time)
		append!(sedt_std_3D, BenchmarkTools.std(sedt).time)
		
		# SEDT Threaded
		f = Bool.(rand([0, 1], n, n, n))
		b_f = boolean_indicator(f)
		sedt_threaded = @benchmark transform($b_f, $tfm, $threads)
		
		append!(sedt_threaded_mean_3D, BenchmarkTools.mean(sedt_threaded).time)
		append!(sedt_threaded_std_3D, BenchmarkTools.std(sedt_threaded).time)

		if has_cuda_gpu()
			# SEDT GPU
			f = Bool.(rand([0, 1], n, n, n))
			b_f = CuArray(boolean_indicator(f))
			output, v, z = CUDA.zeros(size(f)), CUDA.ones(Int32, size(f)), CUDA.ones(size(f) .+ 1)
			sedt_gpu_3D = @benchmark transform($b_f, $tfm; output=$output, v=$v, z=$z)
			
			append!(sedt_gpu_mean_3D, BenchmarkTools.mean(sedt_gpu_3D).time)
			append!(sedt_gpu_std_3D, BenchmarkTools.std(sedt_gpu_3D).time)
		end
	end
end

# ╔═╡ adaae214-221e-419e-97ac-09fbb06e66a3
let 
    f = Figure()
    ax = Axis(
		f[1, 1],
		title = "Distance Transforms (3D)",
		xlabel = "Number of Elements",
		ylabel = "Time (ns)"
	)
	
    scatterlines!(sizes_3D, edt_mean_3D, label = "Maurer")
	scatterlines!(sizes_3D, sedt_mean_3D, label = "Felzenszwalb")
	scatterlines!(sizes_3D, sedt_threaded_mean_3D, label = "Felzenszwalb Threaded")

	if has_cuda_gpu()
		scatter!(sizes_3D, sedt_gpu_mean_3D, label = "Felzenszwalb GPU")
	end

	axislegend(ax; position = :lt)
	
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
# ╠═63223fd0-b173-4f3b-ba80-12cbf52fe035
# ╠═5a93c54e-7afe-4fa9-8923-7f9e127932ea
# ╟─adaae214-221e-419e-97ac-09fbb06e66a3
