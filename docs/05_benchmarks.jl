### A Pluto.jl notebook ###
# v0.19.32

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

	# Import Comparison Distance Transforms
	using DistanceTransformsPy: pytransform # Scipy 
	using ImageMorphology: distance_transform, feature_transform # ImageMorphology

	# Import Packages
	using PlutoUI, BenchmarkTools, CairoMakie, Random

	# Import Various GPU packages
	using KernelAbstractions
	using CUDA, AMDGPU, Metal

	using DistanceTransforms
end

# ╔═╡ 7f3a06bd-195f-4732-8844-3bdafff90cce
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

# ╔═╡ 4cd108b6-ec51-446a-b8e1-c52078a9e13d
TableOfContents()

# ╔═╡ 6f771ea0-dcf7-4528-84d5-e29da66de753
md"""
## 2D Benchmarks
"""

# ╔═╡ 967518da-7718-4f11-a5c2-8b65b9d5f8a3
Threads.nthreads()

# ╔═╡ 32029509-72ed-4f84-9c87-08d5046633f7
num_range = collect(10:200:1010)

# ╔═╡ 75b7c7c5-baa0-43cd-a6c4-d69a6fa75b6a
begin
	distance_transforms_2D = Float32[]
	distance_transforms_gpu_2D = Float32[]
	image_morphology_2D = Float32[]
	scipy_2D = Float32[]

	sizes_2D = Float32[]
	for n in num_range
		@info n
		append!(sizes_2D, n^2)
		f = Bool.(rand([0f0, 1f0], n, n))

		# DistanceTransforms.jl
		tfm = @benchmark transform($boolean_indicator($f))
		append!(distance_transforms_2D, BenchmarkTools.mean(tfm).time)

		# DistanceTransforms.jl GPU
		if backend != CPU()
			f_gpu = dev(rand([0f0, 1f0], n, n))
			tfm = @benchmark transform($f_gpu)
			append!(distance_transforms_gpu_2D, BenchmarkTools.mean(tfm).time)
		end
		
		# ImageMorphology.jl
		tfm = @benchmark distance_transform($feature_transform($f))
		append!(image_morphology_2D, BenchmarkTools.mean(tfm).time)
		
		# Scipy
		tfm = @benchmark pytransform($f)
		append!(scipy_2D, BenchmarkTools.mean(tfm).time)
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
	
    scatterlines!(sizes_2D, distance_transforms_2D, label = "DistanceTransforms.jl")
	if backend != CPU()
		scatterlines!(sizes_2D, distance_transforms_gpu_2D, label = "DistanceTransforms.jl (GPU)")
	end
	scatterlines!(sizes_2D, image_morphology_2D, label = "ImageMorphology.jl")
	scatterlines!(sizes_2D, scipy_2D, label = "Scipy")

	axislegend(ax; position = :lt)
	
	f
end

# ╔═╡ 169156d8-2188-44eb-8cdb-97f6ed582cce
md"""
## 3D Benchmarks
"""

# ╔═╡ f00f036e-a9e3-48c0-a278-26636b89944c
collect(10:20:110)

# ╔═╡ 63223fd0-b173-4f3b-ba80-12cbf52fe035
num_range2 = collect(10:20:110)

# ╔═╡ 6a4e4d88-b7d8-4b7b-8237-94ca6a7bce6b
begin
	distance_transforms_3D = Float32[]
	distance_transforms_gpu_3D = Float32[]
	image_morphology_3D = Float32[]
	scipy_3D = Float32[]

	sizes_3D = Float32[]
	for n in num_range2
		@info n
		append!(sizes_3D, n^3) # Changed to n^3 for 3D
		f = Bool.(rand([0f0, 1f0], n, n, n)) # 3D array

		# DistanceTransforms.jl
		tfm = @benchmark transform($boolean_indicator($f))
		append!(distance_transforms_3D, BenchmarkTools.mean(tfm).time)

		# # DistanceTransforms.jl GPU
		# if backend != CPU()
		# 	f_gpu = dev(rand([0f0, 1f0], n, n, n)) # 3D array for GPU
		# 	tfm = @benchmark transform($f_gpu)
		# 	append!(distance_transforms_gpu_3D, BenchmarkTools.mean(tfm).time)
		# end
		
		# ImageMorphology.jl
		tfm = @benchmark distance_transform($feature_transform($f))
		append!(image_morphology_3D, BenchmarkTools.mean(tfm).time)
		
		# Scipy
		tfm = @benchmark pytransform($f)
		append!(scipy_3D, BenchmarkTools.mean(tfm).time)
	end
end

# ╔═╡ 42578312-87f1-4091-80c7-046f398dd767
let 
    f = Figure()
    ax = Axis(
		f[1, 1],
		title = "Distance Transforms (3D)",
		xlabel = "Number of Elements",
		ylabel = "Time (ns)"
	)
	
    scatterlines!(sizes_3D, distance_transforms_3D, label = "DistanceTransforms.jl")
	# if backend != CPU()
	# 	scatterlines!(sizes_3D, distance_transforms_gpu_3D, label = "DistanceTransforms.jl (GPU)")
	# end
	scatterlines!(sizes_3D, image_morphology_3D, label = "ImageMorphology.jl")
	scatterlines!(sizes_3D, scipy_3D, label = "Scipy")

	axislegend(ax; position = :lt)
	
	f
end

# ╔═╡ adaae214-221e-419e-97ac-09fbb06e66a3
# ╠═╡ disabled = true
#=╠═╡
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
  ╠═╡ =#

# ╔═╡ Cell order:
# ╠═39846dfe-2bcf-11ed-1ec9-11cd75a2608e
# ╠═7f3a06bd-195f-4732-8844-3bdafff90cce
# ╠═4cd108b6-ec51-446a-b8e1-c52078a9e13d
# ╟─6f771ea0-dcf7-4528-84d5-e29da66de753
# ╠═967518da-7718-4f11-a5c2-8b65b9d5f8a3
# ╠═32029509-72ed-4f84-9c87-08d5046633f7
# ╠═75b7c7c5-baa0-43cd-a6c4-d69a6fa75b6a
# ╟─deb92c24-d289-41e2-b013-533dc636bfe9
# ╟─169156d8-2188-44eb-8cdb-97f6ed582cce
# ╠═f00f036e-a9e3-48c0-a278-26636b89944c
# ╠═63223fd0-b173-4f3b-ba80-12cbf52fe035
# ╠═6a4e4d88-b7d8-4b7b-8237-94ca6a7bce6b
# ╠═42578312-87f1-4091-80c7-046f398dd767
# ╟─adaae214-221e-419e-97ac-09fbb06e66a3
