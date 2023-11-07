using ImageMorphology: distance_transform, feature_transform

test_transform(img) = distance_transform(feature_transform(Bool.(img)))

@testset "1D" begin
	ns = [10, 100]
	for n in ns
		img = rand([0f0, 1f0], n)
		img_bool = boolean_indicator(img)
		output = transform(img_bool)
		img_test = test_transform(img).^2
		@test isapprox(Array(output), img_test; atol=1e-3)
	end
end

@testset "2D" begin
	ns = [10, 100]
	for n in ns
		img = rand([0f0, 1f0], n, n)
		img_bool = boolean_indicator(img)
		output = transform(img_bool)
		img_test = test_transform(img).^2
		@test isapprox(Array(output), img_test; atol=1e-3)
	end
	for n in ns
		img = rand([0f0, 1f0], n, n)
		img_bool = boolean_indicator(img)
		output = transform(img_bool; threaded = false)
		img_test = test_transform(img).^2
		@test isapprox(Array(output), img_test; atol=1e-3)
	end
end

@testset "3D" begin
	ns = [10, 100]
	for n in ns
		img = rand([0f0, 1f0], n, n, n)
		img_bool = boolean_indicator(img)
		output = transform(img_bool)
		img_test = test_transform(img).^2
		@test isapprox(Array(output), img_test; atol=1e-3)
	end
	for n in ns
		img = rand([0f0, 1f0], n, n, n)
		img_bool = boolean_indicator(img)
		output = transform(img_bool; threaded = false)
		img_test = test_transform(img).^2
		@test isapprox(Array(output), img_test; atol=1e-3)
	end
end

using KernelAbstractions
using CUDA, AMDGPU, oneAPI, Metal
using Random

if CUDA.functional()
	@info "Using CUDA"
	CUDA.versioninfo()
	backend = CUDABackend()
elseif AMDGPU.functional()
	@info "Using AMD"
	AMDGPU.versioninfo()
	backend = ROCBackend()
elseif oneAPI.functional()
	@info "Using oneAPI"
	oneAPI.versioninfo()
	backend = oneBackend()
elseif Metal.functional()
	@info "Using Metal"
	Metal.versioninfo()
	backend = MetalBackend()
else
    @info "No GPU is available. Using CPU."
	backend = CPU()
end

@testset "2D GPU" begin
	n = 100
	img = rand!(KernelAbstractions.allocate(backend, Float32, 10, 10)) .> 0.5f0
	img_bool = boolean_indicator(img)
	output = transform(img_bool)
	img_test = test_transform(Array(img)).^2
	@test isapprox(Array(output), img_test; atol=1e-3)
end
