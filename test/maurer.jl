
@testset "Maurer 1D" begin
	x = [1, 1, 0, 0]
	test = transform(x, Maurer())
	answer = [0, 0, 1, 2]
	@test test == answer

	x = [1, 0, 0, 1, 1, 1]
	test = transform(x, Maurer())
	answer = [0, 1, 1, 0, 0, 0]
	@test test == answer
end;

@testset "Maurer 2D" begin
	x = [
		1 1 0 0
		0 1 1 0
		0 1 0 1
		0 1 0 0
	]
	test = transform(x, Maurer())
	answer = [
		 0.0  0.0  1.0  1.4142135623730951
		 1.0  0.0  0.0  1.0
		 1.0  0.0  1.0  0.0
		 1.0  0.0  1.0  1.0
	]
	@test test ≈ answer

	x = [
		1 1 1 0
		0 1 1 1
		0 1 1 1
		0 1 1 1
	]
	test = transform(x, Maurer())
	answer = [
		 0.0  0.0  0.0  1.0
		 1.0  0.0  0.0  0.0
		 1.0  0.0  0.0  0.0
		 1.0  0.0  0.0  0.0
	]
	@test test ≈ answer

	x = Bool.([
		1 1 0 0
		0 1 1 0
		0 1 0 1
		0 1 0 0
	])
	test = transform(x, Maurer())
	answer = [
		 0.0  0.0  1.0  1.4142135623730951
		 1.0  0.0  0.0  1.0
		 1.0  0.0  1.0  0.0
		 1.0  0.0  1.0  1.0
	]
	@test test == answer
end

@testset "Maurer 3D" begin
	img = [
		0 0 0 0 0 0 0 0 0 0 0
		0 0 0 0 0 0 0 0 0 0 0
		0 0 0 0 0 0 0 0	0 0 0
		0 0 0 1 1 1 0 0 0 0 0
		0 0 1 0 0 1 0 0 0 0 0
		0 0 1 0 0 1 1 1 0 0 0
		0 0 1 0 0 0 0 1 0 0 0
		0 0 1 0 0 0 0 1 0 0 0
		0 0 0 1 1 1 1 0 0 0 0	
		0 0 0 0 0 0 0 0 0 0 0	
		0 0 0 0 0 0 0 0 0 0 0
	]
	img_inv = @. ifelse(img == 0, 1, 0)
	vol = cat(img, img_inv, dims=3)
	container2 = []
	for i in 1:10
		push!(container2, vol)
	end
	vol_inv = cat(container2..., dims=3)
	test = transform(vol_inv, Maurer())
	a1 = img_inv
	a2 = img
	ans = cat(a1, a2, dims=3)
	container_a = []
	for i in 1:10
		push!(container_a, ans)
	end
	answer = cat(container_a..., dims=3)
	@test test == answer
end