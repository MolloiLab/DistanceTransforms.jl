@testset "boolean_indicator CPU" begin
	ns = [10, 100]
	for n in ns
		f = rand([0, 1], n)
		f_bool = Bool.(f)
		@test boolean_indicator(f) == boolean_indicator(f_bool)
	end
	for n in ns
		f = rand([0, 1], n, n)
		f_bool = Bool.(f)
		@test boolean_indicator(f) == boolean_indicator(f_bool)
	end
	for n in ns
		f = rand([0, 1], n, n, n)
		f_bool = Bool.(f)
		@test boolean_indicator(f) == boolean_indicator(f_bool)
	end
end

@testset "boolean_indicator GPU" begin
	ns = [10, 100]
	for n in ns
		f = rand!(KernelAbstractions.allocate(backend, Float32, n)) .> 0.5f0
		f_arr = Array(f)
		@test Array(boolean_indicator(f)) == boolean_indicator(f_arr)
	end
	for n in ns
		f = rand!(KernelAbstractions.allocate(backend, Float32, n, n)) .> 0.5f0
		f_arr = Array(f)
		@test Array(boolean_indicator(f)) == boolean_indicator(f_arr)
	end
	for n in ns
		f = rand!(KernelAbstractions.allocate(backend, Float32, n, n, n)) .> 0.5f0
		f_arr = Array(f)
		@test Array(boolean_indicator(f)) == boolean_indicator(f_arr)
	end
end