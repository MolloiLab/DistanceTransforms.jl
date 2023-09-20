@testset "boolean_indicator" begin
	f = rand([0, 1], 100, 100)
	f_bool = Bool.(f)

	@test boolean_indicator(f) == boolean_indicator(f_bool)
end
