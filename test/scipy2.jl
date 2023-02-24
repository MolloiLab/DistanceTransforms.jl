
using Test, DistanceTransforms

@testset "Scipy 1D" begin
    x = [1, 1, 0, 0]
    test = transform(x, Scipy())
    answer = transform(x, Maurer())
    @test test == answer
end;