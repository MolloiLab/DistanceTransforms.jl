### A Pluto.jl notebook ###
# v0.19.11

using Markdown
using InteractiveUtils

# ╔═╡ cd8bf944-2329-11ed-208f-1b2e91673a5e
# ╠═╡ show_logs = false
begin
	using Pkg
	Pkg.activate("..")
	using Revise
	using PlutoUI
	using Test
	using DistanceTransforms
end

# ╔═╡ 373a8802-bbac-4ab5-abe4-420ccbb61ea0
TableOfContents()

# ╔═╡ 81f78cda-0efa-4a74-bf4c-a39197d9b73f
md"""
## 1D
"""

# ╔═╡ d3f4223e-f697-4a31-8825-66b1fae2b4f5
let
	@testset "squared euclidean 1D" begin
		f = [1, 1, 0, 0, 0, 1, 1]
		output, v, z = zeros(length(f)), ones(Int32, length(f)), ones(length(f) .+ 1)
		tfm = SquaredEuclidean()
		test = transform(boolean_indicator(f), tfm; output=output, v=v, z=z)
		answer = [0.0, 0.0, 1.0, 4.0, 1.0, 0.0, 0.0]
		@test test == answer
	end
end

# ╔═╡ d48cc7be-b769-4a64-970e-5982105fd382
let
    @testset "squared euclidean 1D" begin
        f = [0, 0, 0, 1]
        output, v, z = zeros(length(f)), ones(Int32, length(f)), ones(length(f) .+ 1)
        tfm = SquaredEuclidean()
        test = transform(boolean_indicator(f), tfm; output=output, v=v, z=z)
        answer = [9.0, 4.0, 1.0, 0.0]
        @test test == answer
    end
end

# ╔═╡ 4ded097b-5e18-47af-bbbb-aa9fb626a9d1
let
    @testset "squared euclidean 1D" begin
        f = [1, 0, 0, 0]
        output, v, z = zeros(length(f)), ones(Int32, length(f)), ones(length(f) .+ 1)
        tfm = SquaredEuclidean()
        test = transform(boolean_indicator(f), tfm; output=output, v=v, z=z)
        answer = [0, 1, 4, 9]
        @test test == answer
    end
end

# ╔═╡ 3b1fcab4-ca39-40c4-8cfa-410acfd6b006
md"""
## 2D
"""

# ╔═╡ eafcd9e8-d691-4ae5-9e71-e999f8b6702c
let
    @testset "squared_euclidean 2D" begin
        img = [
            0 1 1 1 0 0 0 1 1
            1 1 1 1 1 0 0 0 1
            1 0 0 0 1 0 0 1 1
            1 0 0 0 1 0 1 1 0
            1 0 0 0 1 1 0 1 0
            1 1 1 1 1 0 0 1 0
            0 1 1 1 0 0 0 0 1
        ]
        output, v, z = zeros(size(img)), ones(Int32, size(img)), ones(size(img) .+ 1)
        tfm = SquaredEuclidean()
        test = transform(boolean_indicator(img), tfm; output=output, v=v, z=z)
        answer = [
            1.0  0.0  0.0  0.0  1.0  2.0  1.0  0.0  0.0
            0.0  0.0  0.0  0.0  0.0  1.0  2.0  1.0  0.0
            0.0  1.0  1.0  1.0  0.0  1.0  1.0  0.0  0.0
            0.0  1.0  4.0  1.0  0.0  1.0  0.0  0.0  1.0
            0.0  1.0  1.0  1.0  0.0  0.0  1.0  0.0  1.0
            0.0  0.0  0.0  0.0  0.0  1.0  1.0  0.0  1.0
            1.0  0.0  0.0  0.0  1.0  2.0  2.0  1.0  0.0
        ]
        @test test == answer
    end
end

# ╔═╡ 2f5755cb-e575-4ca9-a8b3-f2fd7846c068
let
	@testset "squared_euclidean 2D" begin
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
        output, v, z = zeros(size(img)), ones(Int32, size(img)), ones(size(img) .+ 1)
        tfm = SquaredEuclidean()
        test = transform(boolean_indicator(img), tfm; output=output, v=v, z=z)
        answer = [
            18.0  13.0  10.0  9.0  9.0  9.0  10.0  13.0  18.0  25.0  34.0
            13.0   8.0   5.0  4.0  4.0  4.0   5.0   8.0  13.0  20.0  25.0
             8.0   5.0   2.0  1.0  1.0  1.0   2.0   5.0  10.0  13.0  18.0
             5.0   2.0   1.0  0.0  0.0  0.0   1.0   4.0   5.0   8.0  13.0
             4.0   1.0   0.0  1.0  1.0  0.0   1.0   1.0   2.0   5.0  10.0
             4.0   1.0   0.0  1.0  1.0  0.0   0.0   0.0   1.0   4.0   9.0
             4.0   1.0   0.0  1.0  2.0  1.0   1.0   0.0   1.0   4.0   9.0
             4.0   1.0   0.0  1.0  1.0  1.0   1.0   0.0   1.0   4.0   9.0
             5.0   2.0   1.0  0.0  0.0  0.0   0.0   1.0   2.0   5.0  10.0
             8.0   5.0   2.0  1.0  1.0  1.0   1.0   2.0   5.0   8.0  13.0
            13.0   8.0   5.0  4.0  4.0  4.0   4.0   5.0   8.0  13.0  18.0
        ]
        @test test == answer
    end
end

# ╔═╡ 755aa045-73cd-425c-8057-50e65c342716
md"""
## 3D
"""

# ╔═╡ afc433aa-ecc9-443a-bbd4-55bd48730937
let
    @testset "squared_euclidean 3D" begin
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
        output, v, z = zeros(size(vol_inv)), ones(Int32, size(vol_inv)), ones(size(vol_inv) .+ 1)
        tfm = SquaredEuclidean()
        test = transform(boolean_indicator(vol_inv), tfm; output=output, v=v, z=z)
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
end

# ╔═╡ bf72b4f5-5e6e-4e2f-b1da-45e389815a60
md"""
## 2D!
"""

# ╔═╡ fcab8ebd-6799-4ef8-90f8-959af5bfb375
let
    @testset "squared_euclidean 2D in-place" begin
        img = [
            0 1 1 1 0 0 0 1 1
            1 1 1 1 1 0 0 0 1
            1 0 0 0 1 0 0 1 1
            1 0 0 0 1 0 1 1 0
            1 0 0 0 1 1 0 1 0
            1 1 1 1 1 0 0 1 0
            0 1 1 1 0 0 0 0 1
        ]
        output, v, z = zeros(size(img)), ones(Int32, size(img)), ones(size(img) .+ 1)
        tfm = SquaredEuclidean()
        test = transform!(boolean_indicator(img), tfm; output=output, v=v, z=z)
        answer = [
            1.0  0.0  0.0  0.0  1.0  2.0  1.0  0.0  0.0
            0.0  0.0  0.0  0.0  0.0  1.0  2.0  1.0  0.0
            0.0  1.0  1.0  1.0  0.0  1.0  1.0  0.0  0.0
            0.0  1.0  4.0  1.0  0.0  1.0  0.0  0.0  1.0
            0.0  1.0  1.0  1.0  0.0  0.0  1.0  0.0  1.0
            0.0  0.0  0.0  0.0  0.0  1.0  1.0  0.0  1.0
            1.0  0.0  0.0  0.0  1.0  2.0  2.0  1.0  0.0
        ]
        @test test == answer
    end
end

# ╔═╡ 17eb24ab-6751-4b43-aa16-e779ae10e676
md"""
## 3D!
"""

# ╔═╡ 4c066e7c-4f1d-4795-bfb7-525b999e2a53
let
    @testset "squared_euclidean 3D in-place" begin
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
        output, v, z = zeros(size(vol_inv)), ones(Int32, size(vol_inv)), ones(size(vol_inv) .+ 1)
        tfm = SquaredEuclidean()
        test = transform!(boolean_indicator(vol_inv), tfm; output=output, v=v, z=z)
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
end

# ╔═╡ d8dc96f7-b61b-4021-8094-bd70a6b66d5e
md"""
## 2D Multi-Threaded
"""

# ╔═╡ 3b5f2010-48f7-4332-8742-60c71c4a01dd
md"""
## 3D Multi-Threaded
"""

# ╔═╡ Cell order:
# ╠═cd8bf944-2329-11ed-208f-1b2e91673a5e
# ╠═373a8802-bbac-4ab5-abe4-420ccbb61ea0
# ╟─81f78cda-0efa-4a74-bf4c-a39197d9b73f
# ╠═d3f4223e-f697-4a31-8825-66b1fae2b4f5
# ╠═d48cc7be-b769-4a64-970e-5982105fd382
# ╠═4ded097b-5e18-47af-bbbb-aa9fb626a9d1
# ╟─3b1fcab4-ca39-40c4-8cfa-410acfd6b006
# ╠═eafcd9e8-d691-4ae5-9e71-e999f8b6702c
# ╠═2f5755cb-e575-4ca9-a8b3-f2fd7846c068
# ╟─755aa045-73cd-425c-8057-50e65c342716
# ╠═afc433aa-ecc9-443a-bbd4-55bd48730937
# ╟─bf72b4f5-5e6e-4e2f-b1da-45e389815a60
# ╠═fcab8ebd-6799-4ef8-90f8-959af5bfb375
# ╟─17eb24ab-6751-4b43-aa16-e779ae10e676
# ╠═4c066e7c-4f1d-4795-bfb7-525b999e2a53
# ╟─d8dc96f7-b61b-4021-8094-bd70a6b66d5e
# ╟─3b5f2010-48f7-4332-8742-60c71c4a01dd
