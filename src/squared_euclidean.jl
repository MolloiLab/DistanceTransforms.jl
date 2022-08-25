### A Pluto.jl notebook ###
# v0.19.11

using Markdown
using InteractiveUtils

# ╔═╡ c7ef37d8-2330-11ed-006d-a16889a98cd1
# ╠═╡ show_logs = false
begin
	using Pkg
	Pkg.activate("..")
	using Revise
	using PlutoUI
	using Test
	using DistanceTransforms
	using FLoops
	using CUDA
end

# ╔═╡ 87dd183b-a462-4744-a800-3936791aea43
TableOfContents()

# ╔═╡ 2fef10c1-412b-4654-b321-122c6bf6095e
md"""
# `SquaredEuclidean`

```julia
struct SquaredEuclidean <: DistanceTransform end
```

Squared euclidean algorithm laid out in 'Distance Transforms of Sampled Functions' [Felzenszwalb and Huttenlocher] (DOI: 10.4086/toc.2012.v008a019)
"""

# ╔═╡ a741ae91-9a73-4047-a1af-a5d8ed1840ea
struct SquaredEuclidean <: DistanceTransforms.DistanceTransform end

# ╔═╡ f86ecd04-e1d8-4fd2-abfa-e811c4bb9c97
md"""
## Regular
"""

# ╔═╡ 423e6942-25cc-44e3-888f-c318a3523765
md"""
### 1D
```julia
transform(f::AbstractVector, tfm::SquaredEuclidean; 
output=zeros(length(f)), v=ones(Int32, length(f)), z=ones(length(f)+1))
```

Applies a squared euclidean distance transform to an input image.
Returns an array with spatial information embedded in the array 
elements.

"""

# ╔═╡ e540d01e-9a53-4659-b2bd-25ad26826927
function transform(f::AbstractVector, tfm::SquaredEuclidean; output=zeros(length(f)), v=ones(Int32, length(f)), z=ones(length(f)+1))
	z[1] = -1f10
	z[2] = 1f10
	k = 1; # Index of the rightmost parabola in the lower envelope
	for q = 2:length(f)
		s = ((f[q] + q^2) - (f[v[k]] + v[k]^2)) / (2*q - 2*v[k])
	    while s ≤ z[k]
	        k -= 1
	        s = ((f[q] + q^2) - (f[v[k]] + v[k]^2)) / (2*q - 2*v[k])
	    end
	    k += 1
	    v[k] = q
	    z[k] = s
		z[k+1] = 1f10
	end
	k = 1
	for q in 1:length(f)
	    while z[k+1] < q
	        k += 1
	    end
	    output[q] = (q-v[k])^2 + f[v[k]]
	end
	return output
end

# ╔═╡ c3aef793-7d05-4e62-82a4-1a8be1535c61
md"""
### 2D
```julia
transform(img::AbstractMatrix, tfm::SquaredEuclidean; 
output=zeros(size(img)), v=ones(Int32, size(img)), z=ones(size(img) .+ 1))
```

Applies a squared euclidean distance transform to an input image.
Returns an array with spatial information embedded in the array 
elements.
"""

# ╔═╡ 5a9e5e00-f508-4dfd-907a-4b391804d6b6
function transform(img::AbstractMatrix, tfm::SquaredEuclidean; output=zeros(size(img)), v=ones(Int32, size(img)), z=ones(size(img) .+ 1))
	for i in axes(img, 1)
	    output[i, :] = transform(img[i, :], tfm; output=output[i,:], v=v[i,:], z=z[i,:])
	end
	for j in axes(img, 2)
	    output[:, j] = transform(output[:, j], tfm; output=output[:,j], v=v[:,j], z=z[:,j])
	end
	return output
end

# ╔═╡ a52dbe18-335b-42a4-88b9-a360bf706a2f
md"""
### 3D
```julia
transform(vol::AbstractArray, tfm::SquaredEuclidean; 
output=zeros(size(vol)), v=ones(Int32, size(vol)), z=ones(size(vol) .+ 1))
```

Applies a squared euclidean distance transform to an input image.
Returns an array with spatial information embedded in the array 
elements.
"""

# ╔═╡ ac097d1a-4180-4327-8d0e-7a62bb2cd655
function transform(vol::AbstractArray, tfm::SquaredEuclidean;
    output=zeros(size(vol)), v=ones(Int32, size(vol)), z=ones(size(vol) .+ 1))
    for k in axes(vol, 3)
        output[:, :, k] = transform(vol[:, :, k], tfm; output=output[:, :, k], v=v[:, :, k], z=z[:, :, k])
    end
    for i in axes(vol, 1)
        for j in axes(vol, 2)
            output[i, j, :] = transform(output[i, j, :], tfm; output=output[i, j, :], v=v[i, j, :], z=z[i, j, :])
        end
    end
    return output
end

# ╔═╡ c604a85f-6995-4653-8ba3-e057b79b8259
md"""
## In-Place
"""

# ╔═╡ c1a492ad-0a6e-43e0-b185-51f6bd413c15
md"""
### 2D!

```julia
transform!(img::AbstractMatrix, tfm::SquaredEuclidean; output=zeros(size(img)), v=ones(Int32, size(img)), z=ones(size(img) .+ 1))
```

Applies a squared euclidean distance transform to an input image.
Returns an array with spatial information embedded in the array 
elements. In-place version of `transform(..., tfm::SquaredEuclidean)`
"""

# ╔═╡ 9fa52f01-82db-4071-b028-aee5567ff04c
function transform!(img::AbstractMatrix, tfm::SquaredEuclidean; output=zeros(size(img)), v=ones(Int32, size(img)), z=ones(size(img) .+ 1))
	for i in axes(img, 1)
		@views transform(img[i, :], tfm; output=output[i,:], v=fill!(v[i,:], 1), z=fill!(z[i,:], 1))
	end
	for j in axes(img, 2)
		@views transform(output[:, j], tfm; output=output[:,j], v=fill!(v[:,j], 1), z=fill!(z[:,j], 1))
	end
	return output
end

# ╔═╡ 03d0379b-d43a-411a-abc1-5a1ddea6a974
md"""
### 3D!

```julia
transform!(vol::AbstractArray, tfm::SquaredEuclidean; output=zeros(size(vol)), v=ones(Int32, size(vol)), z=ones(size(vol) .+ 1))
```

Applies a squared euclidean distance transform to an input image. Returns an array with spatial information embedded in the array elements. In-place version of transform(..., tfm::SquaredEuclidean)
"""

# ╔═╡ 622922b0-8bf0-46d1-acfe-ac4efba796c2
function transform!(vol::AbstractArray, tfm::SquaredEuclidean; output=zeros(size(vol)), v=ones(Int32, size(vol)), z=ones(size(vol) .+ 1))
    for k in axes(vol, 3)
        @views transform!(vol[:, :, k], tfm; output=output[:, :, k], v=fill!(v[:, :, k], 1), z=fill!(z[:, :, k], 1))
    end
    for i in axes(vol, 1)
        for j in axes(vol, 2)
            @views transform(output[i, j, :], tfm; output=output[i, j, :], v=fill!(v[i, j, :], 1), z=fill!(z[i, j, :], 1))
        end
    end
    return output
end

# ╔═╡ a7385641-6aa3-4cbc-94a0-969bb6ca42b4
md"""
## Multi-Threaded
"""

# ╔═╡ 3fa05ecf-5848-4aba-8f62-af1eadecbe5c
md"""
### 2D!
```julia
transform!(img::AbstractMatrix, tfm::SquaredEuclidean, nthreads; output=zeros(size(img)), v=ones(Int32, size(img)), z=ones(size(img) .+ 1))
```

Applies a squared euclidean distance transform to an input image.
Returns an array with spatial information embedded in the array 
elements. Multi-Threaded version of `transform!(..., tfm::SquaredEuclidean)`
"""

# ╔═╡ 4f4bd88d-9941-486c-b562-ec3fb9cadba7
function transform!(img::AbstractMatrix, tfm::SquaredEuclidean, nthreads; output=zeros(size(img)), v=ones(Int32, size(img)), z=ones(size(img) .+ 1))
	Threads.@threads for i in axes(img, 1)
		@views transform(img[i, :], tfm; output=output[i,:], v=fill!(v[i,:], 1), z=fill!(z[i,:], 1))
	end
	Threads.@threads for j in axes(img, 2)
		@views transform(output[:, j], tfm; output=output[:,j], v=fill!(v[:,j], 1), z=fill!(z[:,j], 1))
	end
	return output
end

# ╔═╡ 1df8eb42-e73c-4680-a8ce-c9d648ee8fe4
md"""
### 3D!
```julia
transform!(vol::AbstractArray, tfm::SquaredEuclidean, nthreads; output=zeros(size(vol)), v=ones(Int32, size(vol)), z=ones(size(vol) .+ 1))
```

Applies a squared euclidean distance transform to an input image. Returns an array with spatial information embedded in the array elements. Multi-Threaded version of `transform!(..., tfm::SquaredEuclidean)`
"""

# ╔═╡ e570b1d5-b2ff-48c0-8562-9cf2d350f9e4
function transform!(vol::AbstractArray, tfm::SquaredEuclidean, nthreads; output=zeros(size(vol)), v=ones(Int32, size(vol)), z=ones(size(vol) .+ 1))
    Threads.@threads for k in axes(vol, 3)
        @views transform!(vol[:, :, k], tfm; output=output[:, :, k], v=fill!(v[:, :, k], 1), z=fill!(z[:, :, k], 1))
    end
    Threads.@threads for i in axes(vol, 1)
        for j in axes(vol, 2)
            @views transform(output[i, j, :], tfm; output=output[i, j, :], v=fill!(v[i, j, :], 1), z=fill!(z[i, j, :], 1))
        end
    end
    return output
end

# ╔═╡ e559548d-1234-489a-b59e-db80aaf81798
# function transform!(vol::AbstractArray, tfm::SquaredEuclidean, nthreads; output=zeros(size(vol)), v=ones(Int32, size(vol)), z=ones(size(vol) .+ 1))
#     Threads.@threads for k in axes(vol, 3)
#         @views transform!(vol[:, :, k], tfm; output=output[:, :, k], v=fill!(v[:, :, k], 1), z=fill!(z[:, :, k], 1))
#     end
#     Threads.@threads for (i, j) in (axes(vol, 2), axes(vol, 1))
# 		@views transform(output[i, j, :], tfm; output=output[i, j, :], v=fill!(v[i, j, :], 1), z=fill!(z[i, j, :], 1))
#     end
#     return output
# end

# ╔═╡ a5a497b8-d1c6-4f00-8ab1-75dc9571cc0a
md"""
## GPU
"""

# ╔═╡ 2ba78c10-a3c2-460c-b30e-d86fc04c581f
md"""
### 2D!
"""

# ╔═╡ b8a0a7a0-887f-4c87-80ad-41a28aa8bf1c
function transform!(img::CuArray{T, 2}, tfm::SquaredEuclidean; output=CUDA.zeros(size(img)), v=CUDA.ones(size(img)), z=CUDA.ones(size(img) .+ 1)) where T
	@floop CUDAEx() for i in axes(img, 1)
		@views transform(img[i, :], tfm; output=output[i,:], v=v[i,:], z=z[i,:])
	end
	@floop CUDAEx() for j in axes(img, 2)
		@views transform(output[:, j], tfm; output=output[:,j], v=fill!(v[:,j], 1), z=fill!(z[:,j], 1))
	end
	return output
end

# ╔═╡ c5d95fe5-fb5f-441d-9392-1ac9eec06924
md"""
### 3D!
"""

# ╔═╡ 4333808c-d4b5-479c-b659-78fc0c17bf51
function transform!(vol::CuArray{T, 3}, tfm::SquaredEuclidean; output=CUDA.zeros(size(vol)), v=CUDA.ones(size(vol)), z=CUDA.ones(size(vol) .+ 1)) where T
    @floop CUDAEx() for k in axes(vol, 3)
        @views transform!(vol[:, :, k], tfm; output=output[:, :, k], v=v[:, :, k], z=z[:, :, k])
    end
	@floop CUDAEx() for i in axes(vol, 1)
		for j in axes(vol, 2)
			@views transform(output[i, j, :], tfm; output=output[i, j, :], v=fill!(v[i, j, :], 1), z=fill!(z[i, j, :], 1))
		end
    end
    return output
end

# ╔═╡ Cell order:
# ╠═c7ef37d8-2330-11ed-006d-a16889a98cd1
# ╠═87dd183b-a462-4744-a800-3936791aea43
# ╟─2fef10c1-412b-4654-b321-122c6bf6095e
# ╠═a741ae91-9a73-4047-a1af-a5d8ed1840ea
# ╟─f86ecd04-e1d8-4fd2-abfa-e811c4bb9c97
# ╟─423e6942-25cc-44e3-888f-c318a3523765
# ╠═e540d01e-9a53-4659-b2bd-25ad26826927
# ╟─c3aef793-7d05-4e62-82a4-1a8be1535c61
# ╠═5a9e5e00-f508-4dfd-907a-4b391804d6b6
# ╟─a52dbe18-335b-42a4-88b9-a360bf706a2f
# ╠═ac097d1a-4180-4327-8d0e-7a62bb2cd655
# ╟─c604a85f-6995-4653-8ba3-e057b79b8259
# ╟─c1a492ad-0a6e-43e0-b185-51f6bd413c15
# ╠═9fa52f01-82db-4071-b028-aee5567ff04c
# ╟─03d0379b-d43a-411a-abc1-5a1ddea6a974
# ╠═622922b0-8bf0-46d1-acfe-ac4efba796c2
# ╟─a7385641-6aa3-4cbc-94a0-969bb6ca42b4
# ╟─3fa05ecf-5848-4aba-8f62-af1eadecbe5c
# ╠═4f4bd88d-9941-486c-b562-ec3fb9cadba7
# ╟─1df8eb42-e73c-4680-a8ce-c9d648ee8fe4
# ╠═e570b1d5-b2ff-48c0-8562-9cf2d350f9e4
# ╠═e559548d-1234-489a-b59e-db80aaf81798
# ╟─a5a497b8-d1c6-4f00-8ab1-75dc9571cc0a
# ╟─2ba78c10-a3c2-460c-b30e-d86fc04c581f
# ╠═b8a0a7a0-887f-4c87-80ad-41a28aa8bf1c
# ╟─c5d95fe5-fb5f-441d-9392-1ac9eec06924
# ╠═4333808c-d4b5-479c-b659-78fc0c17bf51
