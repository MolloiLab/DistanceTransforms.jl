### A Pluto.jl notebook ###
# v0.19.14

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
	using FoldsThreads
end

# ╔═╡ 87dd183b-a462-4744-a800-3936791aea43
TableOfContents()

# ╔═╡ 2fef10c1-412b-4654-b321-122c6bf6095e
md"""
# `Felzenszwalb`
"""

# ╔═╡ a741ae91-9a73-4047-a1af-a5d8ed1840ea
"""
```julia
struct Felzenszwalb <: DistanceTransform end
```

Squared euclidean algorithm laid out in 'Distance Transforms of Sampled Functions' [Felzenszwalb and Huttenlocher] (DOI: 10.4086/toc.2012.v008a019)
"""
struct Felzenszwalb <: DistanceTransform end

# ╔═╡ f86ecd04-e1d8-4fd2-abfa-e811c4bb9c97
md"""
## Regular
"""

# ╔═╡ 423e6942-25cc-44e3-888f-c318a3523765
md"""
### 1D
"""

# ╔═╡ e540d01e-9a53-4659-b2bd-25ad26826927
"""
```julia
transform(f::AbstractVector, tfm::Felzenszwalb; 
output=zeros(length(f)), v=ones(Int32, length(f)), z=ones(length(f)+1))
```

Applies a squared euclidean distance transform to an input image.
Returns an array with spatial information embedded in the array 
elements.
"""
function transform(f::AbstractVector, tfm::Felzenszwalb; output=similar(f), v=ones(Int32, length(f)), z=ones(Float32, length(f)+1))
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
"""

# ╔═╡ 5a9e5e00-f508-4dfd-907a-4b391804d6b6
"""
```julia
transform(img::AbstractMatrix, tfm::Felzenszwalb; 
output=zeros(size(img)), v=ones(Int32, size(img)), z=ones(size(img) .+ 1))
```

Applies a squared euclidean distance transform to an input image.
Returns an array with spatial information embedded in the array 
elements.
"""
function transform(img::AbstractMatrix, tfm::Felzenszwalb; output=similar(img), v=ones(Int32, size(img)), z=ones(Float32, size(img) .+ 1))
	# 1
	for k in CartesianIndices(@view(img[:, 1]))
	    @views transform(img[k,:], tfm; output=output[k,:], v=v[k,:], z=z[k,:])
	end
	output2 = similar(output)
	# 2
	for k in CartesianIndices(@view(img[1, :]))
	    @views transform(output[:,k], tfm; output=output2[:,k], v=fill!(v[:,k], 1), z=fill!(z[:,k], 1))
	end
	# end
	return output2
end

# ╔═╡ a52dbe18-335b-42a4-88b9-a360bf706a2f
md"""
### 3D
"""

# ╔═╡ ac097d1a-4180-4327-8d0e-7a62bb2cd655
"""
```julia
transform(vol::AbstractArray, tfm::Felzenszwalb; 
output=zeros(size(vol)), v=ones(Int32, size(vol)), z=ones(size(vol) .+ 1))
```

Applies a squared euclidean distance transform to an input image.
Returns an array with spatial information embedded in the array 
elements.
"""
function transform(vol::AbstractArray, tfm::Felzenszwalb;
    output=similar(vol), v=ones(Int32, size(vol)), z=ones(Float32, size(vol) .+ 1))
	# 1
    for k in CartesianIndices(@view(vol[1,:,:]))
        @views transform(vol[:, k], tfm; output=output[:, k], v=v[:, k], z=z[:, k])
    end
	output2 = similar(output)
	# 2
    for k in CartesianIndices(@view(vol[:,1,:]))
        @views transform(output[k[1], :, k[2]], tfm; output=output2[k[1], :, k[2]], v=fill!(v[k[1], :, k[2]], 1), z=fill!(z[k[1], :, k[2]], 1))
    end
	# 3
    for k in CartesianIndices(@view(vol[:,:,1]))
        @views transform(output2[k, :], tfm; output=output[k, :], v=fill!(v[k, :], 1), z=fill!(z[k, :], 1))
    end
    return output
end

# ╔═╡ a7385641-6aa3-4cbc-94a0-969bb6ca42b4
md"""
## Multi-Threaded
"""

# ╔═╡ 3fa05ecf-5848-4aba-8f62-af1eadecbe5c
md"""
### 2D
"""

# ╔═╡ 4f4bd88d-9941-486c-b562-ec3fb9cadba7
"""

```julia
transform!(img::AbstractMatrix, tfm::Felzenszwalb, nthreads; output=zeros(size(img)), v=ones(Int32, size(img)), z=ones(size(img) .+ 1))
```

Applies a squared euclidean distance transform to an input image.
Returns an array with spatial information embedded in the array 
elements. Multi-Threaded version of `transform!(..., tfm::Felzenszwalb)`
"""
function transform(img::AbstractMatrix, tfm::Felzenszwalb, nthreads::Number; output=similar(img), v=ones(Int32, size(img)), z=ones(Float32, size(img) .+ 1))
	# 1
	Threads.@threads for k in CartesianIndices(@view(img[:, 1]))
	    @views transform(img[k,:], tfm; output=output[k,:], v=v[k,:], z=z[k,:])
	end
	output2 = similar(output)
	# 2
	Threads.@threads for k in CartesianIndices(@view(img[1, :]))
	    @views transform(output[:,k], tfm; output=output2[:,k], v=fill!(v[:,k], 1), z=fill!(z[:,k], 1))
	end
	# end
	return output2
end

# ╔═╡ 1df8eb42-e73c-4680-a8ce-c9d648ee8fe4
md"""
### 3D
"""

# ╔═╡ e570b1d5-b2ff-48c0-8562-9cf2d350f9e4
"""

```julia
transform!(vol::AbstractArray, tfm::Felzenszwalb, nthreads; output=zeros(size(vol)), v=ones(Int32, size(vol)), z=ones(size(vol) .+ 1))
```

Applies a squared euclidean distance transform to an input image. Returns an array with spatial information embedded in the array elements. Multi-Threaded version of `transform!(..., tfm::Felzenszwalb)`
"""
function transform(vol::AbstractArray, tfm::Felzenszwalb, nthreads::Number; 
    output=similar(vol), v=ones(Int32, size(vol)), z=ones(Float32, size(vol) .+ 1))
	# 1
    Threads.@threads for k in CartesianIndices(@view(vol[1,:,:]))
        @views transform(vol[:, k], tfm; output=output[:, k], v=v[:, k], z=z[:, k])
    end
	output2 = similar(output)
	# 2
    Threads.@threads for k in CartesianIndices(@view(vol[:,1,:]))
        @views transform(output[k[1], :, k[2]], tfm; output=output2[k[1], :, k[2]], v=fill!(v[k[1], :, k[2]], 1), z=fill!(z[k[1], :, k[2]], 1))
    end
	# 3
    Threads.@threads for k in CartesianIndices(@view(vol[:,:,1]))
        @views transform(output2[k, :], tfm; output=output[k, :], v=fill!(v[k, :], 1), z=fill!(z[k, :], 1))
    end
    return output
end

# ╔═╡ a5a497b8-d1c6-4f00-8ab1-75dc9571cc0a
md"""
## GPU
"""

# ╔═╡ 824a5bf4-7956-4582-9019-4fb68717229a
md"""
### 2D
"""

# ╔═╡ b8a0a7a0-887f-4c87-80ad-41a28aa8bf1c
"""
```julia
transform!(img::CuArray{T, 2}, tfm::Felzenszwalb; output=CUDA.zeros(size(img)), v=CUDA.ones(size(img)), z=CUDA.ones(size(img) .+ 1)) where T
```

Applies a squared euclidean distance transform to an input image.
Returns an array with spatial information embedded in the array 
elements. GPU version of `transform!(..., tfm::Felzenszwalb)`
"""
function transform(img::CuArray{T, 2}, tfm::Felzenszwalb; output=similar(img), v=CUDA.ones(size(img)), z=CUDA.ones(size(img) .+ 1)) where T
	# 1
	@floop CUDAEx() for k in CartesianIndices(@view(img[:, 1]))
	    @views transform(img[k,:], tfm; output=output[k,:], v=v[k,:], z=z[k,:])
	end
	output2 = similar(output)
	# 2
	@floop CUDAEx() for k in CartesianIndices(@view(img[1, :]))
	    @views transform(output[:,k], tfm; output=output2[:,k], v=fill!(v[:,k], 1), z=fill!(z[:,k], 1))
	end
	# end
	return output2
end

# ╔═╡ f5c97a87-04e1-43e2-9b06-249443b3826c
md"""
### 3D
"""

# ╔═╡ 4333808c-d4b5-479c-b659-78fc0c17bf51
"""
```julia
transform!(vol::CuArray{T, 3}, tfm::Felzenszwalb; output=CUDA.zeros(size(vol)), v=CUDA.ones(size(vol)), z=CUDA.ones(size(vol) .+ 1)) where T
```

Applies a squared euclidean distance transform to an input image.
Returns an array with spatial information embedded in the array 
elements. GPU version of `transform!(..., tfm::Felzenszwalb)`
"""
function transform(vol::CuArray{T, 3}, tfm::Felzenszwalb; output=similar(vol), v=CUDA.ones(size(vol)), z=CUDA.ones(size(vol) .+ 1)) where T
	# 1
    @floop CUDAEx() for k in CartesianIndices(@view(vol[1,:,:]))
        @views transform(vol[:, k], tfm; output=output[:, k], v=v[:, k], z=z[:, k])
    end
	output2 = similar(output)
	# 2
    @floop CUDAEx() for k in CartesianIndices(@view(vol[:,1,:]))
        @views transform(output[k[1], :, k[2]], tfm; output=output2[k[1], :, k[2]], v=fill!(v[k[1], :, k[2]], 1), z=fill!(z[k[1], :, k[2]], 1))
    end
	# 3
    @floop CUDAEx() for k in CartesianIndices(@view(vol[:,:,1]))
        @views transform(output2[k, :], tfm; output=output[k, :], v=fill!(v[k, :], 1), z=fill!(z[k, :], 1))
    end
    return output
end

# ╔═╡ 3bb1c028-4f7a-414a-86bd-a4522af64957
md"""
## Various Multi-Threading
"""

# ╔═╡ 9d0f0cde-66b0-4fb5-8ff3-c96e561a5ba2
md"""
### 2D
"""

# ╔═╡ 0726f423-2044-4d78-8eca-9433e9f1dc95
"""
```julia
transform!(img::AbstractMatrix, tfm::Felzenszwalb, ex; output=zeros(size(img)), v=ones(size(img)), z=ones(size(img) .+ 1))
```

Applies a squared euclidean distance transform to an input image.
Returns an array with spatial information embedded in the array 
elements. Multi-Threaded version of `transform!(..., tfm::Felzenszwalb)` but utilizes FoldsThreads.jl for different threaded executors. `ex`=(FoldsThreads.DepthFirstEx(), FoldsThreads.NonThreadedEx(), FoldsThreads.WorkStealingEx())
"""
function transform(img::AbstractMatrix, tfm::Felzenszwalb, ex; output=similar(img), v=ones(size(img)), z=ones(size(img) .+ 1))
	# 1
	@floop ex for k in CartesianIndices(@view(img[:, 1]))
	    @views transform(img[k,:], tfm; output=output[k,:], v=v[k,:], z=z[k,:])
	end
	output2 = similar(output)
	# 2
	@floop ex for k in CartesianIndices(@view(img[1, :]))
	    @views transform(output[:,k], tfm; output=output2[:,k], v=fill!(v[:,k], 1), z=fill!(z[:,k], 1))
	end
	# end
	return output2
end

# ╔═╡ c18c3ed9-1321-4884-8dad-70f4e24adcc5
md"""
### 3D
"""

# ╔═╡ 07e9ebfa-a287-45d9-a38a-aab7690992f9
"""
```julia
transform!(vol::AbstractArray, tfm::Felzenszwalb, ex; output=zeros(size(vol)), v=ones(size(vol)), z=ones(size(vol) .+ 1))
```

Applies a squared euclidean distance transform to an input image.
Returns an array with spatial information embedded in the array 
elements. Multi-Threaded version of `transform!(..., tfm::Felzenszwalb)` but utilizes FoldsThreads.jl for different threaded executors. `ex`=(FoldsThreads.DepthFirstEx(), FoldsThreads.NonThreadedEx(), FoldsThreads.WorkStealingEx())
"""
function transform(vol::AbstractArray, tfm::Felzenszwalb, ex; output=similar(vol), v=ones(size(vol)), z=ones(size(vol) .+ 1))
	# 1
    @floop ex for k in CartesianIndices(@view(vol[1,:,:]))
        @views transform(vol[:, k], tfm; output=output[:, k], v=v[:, k], z=z[:, k])
    end
	output2 = similar(output)
	# 2
    @floop ex for k in CartesianIndices(@view(vol[:,1,:]))
        @views transform(output[k[1], :, k[2]], tfm; output=output2[k[1], :, k[2]], v=fill!(v[k[1], :, k[2]], 1), z=fill!(z[k[1], :, k[2]], 1))
    end
	# 3
    @floop ex for k in CartesianIndices(@view(vol[:,:,1]))
        @views transform(output2[k, :], tfm; output=output[k, :], v=fill!(v[k, :], 1), z=fill!(z[k, :], 1))
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
# ╟─a7385641-6aa3-4cbc-94a0-969bb6ca42b4
# ╟─3fa05ecf-5848-4aba-8f62-af1eadecbe5c
# ╠═4f4bd88d-9941-486c-b562-ec3fb9cadba7
# ╟─1df8eb42-e73c-4680-a8ce-c9d648ee8fe4
# ╠═e570b1d5-b2ff-48c0-8562-9cf2d350f9e4
# ╟─a5a497b8-d1c6-4f00-8ab1-75dc9571cc0a
# ╟─824a5bf4-7956-4582-9019-4fb68717229a
# ╠═b8a0a7a0-887f-4c87-80ad-41a28aa8bf1c
# ╟─f5c97a87-04e1-43e2-9b06-249443b3826c
# ╠═4333808c-d4b5-479c-b659-78fc0c17bf51
# ╟─3bb1c028-4f7a-414a-86bd-a4522af64957
# ╟─9d0f0cde-66b0-4fb5-8ff3-c96e561a5ba2
# ╠═0726f423-2044-4d78-8eca-9433e9f1dc95
# ╟─c18c3ed9-1321-4884-8dad-70f4e24adcc5
# ╠═07e9ebfa-a287-45d9-a38a-aab7690992f9
