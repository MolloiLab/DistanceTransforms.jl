using FLoops
using CUDA

"""
## Felzenszwalb

```julia
struct Felzenszwalb <: DistanceTransform end
```

Squared euclidean algorithm laid out in 'Distance Transforms of Sampled Functions' [Felzenszwalb and Huttenlocher] (DOI: 10.4086/toc.2012.v008a019)
"""
struct Felzenszwalb <: DistanceTransform end

"""
## transform (Felzenszwalb)

```julia
transform(f::AbstractVector, tfm::Felzenszwalb; 
output=zeros(length(f)), v=ones(Int32, length(f)), z=ones(length(f)+1))

transform(img::AbstractMatrix, tfm::Felzenszwalb; 
output=zeros(size(img)), v=ones(Int32, size(img)), z=ones(size(img) .+ 1))

transform(vol::AbstractArray, tfm::Felzenszwalb; 
output=zeros(size(vol)), v=ones(Int32, size(vol)), z=ones(size(vol) .+ 1))

transform(img::AbstractMatrix, tfm::Felzenszwalb, nthreads; 
output=zeros(size(img)), v=ones(Int32, size(img)), z=ones(size(img) .+ 1))

transform(vol::AbstractArray, tfm::Felzenszwalb, nthreads; 
output=zeros(size(vol)), v=ones(Int32, size(vol)), z=ones(size(vol) .+ 1))

transform(img::CuArray{T, 2}, tfm::Felzenszwalb; 
output=CUDA.zeros(size(img)), v=CUDA.ones(size(img)), z=CUDA.ones(size(img) .+ 1))

transform!(vol::CuArray{T, 3}, tfm::Felzenszwalb; 
output=CUDA.zeros(size(vol)), v=CUDA.ones(size(vol)), z=CUDA.ones(size(vol) .+ 1)) where T

transform(img::AbstractMatrix, tfm::Felzenszwalb, ex; 
output=zeros(size(img)), v=ones(size(img)), z=ones(size(img) .+ 1))

transform!(vol::AbstractArray, tfm::Felzenszwalb, ex; 
output=zeros(size(vol)), v=ones(size(vol)), z=ones(size(vol) .+ 1))

```

Applies a squared euclidean distance transform to an input image.
Returns an array with spatial information embedded in the array 
elements.

Applies a squared euclidean distance transform to an input image.
Returns an array with spatial information embedded in the array 
elements. Multi-Threaded version of `transform!(..., tfm::Felzenszwalb)`

Applies a squared euclidean distance transform to an input image.
Returns an array with spatial information embedded in the array 
elements. GPU version of `transform!(..., tfm::Felzenszwalb)`

Applies a squared euclidean distance transform to an input image.
Returns an array with spatial information embedded in the array 
elements. Multi-Threaded version of `transform!(..., tfm::Felzenszwalb)` 
but utilizes FoldsThreads.jl for different threaded executors. 
`ex`=(FoldsThreads.DepthFirstEx(), FoldsThreads.NonThreadedEx(), FoldsThreads.WorkStealingEx())
"""
function transform(f::AbstractVector, tfm::Felzenszwalb; output=similar(f, Float32), v=ones(Int32, length(f)), z=ones(Float32, length(f)+1))
	z[1] = -1f10
	z[2] = 1f10
	k = 1 # Index of the rightmost parabola in the lower envelope
	for q in eachindex(f)[2:end]
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
	for q in eachindex(f)
	    while z[k+1] < q
	        k += 1
	    end
	    output[q] = (q-v[k])^2 + f[v[k]]
	end
	return output
end

function transform(img::AbstractMatrix, tfm::Felzenszwalb; output=similar(img, Float32), v=ones(Int32, size(img)), z=ones(Float32, size(img) .+ 1))
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

function transform(vol::AbstractArray, tfm::Felzenszwalb;
    output=similar(vol, Float32), v=ones(Int32, size(vol)), z=ones(Float32, size(vol) .+ 1))
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

function transform(img::AbstractMatrix, tfm::Felzenszwalb, nthreads::Number; output=similar(img, Float32), v=ones(Int32, size(img)), z=ones(Float32, size(img) .+ 1))
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

function transform(vol::AbstractArray, tfm::Felzenszwalb, nthreads::Number; 
    output=similar(vol, Float32), v=ones(Int32, size(vol)), z=ones(Float32, size(vol) .+ 1))
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

function transform(img::CuArray{T, 2}, tfm::Felzenszwalb; output=similar(img, Float32), v=CUDA.ones(Int32, size(img)), z=CUDA.ones(Float32, size(img) .+ 1)) where T
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

function transform(vol::CuArray{T, 3}, tfm::Felzenszwalb; output=similar(vol, Float32), v=CUDA.ones(Int32, size(vol)), z=CUDA.ones(Float32, size(vol) .+ 1)) where T
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
