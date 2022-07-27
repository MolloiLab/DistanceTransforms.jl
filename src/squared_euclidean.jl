"""
    struct SquaredEuclidean <: DistanceTransform end

Squared euclidean algorithm laid out in 'Distance Transforms of Sampled Functions' 
[Felzenszwalb and Huttenlocher] (DOI: 10.4086/toc.2012.v008a019)
"""
struct SquaredEuclidean <: DistanceTransform end

## 1D
"""
    transform(f::AbstractVector, tfm::SquaredEuclidean; output=zeros(length(f)), v=ones(Int32, length(f)), z=ones(length(f)+1))
    transform(img::AbstractMatrix, tfm::SquaredEuclidean; output=zeros(size(img)), v=ones(Int32, size(img)), z=ones(size(img) .+ 1))
    transform(vol::AbstractArray, tfm::SquaredEuclidean; output=zeros(size(vol)), v=ones(Int32, size(vol)), z=ones(size(vol) .+ 1))

Applies a squared euclidean distance transform to an input image.
Returns an array with spatial information embedded in the array 
elements.
"""
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

## 2D
function transform(img::AbstractMatrix, tfm::SquaredEuclidean; output=zeros(size(img)), v=ones(Int32, size(img)), z=ones(size(img) .+ 1))
	for i in axes(img, 1)
	    output[i, :] = transform(img[i, :], tfm; output=output[i,:], v=v[i,:], z=z[i,:])
	end
	for j in axes(img, 2)
	    output[:, j] = transform(output[:, j], tfm; output=output[:,j], v=v[:,j], z=z[:,j])
	end
	return output
end

## 3D
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

## In-Place/Multi-Threaded/GPU
## 2D
"""
    transform!(img::AbstractMatrix, tfm::SquaredEuclidean; output=zeros(size(img)), v=ones(Int32, size(img)), z=ones(size(img) .+ 1))
    transform!(img::AbstractMatrix, tfm::SquaredEuclidean, nthreads; output=zeros(size(img)), v=ones(Int32, size(img)), z=ones(size(img) .+ 1))
    transform!(vol::AbstractArray, tfm::SquaredEuclidean; output=zeros(size(vol)), v=ones(Int32, size(vol)), z=ones(size(vol) .+ 1))
    transform!(vol::AbstractArray, tfm::SquaredEuclidean, nthreads; output=zeros(size(vol)), v=ones(Int32, size(vol)), z=ones(size(vol) .+ 1))

Applies a squared euclidean distance transform to an input image.
Returns an array with spatial information embedded in the array 
elements. In-place version of `transform(..., tfm::SquaredEuclidean)`
with optional multi-threading or GPU support
"""
function transform!(img::AbstractMatrix, tfm::SquaredEuclidean; output=zeros(size(img)), v=ones(Int32, size(img)), z=ones(size(img) .+ 1))
	for i in axes(img, 1)
		@views transform(img[i, :], tfm; output=output[i,:], v=fill!(v[i,:], 1), z=fill!(z[i,:], 1))
	end
	for j in axes(img, 2)
		@views transform(output[:, j], tfm; output=output[:,j], v=fill!(v[:,j], 1), z=fill!(z[:,j], 1))
	end
	return output
end

function transform!(img::AbstractMatrix, tfm::SquaredEuclidean, nthreads; output=zeros(size(img)), v=ones(Int32, size(img)), z=ones(size(img) .+ 1))
	Threads.@threads for i in axes(img, 1)
		@views transform(img[i, :], tfm; output=output[i,:], v=fill!(v[i,:], 1), z=fill!(z[i,:], 1))
	end
	Threads.@threads for j in axes(img, 2)
		@views transform(output[:, j], tfm; output=output[:,j], v=fill!(v[:,j], 1), z=fill!(z[:,j], 1))
	end
	return output
end

# function transform!(img::CuArray{T,2}, tfm::SquaredEuclidean; output=CUDA.zeros(size(img)), v=CUDA.ones(size(img)), z=CUDA.ones(size(img) .+ 1))
# 	@floop CUDAEx() for i in axes(img, 1)
# 		@views transform(img[i, :], tfm; output=output[i,:], v=fill!(v[i,:], 1), z=fill!(z[i,:], 1))
# 	end
# 	@floop CUDAEx() for j in axes(img, 2)
# 		@views transform(output[:, j], tfm; output=output[:,j], v=fill!(v[:,j], 1), z=fill!(z[:,j], 1))
# 	end
# 	return output
# end

## 3D
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

function transform!(vol::AbstractArray, tfm::SquaredEuclidean, nthreads; output=zeros(size(vol)), v=ones(Int32, size(vol)), z=ones(size(vol) .+ 1))
    Threads.@threads for k in axes(vol, 3)
        @views transform!(vol[:, :, k], tfm; output=output[:, :, k], v=fill!(v[:, :, k], 1), z=fill!(z[:, :, k], 1))
    end
    Threads.@threads for i in axes(vol, 1)
        Threads.@threads for j in axes(vol, 2)
            @views transform(output[i, j, :], tfm; output=output[i, j, :], v=fill!(v[i, j, :], 1), z=fill!(z[i, j, :], 1))
        end
    end
    return output
end

# function transform!(vol::CuArray{T,3}, tfm::SquaredEuclidean; output=CUDA.zeros(size(vol)), v=CUDA.ones(size(vol)), z=CUDA.ones(size(vol) .+ 1))
#     @floop CUDAEx() for k in axes(vol, 3)
#         @views transform!(vol[:, :, k], tfm; output=output[:, :, k], v=fill!(v[:, :, k], 1), z=fill!(z[:, :, k], 1))
#     end
#     @floop CUDAEx() for i in axes(vol, 1)
#         @floop CUDAEx() for j in axes(vol, 2)
#             @views transform(output[i, j, :], tfm; output=output[i, j, :], v=fill!(v[i, j, :], 1), z=fill!(z[i, j, :], 1))
#         end
#     end
#     return output
# end