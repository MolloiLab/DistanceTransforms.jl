using GPUArraysCore: AbstractGPUVector, AbstractGPUMatrix, AbstractGPUArray
using KernelAbstractions

"""
## `transform!`

```julia
transform!(f::AbstractVector, output, v, z)
transform!(img::AbstractMatrix, output, v, z; threaded=true)
transform!(vol::AbstractArray{<:Real,3}, output, v, z, temp; threaded=true)
transform!(img::AbstractGPUMatrix, output, v, z)
```
In-place squared Euclidean distance transforms. These functions apply the transform to the input data and store the result in the `output` argument.

- The first function operates on vectors.
- The second function operates on matrices with optional threading.
- The third function operates on 3D arrays with optional threading.
- The fourth function is specialized for GPU matrices.

The intermediate arrays `v` and `z` (and `temp` for 3D arrays) are used for computation. The `threaded` parameter enables parallel computation on the CPU.

# Arguments
- `f`: Input vector, matrix, or 3D array.
- `output`: Preallocated array to store the result.
- `v`: Preallocated array for indices, matching the dimensions of `f`.
- `z`: Preallocated array for intermediate values, one element larger than `f`.
- `temp`: Preallocated array for intermediate values when transforming 3D arrays, matching the dimensions of `output`.

# Examples
```julia
f = rand([0f0, 1f0], 10)
f_bool = boolean_indicator(f)
output = similar(f)
v = ones(Int32, size(f))
z = ones(eltype(f), size(f) .+ 1)
transform!(f_bool, output, v, z)
```
"""
function transform!(f::AbstractVector, output, v, z)
    z[1] = -Inf32
    z[2] = Inf32

    k = 1
    @inbounds for q in 2:length(f)
        s = ((f[q] + q^2) - (f[v[k]] + v[k]^2)) / (2 * q - 2 * v[k])
        while s ≤ z[k]
            k -= 1
            s = ((f[q] + q^2) - (f[v[k]] + v[k]^2)) / (2 * q - 2 * v[k])
        end
        k += 1
        v[k] = q
        z[k] = s
        z[k+1] = Inf32
    end

    k = 1
    @inbounds for q in 1:length(f)
        while z[k+1] < q
            k += 1
        end
        output[q] = (q - v[k])^2 + f[v[k]]
    end
end

function transform!(img::AbstractMatrix, output, v, z; threaded=true)
    if threaded
        Threads.@threads for i in CartesianIndices(@view(img[:, 1]))
            @views transform!(img[i, :], output[i, :], v[i, :], z[i, :])
        end

        Threads.@threads for j in CartesianIndices(@view(img[1, :]))
            @views transform!(
                output[:, j], output[:, j], fill!(v[:, j], 1), fill!(z[:, j], 1)
            )
        end
    else
        for i in CartesianIndices(@view(img[:, 1]))
            @views transform!(img[i, :], output[i, :], v[i, :], z[i, :])
        end

        for j in CartesianIndices(@view(img[1, :]))
            @views transform!(
                output[:, j], output[:, j], fill!(v[:, j], 1), fill!(z[:, j], 1)
            )
        end
    end
end

function transform!(vol::AbstractArray{<:Real,3}, output, v, z, temp; threaded=true)
    if threaded
        Threads.@threads for i in CartesianIndices(@view(vol[:, :, 1]))
            @views transform!(
                vol[i, :], output[i, :], v[i, :], z[i, :]
            )
        end

        Threads.@threads for j in CartesianIndices(@view(vol[1, :, :]))
            @views transform!(
                output[:, j], temp[:, j], fill!(v[:, j], 1), fill!(z[:, j], 1)
            )
        end

        Threads.@threads for k in CartesianIndices(@view(vol[:, 1, :]))
            @views transform!(
                temp[k[1], :, k[2]], output[k[1], :, k[2]], fill!(v[k[1], :, k[2]], 1), fill!(z[k[1], :, k[2]], 1)
            )
        end
    else
        for i in CartesianIndices(@view(vol[:, :, 1]))
            @views transform!(
                vol[i, :], output[i, :], v[i, :], z[i, :]
            )
        end

        for j in CartesianIndices(@view(vol[1, :, :]))
            @views transform!(
                output[:, j], temp[:, j], fill!(v[:, j], 1), fill!(z[:, j], 1)
            )
        end

        for k in CartesianIndices(@view(vol[:, 1, :]))
            @views transform!(
                temp[k[1], :, k[2]], output[k[1], :, k[2]], fill!(v[k[1], :, k[2]], 1), fill!(z[k[1], :, k[2]], 1)
            )
        end
    end
end

@kernel function transform_kernel_cols!(img, output, v, z)
    i = @index(Global)
    @views transform!(img[i, :], output[i, :], v[i, :], z[i, :])
end

@kernel function transform_kernel_rows!(img, output, v, z)
    j = @index(Global)
    @views transform!(img[:, j], output[:, j], fill!(v[:, j], 1), fill!(z[:, j], 1))
end

function transform!(img::AbstractGPUMatrix, output, v, z)
    backend = get_backend(img)
    kernel_cols = transform_kernel_cols!(backend)
    kernel_cols(img, output, v, z, ndrange=size(img, 1))

    B = similar(output)
    copyto!(B, output)

    kernel_rows = transform_kernel_rows!(backend)
    kernel_rows(B, output, v, z, ndrange=size(img, 2))
    KernelAbstractions.synchronize(backend)
end

export transform!

"""
## `transform`

```julia
transform(f::AbstractVector)
transform(img::AbstractMatrix; threaded=true)
transform(vol::AbstractArray{<:Real,3}; threaded=true)
transform(img::AbstractGPUMatrix)
```

Non-in-place squared Euclidean distance transforms that return a new array with the result. They allocate necessary intermediate arrays internally.

- The first function operates on vectors.
- The second function operates on matrices with optional threading.
- The third function operates on 3D arrays with optional threading.
- The fourth function is specialized for GPU matrices.

The `threaded` parameter can be used to enable or disable parallel computation on the CPU.

# Arguments
- `f/img/vol`: Input vector, matrix, or 3D array to be transformed.

# Examples
```julia
f = rand([0f0, 1f0], 10)
f_bool = boolean_indicator(f)
f_tfm = transform(f_bool)
```
"""
function transform(f::AbstractVector)
    output = similar(f, eltype(f))
    v = ones(Int32, length(f))
    z = ones(eltype(f), length(f) + 1)

    transform!(f, output, v, z)
    return output
end

function transform(img::AbstractMatrix; threaded=true)
    output = similar(img, eltype(img))
    v = ones(Int32, size(img))
    z = ones(eltype(img), size(img) .+ 1)

    transform!(img, output, v, z; threaded=threaded)
    return output
end

function transform(vol::AbstractArray{<:Real,3}; threaded=true)
    output = similar(vol, eltype(vol))
    v = ones(Int32, size(vol))
    z = ones(eltype(vol), size(vol) .+ 1)
    temp = similar(output)

    transform!(vol, output, v, z, temp; threaded=threaded)
    return output
end

function transform(img::AbstractGPUMatrix)
	backend = KernelAbstractions.get_backend(img)
	output = similar(img, Float32)
	v = KernelAbstractions.ones(backend, Int32, size(img))
	z = KernelAbstractions.ones(backend, Float32, size(img) .+ 1)
	transform!(img, output, v, z)
	
	return output
end

export transform