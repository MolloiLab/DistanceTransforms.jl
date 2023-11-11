using KernelAbstractions
using GPUArraysCore: AbstractGPUArray
using LoopVectorization

"""
## `boolean_indicator`
    
```julia
boolean_indicator(f::AbstractArray)
boolean_indicator(f::AbstractGPUArray)
boolean_indicator(f::BitArray)
```

Create a float representation of a boolean indicator array where `0` represents the background and `1` represents the foreground.
This function converts a logical array into a floating-point array where foreground values (logical `1`) are marked as `0.0f0` (float representation of `0`), and background values (logical `0`) are marked with a large float number `1.0f10`. This representation is useful in distance transform operations to differentiate between the foreground and the background.

#### Arguments
- `f`: An array of boolean values or an `AbstractGPUArray` of boolean values, where `true` indicates the foreground and `false` indicates the background.

#### Returns
- A floating-point array of the same dimensions as `f`, with foreground values set to `0.0f0` and background values set to `1.0f10`.

#### Performance
- If `f` is a `BitArray`, the conversion uses LoopVectorization.jl for a potential speedup. The warning check arguments are disabled for performance reasons.
- If `f` is an `AbstractGPUArray`, the computation is offloaded to the GPU using a custom kernel, `boolean_indicator_kernel`, which is expected to yield a significant speedup on compatible hardware.

#### Examples
```julia
f = BitArray([true, false, true, false])
f_float = boolean_indicator(f)
# f_float will be Float32[0.0f0, 1.0f10, 0.0f0, 1.0f10]

f_gpu = CUDA.zeros(Bool, 10) # assuming CUDA.jl is used for GPU arrays
f_gpu[5] = true
f_float_gpu = boolean_indicator(f_gpu)
# f_float_gpu will be a GPU array with the fifth element as 0.0f0 and others as 1.0f10
```

#### Notes
- The choice of `1.0f10` as the large number is arbitrary and can be adjusted if needed for specific applications.
- The `@turbo` macro from LoopVectorization.jl is used when `f` is a `BitArray` to unroll and vectorize the loop for optimal performance.
- When `f` is an `AbstractGPUArray`, the `boolean_indicator_kernel` kernel function is used to perform the operation in parallel on the GPU.
- The `KernelAbstractions.synchronize(backend)` call ensures that all GPU operations are completed before returning the result.

"""
function boolean_indicator(f)
    return @. ifelse(f == 0, 1.0f10, 0.0f0)
end

function boolean_indicator(f::BitArray)
    f_new = similar(f, Float32)
    @turbo warn_check_args = false for i in CartesianIndices(f_new)
        @inbounds f_new[i] = f[i] ? 0.0f0 : 1.0f10
    end
    return f_new
end

@kernel function boolean_indicator_kernel(f, output)
    i = @index(Global)
    output[i] = ifelse(f[i] == 0, 1.0f10, 0.0f0)
end

function boolean_indicator(f::AbstractGPUArray)
    backend = get_backend(f)
    output = similar(f, Float32)
    kernel = boolean_indicator_kernel(backend)
    kernel(f, output, ndrange=size(f))
    KernelAbstractions.synchronize(backend)
    return output
end

export boolean_indicator
