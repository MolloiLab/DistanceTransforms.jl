using LoopVectorization

"""
## `boolean_indicator`
```julia
boolean_indicator(f)
boolean_indicator(f::BitArray)
```

If `f` is a boolean indicator where 0's correspond to background and 1s correspond to the foreground, then mark background pixels with a large number `1e10`.
Uses LoopVectorization.jl for speed up if array is `BitArray`
"""
boolean_indicator(f) = @. ifelse(f < 0.5, 1.0f10, 0.0f0)

function boolean_indicator(f::BitArray)
	f_new = similar(f, Float32)
	@turbo warn_check_args=false for i in CartesianIndices(f_new)
		   @inbounds f_new[i] = f[i] ? 0f0 : 1f10
	end
	return f_new
end

export boolean_indicator