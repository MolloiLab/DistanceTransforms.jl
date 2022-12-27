### A Pluto.jl notebook ###
# v0.19.8

using Markdown
using InteractiveUtils

# ╔═╡ 4b4b6c39-e155-462a-b0ad-65688c949577
# ╠═╡ show_logs = false
begin
	using Pkg
	Pkg.activate("..")
	using Revise
	using PlutoUI
	using Test
	using CUDA
	using DistanceTransforms
	using LoopVectorization
end

# ╔═╡ 4db00b10-0440-4f28-bf91-01dca21847b0
TableOfContents()

# ╔═╡ 64661943-81d4-40c5-9479-fcc381544185
md"""
## Boolean Indicator
"""

# ╔═╡ 5f558b4b-d109-45d7-b2f8-fa03c0ff5db6
begin
"""
```julia
boolean_indicator(f)
```

If `f` is a boolean indicator where 0's correspond to background and 1s correspond to the foreground, then mark background pixels with a large number `1e10`
"""
boolean_indicator(f) = @. ifelse(f < 0.5, 1.0f10, 0.0f0)

"""
```julia
boolean_indicator(f::BitArray)
```

If `f` is a boolean indicator where 0's correspond to background and 1s correspond to the foreground, then mark background pixels with a large number `1e10`. Uses LoopVectorization.jl for speed up
"""
function boolean_indicator(f::BitArray)
	f_new = similar(f, Float32)
	@turbo warn_check_args=false for i in CartesianIndices(f_new)
		   @inbounds f_new[i] = f[i] ? 0f0 : 1f10
	end
	return f_new
end
end

# ╔═╡ Cell order:
# ╠═4b4b6c39-e155-462a-b0ad-65688c949577
# ╠═4db00b10-0440-4f28-bf91-01dca21847b0
# ╟─64661943-81d4-40c5-9479-fcc381544185
# ╠═5f558b4b-d109-45d7-b2f8-fa03c0ff5db6
