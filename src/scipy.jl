### A Pluto.jl notebook ###
# v0.19.22

using Markdown
using InteractiveUtils

# ╔═╡ 4ced48c7-b2dc-4c25-8c04-81fa5a920238
# ╠═╡ show_logs = false
begin
	using Pkg
	Pkg.activate("..")
	using Revise, PlutoUI, PythonCall, DistanceTransforms
end

# ╔═╡ c446ce9f-2073-41ff-9533-624f0e4ab841
md"""
# `Scipy`
"""

# ╔═╡ 8efe6999-2f0b-4b24-99b8-81acd00eeac1
"""
```julia
struct Scipy <: DistanceTransform end
```

Exact euclidean algorithm ported from [Scipy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.distance_transform_edt.html)
"""
struct Scipy <: DistanceTransform end

# ╔═╡ dbc31ea1-9559-4e25-a2f4-21e3fae32908
function transform(array, tfm::Scipy)
	return pyconvert(Array{Float32}, scipy.ndimage.distance_transform_edt(array))
end

# ╔═╡ Cell order:
# ╠═4ced48c7-b2dc-4c25-8c04-81fa5a920238
# ╟─c446ce9f-2073-41ff-9533-624f0e4ab841
# ╠═8efe6999-2f0b-4b24-99b8-81acd00eeac1
# ╠═dbc31ea1-9559-4e25-a2f4-21e3fae32908
