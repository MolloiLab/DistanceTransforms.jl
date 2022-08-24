### A Pluto.jl notebook ###
# v0.19.11

using Markdown
using InteractiveUtils

# ╔═╡ 19f1c4b6-23c4-11ed-02f2-fb3e9263a1a1
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

# ╔═╡ 69d06c40-9861-41d5-b1c3-cc7b7ccd1d48
TableOfContents()

# ╔═╡ 8e63a0f7-9c14-4817-9053-712d5d306a90
md"""
# `Wenbo`

```julia
struct Wenbo <: DistanceTransforms.DistanceTransform end
```

Squared euclidean algorithm utilizing dynamic programming approaches to improve upon the speed of [Felzenszwalb and Huttenlocher] (DOI: 10.4086/toc.2012.v008a019)
"""

# ╔═╡ 26ee61d8-10ee-411a-9d60-0d2c0b8a6833
struct Wenbo <: DistanceTransforms.DistanceTransform end

# ╔═╡ 86168cdf-7f07-42bf-81ee-6fae7d68cebd
md"""
## Regular
"""

# ╔═╡ fa21c417-6b0e-48a0-8993-f13c995141a6
md"""
### 1D
```julia
transform(f, tfm::Wenbo; output=zeros(length(f)), pointerA=1, pointerB=1)
```

Assume length(f)>0. This is a one pass algorithm. Time complexity=O(n). Space complexity=O(1)
"""

# ╔═╡ e2a9abc7-1102-43ae-9b58-4af81795ee75
md"""
```julia
function _DT1(input, output, i, j)
```

Helper function for 1-D Wenbo distance transform `transform(f::AbstractVector, tfm::Wenbo)`
"""

# ╔═╡ 414a288e-2a79-449d-965e-63102f12bd14
function _DT1(output, i, j)
	if (i==-1 && j==-1)
		i=1
		while(i<=length(output))
			output[i]=1f10
			i=i+1
		end
	elseif(i==-1)
		temp=1
		while(j>0)
			output[j]=temp^2
			j=j-1
			temp=temp+1
		end
	elseif(j==-1)
		temp=1
		while(i<=length(output))
			output[i]=temp^2
			i=i+1
			temp=temp+1
		end
	else
		temp=1
		while(i<=j)
			output[i]=output[j]=temp^2
			temp=temp+1
			i=i+1
			j=j-1
		end
	end
	return output
end

# ╔═╡ 167c008e-5a5f-4ba1-b1ff-2ae137b10c98
function transform(f::AbstractVector, tfm::Wenbo; output=zeros(length(f)), pointerA=1, pointerB=1)
	while (pointerA<=length(f))
		if(f[pointerA] == 0)
			output[pointerA]=0
			pointerA=pointerA+1
			pointerB=pointerB+1
		else
			while(pointerB <= length(f) && f[pointerB]==1f10)
				pointerB=pointerB+1
			end
			if (pointerB > length(f))
				if (pointerA == 1)
					output = _DT1(output, -1, -1)
				else
					output = _DT1(output, pointerA, -1)
				end
			else
				if (pointerA == 1)
					output = _DT1(output, -1, pointerB-1)
				else
					output = _DT1(output, pointerA, pointerB-1)
				end
			end
			pointerA=pointerB
		end
	end
	return output
end

# ╔═╡ cf740dd8-79bb-4dd8-b40c-7efcf7844256
md"""
### 2D
```julia
transform(img::AbstractMatrix, tfm::Wenbo; output=zeros(size(img)), pointerA=1, pointerB=1)
```

2-D Wenbo Distance Transform.
"""

# ╔═╡ 5d43c997-bb5f-45a7-a662-a658b0efe286
md"""
```julia
_DT2(f; output=zeros(length(f)), pointerA=1)
```

Helper function for 2-D Wenbo distance transform `transform(f::AbstractVector, tfm::Wenbo)`
Computes the vertical operation.
"""

# ╔═╡ a91599c1-3b94-4d7e-9c65-7c11df0d3aa2
function _DT2(f; output=zeros(length(f)), pointerA=1)
	while (pointerA<=length(f))
		output[pointerA]=f[pointerA]
		if(f[pointerA] > 1)
			if (length(f) - pointerA <= pointerA - 1)
				temp = 1
				while (output[pointerA]>1 && temp <= length(f) - pointerA)
					if (f[pointerA+temp]<output[pointerA])
						output[pointerA]=min(output[pointerA], f[pointerA+temp]+temp^2)
					end
					if (f[pointerA-temp]<output[pointerA])
						output[pointerA]=min(output[pointerA], f[pointerA-temp]+temp^2)
					end
					temp = temp + 1
				end
				if(f[pointerA] > 1)
					while (output[pointerA]>1 && temp <= pointerA - 1)
						if (f[pointerA-temp]<output[pointerA])
							output[pointerA]=min(output[pointerA], f[pointerA-temp]+temp^2)
						end
						temp = temp + 1
					end
				end
			else
				temp = 1
				while (output[pointerA]>1 && temp <= pointerA - 1)
					if (f[pointerA+temp]<output[pointerA])
						output[pointerA]=min(output[pointerA], f[pointerA+temp]+temp^2)
					end
					if (f[pointerA-temp]<output[pointerA])
						output[pointerA]=min(output[pointerA], f[pointerA-temp]+temp^2)
					end
					temp = temp + 1
				end
				if(f[pointerA] > 1)
					while (output[pointerA]>1 && temp <= length(f) - pointerA)
						if (f[pointerA+temp]<output[pointerA])
							output[pointerA]=min(output[pointerA], f[pointerA+temp]+temp^2)
						end
						temp = temp + 1
					end
				end
			end
		end
		pointerA=pointerA+1
	end
	return output
end

# ╔═╡ c45c571d-e0e0-400c-86f0-af7da87cbf6b
function transform(img::AbstractMatrix, tfm::Wenbo; output=zeros(size(img)), pointerA=1, pointerB=1)
	# This is a worst case = O(n^3) implementation
	for i in axes(img, 1)
	    output[i, :] = transform(img[i, :], tfm; output=output[i, :], pointerA=pointerA, pointerB=pointerB) 
	end

	for j in axes(img, 2)
	    output[:, j] = _DT2(output[:, j]; output=output[:, j], pointerA=pointerA) 
	end
	return output
end

# ╔═╡ 881da22a-5a03-4278-9f38-cd9aa28121ee
md"""
### 3D
```julia
transform(f::AbstractArray, tfm::Wenbo; D=zeros(size(f)), pointerA=1, pointerB=1)
```

3-D Wenbo Distance Transform.
"""

# ╔═╡ cb064c9b-ea0e-437f-9a99-8a5a0043a462
function transform(f::AbstractArray, tfm::Wenbo; output=zeros(size(f)), pointerA=1, pointerB=1)
	for i in axes(f, 3)
	    output[:, :, i] = transform(f[:, :, i], Wenbo(); output=output[:, :, i], pointerA=pointerA, pointerB=pointerB)
	end
	for i in axes(f, 1)
		for j in axes(f, 2)
	    	output[i, j, :] = _DT2(output[i, j, :]; output=output[i, j, :], pointerA=pointerA)
		end
	end
	return output
end

# ╔═╡ Cell order:
# ╠═19f1c4b6-23c4-11ed-02f2-fb3e9263a1a1
# ╠═69d06c40-9861-41d5-b1c3-cc7b7ccd1d48
# ╟─8e63a0f7-9c14-4817-9053-712d5d306a90
# ╠═26ee61d8-10ee-411a-9d60-0d2c0b8a6833
# ╟─86168cdf-7f07-42bf-81ee-6fae7d68cebd
# ╟─fa21c417-6b0e-48a0-8993-f13c995141a6
# ╠═167c008e-5a5f-4ba1-b1ff-2ae137b10c98
# ╟─e2a9abc7-1102-43ae-9b58-4af81795ee75
# ╠═414a288e-2a79-449d-965e-63102f12bd14
# ╟─cf740dd8-79bb-4dd8-b40c-7efcf7844256
# ╠═c45c571d-e0e0-400c-86f0-af7da87cbf6b
# ╟─5d43c997-bb5f-45a7-a662-a658b0efe286
# ╠═a91599c1-3b94-4d7e-9c65-7c11df0d3aa2
# ╟─881da22a-5a03-4278-9f38-cd9aa28121ee
# ╠═cb064c9b-ea0e-437f-9a99-8a5a0043a462
