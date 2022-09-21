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
	using FoldsThreads
end

# ╔═╡ 69d06c40-9861-41d5-b1c3-cc7b7ccd1d48
TableOfContents()

# ╔═╡ 8e63a0f7-9c14-4817-9053-712d5d306a90
md"""
# `Wenbo`
"""

# ╔═╡ 26ee61d8-10ee-411a-9d60-0d2c0b8a6833
"""
```julia
struct Wenbo <: DistanceTransform end
```
Prepares an array to be `transform`ed
"""
struct Wenbo <: DistanceTransform end

# ╔═╡ 86168cdf-7f07-42bf-81ee-6fae7d68cebd
md"""
## In-Place
"""

# ╔═╡ fa21c417-6b0e-48a0-8993-f13c995141a6
md"""
### 1D!
"""

# ╔═╡ 167c008e-5a5f-4ba1-b1ff-2ae137b10c98
"""
```julia
transform!(f::AbstractVector, tfm::Wenbo)
```

Applies a squared euclidean distance transform to an input image using the Wenbo algorithm. Returns an array with spatial information embedded in the array elements.
"""
function transform!(f::AbstractVector, tfm::Wenbo)
	pointerA = 1
	l = length(f)
	while pointerA <= l
		while pointerA <= l && f[pointerA] == 0
			pointerA+=1
		end
		pointerB = pointerA
		while pointerB <= l && f[pointerB] == 1f10
			pointerB+=1
		end
		if pointerB > length(f)
			if pointerA == 1
				i=length(f) 
				while i>0
					f[i]=1f10
					i-=1
				end
			else
				i = pointerA
				temp=1
				l = length(f)
				while i<=l
					f[i]=temp^2
					i+=1
					temp+=1
				end
			end
		else
			if pointerA == 1
				j = pointerB-1
				temp=1
				while j>0
					f[j]=temp^2
					j-=1
					temp+=1
				end
			else
				i = pointerA
				j = pointerB-1
				temp=1
				while(i<=j)
					f[i]=f[j]=temp^2
					temp+=1
					i+=1
					j-=1
				end
			end
		end
		pointerA=pointerB
	end
	return f
end

# ╔═╡ cf740dd8-79bb-4dd8-b40c-7efcf7844256
md"""
### 2D!
"""

# ╔═╡ 16991d8b-ec84-49d0-90a9-15a78f1668bb
function _encode(leftD, rightf)
	if rightf == 1f10
		return -leftD
	end
	idx = 0
	while(rightf>1)
		rightf/=10
		idx+=1 
	end
	return -leftD-idx/10-rightf/10
end

# ╔═╡ e7dbc916-c5cb-4f86-8ea1-adbcb0bdf8ea
function _decode(curr)	
	curr *= -10   				
	temp = Int(floor(curr))		
	curr -= temp 				
	if curr == 0
		return 1f10
	end
	temp %= 10
	while temp > 0
		temp -= 1
		curr*=10
	end
	return round(curr)
end

# ╔═╡ 32a4bf03-98f8-4ed9-9c12-f45c09b0b0dd
function _transform!(f)
	l = length(f)
	pointerA = 1
	while pointerA<=l && f[pointerA] <= 1
		pointerA += 1
	end
	p = 0
	while pointerA<=l
		curr = f[pointerA]
		prev = curr
		temp = min(pointerA-1, p+1)
		p = 0
		while (0 < temp)
			fi = f[pointerA-temp]
			fi = fi < 0 ? _decode(fi) : fi
			newDistance = muladd(temp, temp, fi)
			if newDistance < curr
				curr = newDistance
				p = temp
			end
			temp -= 1
		end
		temp = 1
		templ = length(f) - pointerA
		while (temp <= templ && muladd(temp, temp, -curr) < 0)
			curr = min(curr, muladd(temp, temp, f[pointerA+temp]))
			temp += 1
		end
		f[pointerA] = _encode(curr, prev)
		# end
		pointerA+=1
		while pointerA<=l && f[pointerA] <= 1
			pointerA += 1
		end
	end
	i = 0
	while i<l
		i+=1
		f[i] = floor(abs(f[i]))
	end
end

# ╔═╡ 423df2ac-b9a2-4d59-b5fc-8de0e8cc6691
"""
```julia
transform!(f::AbstractMatrix, tfm::Wenbo)
```

Applies a squared euclidean distance transform to an input image using the Wenbo algorithm. Returns an array with spatial information embedded in the array elements.
"""
function transform!(f::AbstractMatrix, tfm::Wenbo)
	for i in axes(f, 1)
		transform!(@view(f[i, :]), tfm)
	end
	for j in axes(f, 2)
		_transform!(@view(f[:, j]))
	end
	return f
end

# ╔═╡ dd8014b7-3960-4a2e-878c-c86bbc5e7303
md"""
### 3D!
"""

# ╔═╡ b2328983-1c71-49b8-9b43-39bb3febf54b
"""
```julia
transform!(f::AbstractArray, tfm::Wenbo)
```

Applies a squared euclidean distance transform to an input image using the Wenbo algorithm. Returns an array with spatial information embedded in the array elements.
"""
function transform!(f::AbstractArray, tfm::Wenbo)
	for k in axes(f, 3)
	    transform!(@view(f[:, :, k]), tfm)
	end
	for i in CartesianIndices(f[:,:,1])
		_transform!(@view(f[i, :]))
	end
	return f
end 

# ╔═╡ 58e1cdff-59b8-44d9-a1b7-ecc14b09556c
md"""
## Multi-Threaded
"""

# ╔═╡ 0f0675ad-899d-4808-9757-deaae19a58a5
md"""
### 2D!
"""

# ╔═╡ 7fecbf6c-59b0-4465-a7c3-c5217b3980c0
"""
```julia
transform!(f::AbstractMatrix, tfm::Wenbo, nthreads::Number)
```

Applies a squared euclidean distance transform to an input image using the Wenbo algorithm. Returns an array with spatial information embedded in the array elements. Multi-threaded version of `transform!(..., tfm::Wenbo)`
"""
function transform!(f::AbstractMatrix, tfm::Wenbo, nthreads::Number)
	Threads.@threads for i in axes(f, 1)
		transform!(@view(f[i, :]), tfm)
	end
	Threads.@threads for j in axes(f, 2)
		_transform!(@view(f[:, j]))
	end
	return f
end

# ╔═╡ 37cccaee-053d-4f9c-81ef-58b274ec25b8
md"""
### 3D!
"""

# ╔═╡ f1977b4e-1834-449a-a8c9-f984a55eeca4
"""
```julia
transform!(f::AbstractArray, tfm::Wenbo, nthreads::Number)
```

Applies a squared euclidean distance transform to an input image using the Wenbo algorithm. Returns an array with spatial information embedded in the array elements. Multi-threaded version of `transform!(..., tfm::Wenbo)`
"""
function transform!(f::AbstractArray, tfm::Wenbo, nthreads::Number)
	Threads.@threads for k in axes(f, 3)
	    transform!(@view(f[:, :, k]), tfm)
	end
	Threads.@threads for i in CartesianIndices(f[:,:,1])
		_transform!(@view(f[i, :]))
	end
	return f
end 

# ╔═╡ 8da39536-8765-40fe-a158-335c905e99e6
md"""
## GPU
"""

# ╔═╡ c41c40b2-e23a-4ddd-a4ae-62b37e399f5c
md"""
### 2D!
"""

# ╔═╡ 58441e91-b837-496c-b1db-5dd428a6eba7
"""
```julia
transform!(f::CuArray{T, 2}, tfm::Wenbo) where T
```

Applies a squared euclidean distance transform to an input image using the Wenbo algorithm. Returns an array with spatial information embedded in the array elements. GPU version of `transform!(..., tfm::Wenbo)`
"""
function transform!(f::CuArray{T, 2}, tfm::Wenbo) where T
	@floop CUDAEx() for i in axes(f, 1)
		transform!(@view(f[i, :]), tfm)
	end
	@floop CUDAEx() for j in axes(f, 2)
		_transform!(@view(f[:, j]))
	end
	return f
end

# ╔═╡ 01719cd4-f69e-47f5-9d84-36229fc3e73c
md"""
### 3D!
"""

# ╔═╡ 43a06f59-9559-4a56-9eb0-47b8909841f7
"""
```julia
transform!(f::CuArray{T, 3}, tfm::Wenbo) where T
```

Applies a squared euclidean distance transform to an input image using the Wenbo algorithm. Returns an array with spatial information embedded in the array elements. GPU version of `transform!(..., tfm::Wenbo)`
"""
function transform!(f::CuArray{T, 3}, tfm::Wenbo) where T
	@floop CUDAEx() for k in axes(f, 3)
	    transform!(@view(f[:, :, k]), tfm)
	end
	@floop CUDAEx() for i in CartesianIndices(f[:,:,1])
		_transform!(@view(f[i, :]))
	end
	return f
end 

# ╔═╡ ebee3240-63cf-4323-9755-a135834208c8
md"""
## Various Multi-Threading
"""

# ╔═╡ fccb36b9-ee1b-411f-aded-147a88b23872
md"""
### 2D!
"""

# ╔═╡ 88806a34-a025-40b2-810d-b3320a137543
"""
```julia
transform!(f::AbstractMatrix, tfm::Wenbo, ex)
```

Applies a squared euclidean distance transform to an input image using the Wenbo algorithm. Returns an array with spatial information embedded in the array elements. Multi-threaded version of `transform!(..., tfm::Wenbo)` but utilizes FoldsThreads.jl for different threaded executors. `ex`=(FoldsThreads.DepthFirstEx(), FoldsThreads.NonThreadedEx(), FoldsThreads.WorkStealingEx())
"""
function transform!(f::AbstractMatrix, tfm::Wenbo, ex)
	@floop ex for i in axes(f, 1)
		transform!(@view(f[i, :]), tfm)
	end
	@floop ex for j in axes(f, 2)
		_transform!(@view(f[:, j]))
	end
	return f
end

# ╔═╡ 91e09975-e7df-4d05-9e77-dbd6c35430f0
md"""
### 3D!
"""

# ╔═╡ da6e01c4-5ef9-4628-a77f-4b43a05aad36
"""
```julia
transform!(f::AbstractArray, tfm::Wenbo, ex)
```

Applies a squared euclidean distance transform to an input image using the Wenbo algorithm. Returns an array with spatial information embedded in the array elements. Multi-threaded version of `transform!(..., tfm::Wenbo)` but utilizes FoldsThreads.jl for different threaded executors. `ex`=(FoldsThreads.DepthFirstEx(), FoldsThreads.NonThreadedEx(), FoldsThreads.WorkStealingEx())
"""
function transform!(f::AbstractArray, tfm::Wenbo, ex)
	@floop ex for k in axes(f, 3)
	    transform!(@view(f[:, :, k]), tfm)
	end
	@floop ex for i in CartesianIndices(f[:,:,1])
		_transform!(@view(f[i, :]))
	end
	return f
end 

# ╔═╡ Cell order:
# ╠═19f1c4b6-23c4-11ed-02f2-fb3e9263a1a1
# ╠═69d06c40-9861-41d5-b1c3-cc7b7ccd1d48
# ╟─8e63a0f7-9c14-4817-9053-712d5d306a90
# ╠═26ee61d8-10ee-411a-9d60-0d2c0b8a6833
# ╟─86168cdf-7f07-42bf-81ee-6fae7d68cebd
# ╟─fa21c417-6b0e-48a0-8993-f13c995141a6
# ╠═167c008e-5a5f-4ba1-b1ff-2ae137b10c98
# ╟─cf740dd8-79bb-4dd8-b40c-7efcf7844256
# ╠═16991d8b-ec84-49d0-90a9-15a78f1668bb
# ╠═e7dbc916-c5cb-4f86-8ea1-adbcb0bdf8ea
# ╠═32a4bf03-98f8-4ed9-9c12-f45c09b0b0dd
# ╠═423df2ac-b9a2-4d59-b5fc-8de0e8cc6691
# ╟─dd8014b7-3960-4a2e-878c-c86bbc5e7303
# ╠═b2328983-1c71-49b8-9b43-39bb3febf54b
# ╟─58e1cdff-59b8-44d9-a1b7-ecc14b09556c
# ╟─0f0675ad-899d-4808-9757-deaae19a58a5
# ╠═7fecbf6c-59b0-4465-a7c3-c5217b3980c0
# ╟─37cccaee-053d-4f9c-81ef-58b274ec25b8
# ╠═f1977b4e-1834-449a-a8c9-f984a55eeca4
# ╟─8da39536-8765-40fe-a158-335c905e99e6
# ╟─c41c40b2-e23a-4ddd-a4ae-62b37e399f5c
# ╠═58441e91-b837-496c-b1db-5dd428a6eba7
# ╟─01719cd4-f69e-47f5-9d84-36229fc3e73c
# ╠═43a06f59-9559-4a56-9eb0-47b8909841f7
# ╟─ebee3240-63cf-4323-9755-a135834208c8
# ╟─fccb36b9-ee1b-411f-aded-147a88b23872
# ╠═88806a34-a025-40b2-810d-b3320a137543
# ╟─91e09975-e7df-4d05-9e77-dbd6c35430f0
# ╠═da6e01c4-5ef9-4628-a77f-4b43a05aad36
