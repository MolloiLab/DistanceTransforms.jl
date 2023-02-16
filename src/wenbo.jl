### A Pluto.jl notebook ###
# v0.19.8

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
## CPU
"""

# ╔═╡ fa21c417-6b0e-48a0-8993-f13c995141a6
md"""
### 1D
"""

# ╔═╡ 9db7eb7e-e47d-4e8d-81c0-f597eae51c04
function _transform1!(f::AbstractVector)
		pointerA = 1
		l = length(f)
		while pointerA <= l
			while pointerA <= l && @inbounds f[pointerA] == 0f0
				pointerA+=1
			end
			pointerB = pointerA
			while pointerB <= l && @inbounds f[pointerB] == 1f10
				pointerB+=1
			end
			if pointerB > length(f)
				if pointerA == 1
					return
				else
					i = pointerA
					temp=i-1
					l = length(f)
					while i<=l
						@inbounds f[i]=(i-temp)^2
						i+=1
					end
				end
			else
				if pointerA == 1
					j = pointerB-1
					temp=j+1
					while j>0
						@inbounds f[j]=(temp-j)^2
						j-=1
					end
				else
					i, j = pointerA, pointerB-1
					temp=1
					while(i<=j)
						@inbounds f[i]=f[j]=temp^2
						temp+=1
						i+=1
						j-=1
					end
				end
			end
			pointerA=pointerB
		end
	end

# ╔═╡ 167c008e-5a5f-4ba1-b1ff-2ae137b10c98
"""
```julia
transform(f::AbstractVector, tfm::Wenbo)
```

Applies a squared euclidean distance transform to an input 1D image using the Wenbo algorithm. Returns an array with spatial information embedded in the array elements.
"""
function transform(f::AbstractVector, tfm::Wenbo)
	f = boolean_indicator(f)
	_transform1!(f)
	return f
end

# ╔═╡ cf740dd8-79bb-4dd8-b40c-7efcf7844256
md"""
### 2D
"""

# ╔═╡ 16991d8b-ec84-49d0-90a9-15a78f1668bb
function _encode(leftD::Float32, rightf::Float32)
	if rightf == 1f10
		return -leftD
	end
	idx = 0f0
	while rightf >= 1f0
		rightf /= 10f0
		idx += 1f0
	end
	return -leftD-idx/10f0-rightf/10f0
end

# ╔═╡ e7dbc916-c5cb-4f86-8ea1-adbcb0bdf8ea
function _decode(curr::Float32)		
	curr *= -1f0    
	curr -= floor(curr) 		
	curr *= 10f0    
	temp = floor(curr % 10f0)
	if temp == 0f0
		return 1f10
	end
	curr -= temp			
	while temp > 0f0
		temp -= 1f0
		curr *= 10f0
	end
	return round(curr)
end

# ╔═╡ 32a4bf03-98f8-4ed9-9c12-f45c09b0b0dd
function _transform2!(f::AbstractVector, org::AbstractVector)
	l = length(f)
	pointerA = 1
	while pointerA<=l && @inbounds f[pointerA] <= 1f0
		pointerA += 1
	end
	p = 0
	while pointerA<=l
		@inbounds curr = f[pointerA]
		# left
		temp = min(pointerA-1, p+1)
		p = 0
		while 0 < temp
			@inbounds newDistance = muladd(temp, temp, org[pointerA-temp])
			if newDistance < curr
				curr = newDistance
				p = temp
			end
			temp -= 1
		end
		# right
		temp = 1
		templ = length(f) - pointerA
		while temp <= templ && muladd(temp, temp, -curr) < 0
			@inbounds curr = min(curr, muladd(temp, temp, f[pointerA+temp]))
			temp += 1
		end
		@inbounds f[pointerA] = curr
		pointerA+=1
		while pointerA<=l && @inbounds f[pointerA] <= 1f0
			pointerA += 1
		end
	end
end

# ╔═╡ 89fed2a6-b09e-47b1-a020-efed76ba57de
function _transform2_EN_DE!(f::AbstractVector)
	l = length(f)
	pointerA = 1
	while pointerA<=l && @inbounds f[pointerA] <= 1f0
		pointerA += 1
	end
	p = 0
	while pointerA<=l
		@inbounds curr = f[pointerA]
		# left
		prev = curr
		temp = min(pointerA-1, p+1)
		p = 0
		while 0 < temp
			@inbounds fi = f[pointerA-temp]
			fi = fi < 0 ? _decode(fi) : fi
			newDistance = muladd(temp, temp, fi)
			if newDistance < curr
				curr = newDistance
				p = temp
			end
			temp -= 1
		end
		# right
		temp = 1
		templ = length(f) - pointerA
		while temp <= templ && muladd(temp, temp, -curr) < 0
			@inbounds curr = min(curr, muladd(temp, temp, f[pointerA+temp]))
			temp += 1
		end
		@inbounds f[pointerA] = _encode(curr, prev)
		pointerA+=1
		while pointerA<=l && @inbounds f[pointerA] <= 1f0
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
transform(f::AbstractMatrix, tfm::Wenbo)
```

Applies a squared euclidean distance transform to an input 2D image using the Wenbo algorithm. Returns an array with spatial information embedded in the array elements.
"""
function transform(f::AbstractMatrix, tfm::Wenbo)
	f = boolean_indicator(f)
	for i in axes(f, 1)
		@inbounds _transform1!(@view(f[i, :]))
	end
	org = copy(f)
	for j in axes(f, 2)
		@inbounds _transform2!(@view(f[:,j]), @view(org[:,j]))
	end
	return f
end

# ╔═╡ dd8014b7-3960-4a2e-878c-c86bbc5e7303
md"""
### 3D
"""

# ╔═╡ b2328983-1c71-49b8-9b43-39bb3febf54b
"""
```julia
transform(f::AbstractArray, tfm::Wenbo)
```

Applies a squared euclidean distance transform to an input 3D image using the Wenbo algorithm. Returns an array with spatial information embedded in the array elements.
"""
function transform(f::AbstractArray, tfm::Wenbo)
	f = boolean_indicator(f)
	for i in CartesianIndices(f[1,:,:])
		@inbounds _transform1!(@view(f[:, i]))
	end
	for i in CartesianIndices(f[:,1,:])
		@inbounds _transform2_EN_DE!(@view(f[i[1], :, i[2]]))
	end
	for i in CartesianIndices(f[:,:,1])
		@inbounds _transform2_EN_DE!(@view(f[i, :]))
	end
	return f
end 

# ╔═╡ 58e1cdff-59b8-44d9-a1b7-ecc14b09556c
md"""
## Multi-Threaded
"""

# ╔═╡ d663cf13-4a3a-4667-8971-ddb5c455d85c
# function _transform4!(f)
# 	Threads.@threads for i in axes(f, 1)
# 		@inbounds _transform1!(@view(f[i, :]))
# 	end
# 	org = copy(f)
# 	Threads.@threads for j in axes(f, 2)
# 		@inbounds _transform2!(@view(f[:,j]), @view(org[:,j]))
# 	end
# end

# ╔═╡ 0f0675ad-899d-4808-9757-deaae19a58a5
md"""
### 2D
"""

# ╔═╡ 7fecbf6c-59b0-4465-a7c3-c5217b3980c0
"""
```julia
transform(f::AbstractMatrix, tfm::Wenbo, nthreads::Number)
```

Applies a squared euclidean distance transform to an input 2D image using the Wenbo algorithm. Returns an array with spatial information embedded in the array elements. Multi-threaded version of `transform(..., tfm::Wenbo)`
"""
function transform(f::AbstractMatrix, tfm::Wenbo, nthreads::Number)
	f = boolean_indicator(f)
	Threads.@threads for i in axes(f, 1)
		@inbounds _transform1!(@view(f[i, :]))
	end
	Threads.@threads for j in axes(f, 2)
		@inbounds _transform2_EN_DE!(@view(f[:,j]))
	end
	return f
end

# ╔═╡ 37cccaee-053d-4f9c-81ef-58b274ec25b8
md"""
### 3D
"""

# ╔═╡ f1977b4e-1834-449a-a8c9-f984a55eeca4
"""
```julia
transform(f::AbstractArray, tfm::Wenbo, nthreads::Number)
```

Applies a squared euclidean distance transform to an input 3D image using the Wenbo algorithm. Returns an array with spatial information embedded in the array elements. Multi-threaded version of `transform(..., tfm::Wenbo)`
"""
function transform(f::AbstractArray, tfm::Wenbo, nthreads::Number)
	f = boolean_indicator(f)
	Threads.@threads for i in CartesianIndices(f[1,:,:])
		@inbounds _transform1!(@view(f[:, i]))
	end
	Threads.@threads for i in CartesianIndices(f[:,1,:])
		@inbounds _transform2_EN_DE!(@view(f[i[1], :, i[2]]))
	end
	Threads.@threads for i in CartesianIndices(f[:,:,1])
		@inbounds _transform2_EN_DE!(@view(f[i, :]))
	end
	return f
end 

# ╔═╡ 948a0099-bc78-4707-9fa1-ad5dc59c34a5
md"""
### CPU-Batch
"""

# ╔═╡ 84555ba9-ac32-4409-81e4-1e21d02aa1a1
"""
```julia
transform(batch_size::Number, f::AbstractArray, tfm::Wenbo, _)
```

Applies squared euclidean distance transforms to a number of batch_size N-dimension images using the Wenbo algorithm. Returns an array with spatial information embedded in the array elements. The length of last dimension of input should be equal to the batch size. 
"""
function transform(is_batch::Bool, f::AbstractArray, tfm::Wenbo, _)
	n_dims = ndims(f)
	num_channels, num_batchs = size(f)[n_dims-1], size(f)[n_dims]
	f_new = similar(f, Float32)
	for batch_idx = 1: num_batchs
		for channel_idx = 1:num_channels
			@inbounds selectdim(selectdim(f_new, n_dims, batch_idx), n_dims-1, channel_idx)[:] = transform(selectdim(selectdim(f, n_dims, batch_idx), n_dims-1, channel_idx), tfm, 16)
		end
	end
	return f_new
end 

# ╔═╡ 8da39536-8765-40fe-a158-335c905e99e6
md"""
## GPU
"""

# ╔═╡ c41c40b2-e23a-4ddd-a4ae-62b37e399f5c
md"""
### 2D
"""

# ╔═╡ ad52080b-7d59-459d-829d-2a77ddf12c5f
function _kernel_2D_1_1!(out, f, row_l, l)
	i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
	if i > l
		return
	end 
	row = cld(i, row_l)
	col = i%row_l+1
	@inbounds if f[row, col] >= 0.5f0
		return 
	end
	ct = 1
	curr_l = min(col-1, row_l-col)
	while ct <= curr_l
		@inbounds if f[row, col-ct] >= 0.5f0 || f[row, col+ct] >= 0.5f0
			@inbounds out[row, col] = ct*ct
			return 
		end
		ct += 1
	end
	while ct < col
		@inbounds if f[row, col-ct] >= 0.5f0
			@inbounds out[row, col] = ct*ct
			return 
		end
		ct += 1    
	end
	while col+ct <= row_l
		@inbounds if f[row, col+ct] >= 0.5f0
			@inbounds out[row, col] = ct*ct
			return 
		end
		ct += 1
	end
	@inbounds out[row, col] = 1f10
	return 
end

# ╔═╡ b5963be3-7794-4ae0-9330-6177a82605ef
# function _kernel_2D_1_2!(out, f, row_l, l)
# 	i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
# 	if i > l
# 		return
# 	end 
# 	row = cld(i, row_l)
# 	col = i%row_l+1
# 	@inbounds if f[row, col] 
# 		return 
# 	end
# 	ct = 1
# 	curr_l = min(col-1, row_l-col)
# 	while ct <= curr_l
# 		@inbounds if f[row, col-ct] || f[row, col+ct]
# 			@inbounds out[row, col] = ct*ct
# 			return 
# 		end
# 		ct += 1
# 	end
# 	while ct < col
# 		@inbounds if f[row, col-ct]
# 			@inbounds out[row, col] = ct*ct
# 			return 
# 		end
# 		ct += 1    
# 	end
# 	while col+ct <= row_l
# 		@inbounds if f[row, col+ct]
# 			@inbounds out[row, col] = ct*ct
# 			return 
# 		end
# 		ct += 1
# 	end
# 	@inbounds out[row, col] = 1f10
# 	return 
# end

# ╔═╡ 927f25f9-1687-415f-b5bb-8a8f40afdd0f
 function _kernel_2D_2!(org, out, row_l, col_l, l)
	i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if i > l
        return
    end 
    row = cld(i, row_l)
    col = i%row_l+1
    ct = 1
    @inbounds curr_l = CUDA.sqrt(out[row, col])
    @inbounds while ct < curr_l && row+ct <= col_l
        @inbounds temp = muladd(ct,ct,org[row+ct, col])
        @inbounds if temp < out[row, col]
            @inbounds out[row, col] = temp
            curr_l = CUDA.sqrt(temp)
        end
        ct += 1
    end
    ct = 1
    @inbounds while ct < curr_l && row > ct
        @inbounds temp = muladd(ct,ct,org[row-ct, col])
        @inbounds if temp < out[row, col]
            @inbounds out[row, col] = temp
            curr_l = CUDA.sqrt(temp)
        end
        ct += 1
    end
    return
end

# ╔═╡ 2a1c7734-805e-4971-9e07-a39c840457f0
function _kernel_2D_1_1_batch!(out, f, row_l, l, channel_idx, batch_idx)
	i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
	if i > l
		return
	end 
	row = cld(i, row_l)
	col = i%row_l+1
	@inbounds if f[row, col, channel_idx, batch_idx] >= 0.5f0
		return 
	end
	ct = 1
	curr_l = min(col-1, row_l-col)
	while ct <= curr_l
		@inbounds if f[row, col-ct, channel_idx, batch_idx] >= 0.5f0 || f[row, col+ct, channel_idx, batch_idx] >= 0.5f0
			@inbounds out[row, col, channel_idx, batch_idx] = ct*ct
			return 
		end
		ct += 1
	end
	while ct < col
		@inbounds if f[row, col-ct, channel_idx, batch_idx] >= 0.5f0
			@inbounds out[row, col, channel_idx, batch_idx] = ct*ct
			return 
		end
		ct += 1    
	end
	while col+ct <= row_l
		@inbounds if f[row, col+ct, channel_idx, batch_idx] >= 0.5f0
			@inbounds out[row, col, channel_idx, batch_idx] = ct*ct
			return 
		end
		ct += 1
	end
	@inbounds out[row, col, channel_idx, batch_idx] = 1f10
	return 
end

# ╔═╡ 716e061a-dec8-4515-b2ce-f893ca23f99e
# function _kernel_2D_1_2_batch!(out, f, row_l, l, channel_idx, batch_idx)
# 	i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
# 	if i > l
# 		return
# 	end 
# 	row = cld(i, row_l)
# 	col = i%row_l+1
# 	@inbounds if f[row, col, channel_idx, batch_idx]
# 		return 
# 	end
# 	ct = 1
# 	curr_l = min(col-1, row_l-col)
# 	while ct <= curr_l
# 		@inbounds if f[row, col-ct, channel_idx, batch_idx] || f[row, col+ct, channel_idx, batch_idx]
# 			@inbounds out[row, col, channel_idx, batch_idx] = ct*ct
# 			return 
# 		end
# 		ct += 1
# 	end
# 	while ct < col
# 		@inbounds if f[row, col-ct, channel_idx, batch_idx]
# 			@inbounds out[row, col, channel_idx, batch_idx] = ct*ct
# 			return 
# 		end
# 		ct += 1    
# 	end
# 	while col+ct <= row_l
# 		@inbounds if f[row, col+ct, channel_idx, batch_idx]
# 			@inbounds out[row, col, channel_idx, batch_idx] = ct*ct
# 			return 
# 		end
# 		ct += 1
# 	end
# 	@inbounds out[row, col, channel_idx, batch_idx] = 1f10
# 	return 
# end

# ╔═╡ aaf5a46a-67c7-457b-bc83-d1c2163583b8
 function _kernel_2D_2_batch!(org, out, row_l, col_l, l, channel_idx, batch_idx)
	i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if i > l
        return
    end 
    row = cld(i, row_l)
    col = i%row_l+1
    ct = 1
    @inbounds curr_l = CUDA.sqrt(out[row, col, channel_idx, batch_idx])
    @inbounds while ct < curr_l && row+ct <= col_l
        @inbounds temp = muladd(ct,ct,org[row+ct, col, channel_idx, batch_idx])
        @inbounds if temp < out[row, col, channel_idx, batch_idx]
            @inbounds out[row, col, channel_idx, batch_idx] = temp
            curr_l = CUDA.sqrt(temp)
        end
        ct += 1
    end
    ct = 1
    @inbounds while ct < curr_l && row > ct
        @inbounds temp = muladd(ct,ct,org[row-ct, col, channel_idx, batch_idx])
        @inbounds if temp < out[row, col, channel_idx, batch_idx]
            @inbounds out[row, col, channel_idx, batch_idx] = temp
            curr_l = CUDA.sqrt(temp)
        end
        ct += 1
    end
    return
end

# ╔═╡ 25b46272-9f45-45f1-bf81-128a4bcf041f
function _transform_batch(f::CuArray{T, 4}, tfm::Wenbo, kernels) where T  
	col_length, row_length, num_channels, batch_size = size(f)
	# println("size = $col_length, $row_length, $batch_size")
	l = col_length * row_length
	f_new = CUDA.zeros(col_length,row_length,num_channels,batch_size)
	@inbounds threads = min(l, kernels[6])
	blocks = cld(l, threads)
	# @inbounds k1 = T<:Bool ? kernels[10] : kernels[9]
	for batch_idx = 1:batch_size
		for channel_idx = 1:num_channels
			@inbounds kernels[7](f_new, f, row_length, l, channel_idx, batch_idx; threads, blocks)
			@inbounds kernels[8](copy(f_new), f_new, row_length, col_length, l, channel_idx, batch_idx; threads, blocks)
		end
	end
	return f_new
end

# ╔═╡ 58441e91-b837-496c-b1db-5dd428a6eba7
"""
```julia
transform(f::CuArray{T, 2}, tfm::Wenbo, kernels) where T 
transform(batch_size, f::CuArray{T, 2}, tfm::Wenbo, kernels, ) where T 
```

Applies a squared euclidean distance transform to an input 2D boolean image using the Wenbo algorithm. Returns an array with spatial information embedded in the array elements. GPU version of `transform(..., tfm::Wenbo)`
"""
function transform(f::CuArray{T, 2}, tfm::Wenbo, kernels) where T  
	col_length, row_length = size(f)
	l = length(f)
	f_new = CUDA.zeros(col_length,row_length)
	threads = min(l, kernels[6])
	blocks = cld(l, threads)
	# k1 = T<:Bool ? kernels[2] : kernels[1]
	@inbounds kernels[1](f_new, f, row_length, l; threads, blocks)
	@inbounds kernels[2](copy(f_new), f_new, row_length, col_length, l; threads, blocks)
	return f_new
end

# ╔═╡ 01719cd4-f69e-47f5-9d84-36229fc3e73c
md"""
### 3D
"""

# ╔═╡ 24527d9b-1fa2-443d-ad4c-76a53b1ba4c2
function _kernel_3D_1_1!(out, f, dim2_l, dim3_l, l)
	i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if i > l
        return
    end 
    temp =  dim2_l*dim3_l
    dim1 = CUDA.cld(i, temp)
    temp2 = (i-1)%temp+1
    dim2 = CUDA.cld(temp2, dim3_l)
    dim3 = temp2 % dim3_l + 1
    # 1d DT along dim2
    @inbounds if f[dim1, dim2, dim3] < 0.5f0
        ct = 1
        curr_l = CUDA.min(dim2-1, dim2_l-dim2)
        while ct <= curr_l
            @inbounds if f[dim1, dim2-ct, dim3]>=0.5f0 || f[dim1, dim2+ct, dim3]>=0.5f0
                @inbounds out[dim1, dim2, dim3] = ct*ct
                return 
            end
            ct += 1
        end
        while ct < dim2
            @inbounds if f[dim1, dim2-ct, dim3]>=0.5f0
                @inbounds out[dim1, dim2, dim3] = ct*ct
                return 
            end
            ct += 1    
        end
        while dim2+ct <= dim2_l
            @inbounds if f[dim1, dim2+ct, dim3]>=0.5f0
                @inbounds out[dim1, dim2, dim3] = ct*ct
                return 
            end
            ct += 1
        end
        @inbounds out[dim1, dim2, dim3] = 1f10
    end
    return 
end

# ╔═╡ 36bee155-1c27-40be-a956-30ef54ab14ef
# function _kernel_3D_1_2!(out, f, dim2_l, dim3_l, l)
# 	i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
#     if i > l
#         return
#     end 
#     temp =  dim2_l*dim3_l
#     dim1 = CUDA.cld(i, temp)
#     temp2 = (i-1)%temp+1
#     dim2 = CUDA.cld(temp2, dim3_l)
#     dim3 = temp2 % dim3_l + 1
#     # 1d DT alone dim2
#     @inbounds if !f[dim1, dim2, dim3]
#         ct = 1
#         curr_l = CUDA.min(dim2-1, dim2_l-dim2)
#         while ct <= curr_l
#             @inbounds if f[dim1, dim2-ct, dim3] || f[dim1, dim2+ct, dim3]
#                 @inbounds out[dim1, dim2, dim3] = ct*ct
#                 return 
#             end
#             ct += 1
#         end
#         while ct < dim2
#             @inbounds if f[dim1, dim2-ct, dim3]
#                 @inbounds out[dim1, dim2, dim3] = ct*ct
#                 return 
#             end
#             ct += 1    
#         end
#         while dim2+ct <= dim2_l
#             @inbounds if f[dim1, dim2+ct, dim3]
#                 @inbounds out[dim1, dim2, dim3] = ct*ct
#                 return 
#             end
#             ct += 1
#         end
#         @inbounds out[dim1, dim2, dim3] = 1f10
#     end
#     return 
# end

# ╔═╡ 22c9dd53-6ae6-45f9-8a44-c3777aefef5c
function _kernel_3D_2!(out, org, dim1_l, dim2_l, dim3_l, l)
	i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if i > l
        return
    end 
    temp =  dim2_l*dim3_l
    dim1 = cld(i, temp)
    temp2 = (i-1)%temp+1
    dim2 = cld(temp2, dim3_l)
    dim3 = temp2 % dim3_l + 1
    # 2d DT along dim1
    ct = 1
    @inbounds curr_l = CUDA.sqrt(out[dim1, dim2, dim3])
    @inbounds while ct < curr_l && dim1+ct <= dim1_l
        @inbounds if org[dim1+ct, dim2, dim3] < out[dim1, dim2, dim3]
            @inbounds out[dim1, dim2, dim3] = min(out[dim1, dim2, dim3], muladd(ct,ct,org[dim1+ct, dim2, dim3]))
            @inbounds curr_l = CUDA.sqrt(out[dim1, dim2, dim3])
        end
        ct += 1
    end
    ct = 1
    @inbounds while ct < curr_l && dim1-ct > 0
        @inbounds if org[dim1-ct, dim2, dim3] < out[dim1, dim2, dim3]
            @inbounds out[dim1, dim2, dim3] = min(out[dim1, dim2, dim3], muladd(ct,ct,org[dim1-ct, dim2, dim3]))
            @inbounds curr_l = CUDA.sqrt(out[dim1, dim2, dim3])
        end
        ct += 1
    end
    return
end

# ╔═╡ 678c8a54-0b50-4419-8984-e0f0507e9b48
function _kernel_3D_3!(out, org, dim2_l, dim3_l, l)
	i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if i > l
        return
    end 
    temp =  dim2_l*dim3_l
    dim1 = cld(i, temp)
    temp2 = (i-1)%temp+1
    dim2 = cld(temp2, dim3_l)
    dim3 = temp2 % dim3_l + 1
    # 2d DT along dim3=
    ct = 1
    @inbounds curr_l = CUDA.sqrt(out[dim1, dim2, dim3])
    @inbounds while ct < curr_l && dim3+ct <= dim3_l
        @inbounds if org[dim1, dim2, dim3+ct] < out[dim1, dim2, dim3]
            @inbounds out[dim1, dim2, dim3] = min(out[dim1, dim2, dim3], muladd(ct,ct,org[dim1, dim2, dim3+ct]))
            @inbounds curr_l = CUDA.sqrt(out[dim1, dim2, dim3])
        end
        ct += 1
    end
    ct = 1
    @inbounds while ct < curr_l && ct < dim3
        @inbounds if org[dim1, dim2, dim3-ct] < out[dim1, dim2, dim3]
            @inbounds out[dim1, dim2, dim3] = min(out[dim1, dim2, dim3], muladd(ct,ct,org[dim1, dim2, dim3-ct]))
            @inbounds curr_l = CUDA.sqrt(out[dim1, dim2, dim3])
        end
        ct += 1
    end
    return
end 

# ╔═╡ a0dc7da8-54dc-43d6-a501-49ae35951563
function _kernel_3D_1_1_batch!(out, f, dim2_l, dim3_l, l, channel_idx, batch_idx)
	i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if i > l
        return
    end 
    temp =  dim2_l*dim3_l
    dim1 = CUDA.cld(i, temp)
    temp2 = (i-1)%temp+1
    dim2 = CUDA.cld(temp2, dim3_l)
    dim3 = temp2 % dim3_l + 1
    # 1d DT alone dim2
    @inbounds if f[dim1, dim2, dim3, channel_idx, batch_idx] < 0.5f0
        ct = 1
        curr_l = CUDA.min(dim2-1, dim2_l-dim2)
        while ct <= curr_l
            @inbounds if f[dim1, dim2-ct, dim3, channel_idx, batch_idx] >= 0.5f0 || f[dim1, dim2+ct, dim3, channel_idx, batch_idx] >= 0.5f0
                @inbounds out[dim1, dim2, dim3, channel_idx, batch_idx] = ct*ct
                return 
            end
            ct += 1
        end
        while ct < dim2
            @inbounds if f[dim1, dim2-ct, dim3, channel_idx, batch_idx] >= 0.5f0
                @inbounds out[dim1, dim2, dim3, channel_idx, batch_idx] = ct*ct
                return 
            end
            ct += 1    
        end
        while dim2+ct <= dim2_l
            @inbounds if f[dim1, dim2+ct, dim3, channel_idx, batch_idx] >= 0.5f0
                @inbounds out[dim1, dim2, dim3, channel_idx, batch_idx] = ct*ct
                return 
            end
            ct += 1
        end
        @inbounds out[dim1, dim2, dim3, channel_idx, batch_idx] = 1f10
    end
    return 
end

# ╔═╡ e4872d2f-1931-4019-993f-14c318362be6
# function _kernel_3D_1_2_batch!(out, f, dim2_l, dim3_l, l, channel_idx, batch_idx)
# 	i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
#     if i > l
#         return
#     end 
#     temp =  dim2_l*dim3_l
#     dim1 = CUDA.cld(i, temp)
#     temp2 = (i-1)%temp+1
#     dim2 = CUDA.cld(temp2, dim3_l)
#     dim3 = temp2 % dim3_l + 1
#     # 1d DT alone dim2
#     @inbounds if !f[dim1, dim2, dim3, channel_idx, batch_idx]
#         ct = 1
#         curr_l = CUDA.min(dim2-1, dim2_l-dim2)
#         while ct <= curr_l
#             @inbounds if f[dim1, dim2-ct, dim3, channel_idx, batch_idx] || f[dim1, dim2+ct, dim3, channel_idx, batch_idx]
#                 @inbounds out[dim1, dim2, dim3, channel_idx, batch_idx] = ct*ct
#                 return 
#             end
#             ct += 1
#         end
#         while ct < dim2
#             @inbounds if f[dim1, dim2-ct, dim3, channel_idx, batch_idx]
#                 @inbounds out[dim1, dim2, dim3, channel_idx, batch_idx] = ct*ct
#                 return 
#             end
#             ct += 1    
#         end
#         while dim2+ct <= dim2_l
#             @inbounds if f[dim1, dim2+ct, dim3, channel_idx, batch_idx]
#                 @inbounds out[dim1, dim2, dim3, channel_idx, batch_idx] = ct*ct
#                 return 
#             end
#             ct += 1
#         end
#         @inbounds out[dim1, dim2, dim3, channel_idx, batch_idx] = 1f10
#     end
#     return 
# end

# ╔═╡ 8d4ea8ae-1fd6-4448-9852-4beb552425a7
function _kernel_3D_2_batch!(out, org, dim1_l, dim2_l, dim3_l, l, channel_idx, batch_idx)
	i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if i > l
        return
    end 
    temp =  dim2_l*dim3_l
    dim1 = cld(i, temp)
    temp2 = (i-1)%temp+1
    dim2 = cld(temp2, dim3_l)
    dim3 = temp2 % dim3_l + 1
    # 2d DT along dim1
    ct = 1
    @inbounds curr_l = CUDA.sqrt(out[dim1, dim2, dim3, channel_idx, batch_idx])
    @inbounds while ct < curr_l && dim1+ct <= dim1_l
        @inbounds if org[dim1+ct, dim2, dim3, channel_idx, batch_idx] < out[dim1, dim2, dim3, channel_idx, batch_idx]
            @inbounds out[dim1, dim2, dim3, channel_idx, batch_idx] = min(out[dim1, dim2, dim3, channel_idx, batch_idx], muladd(ct,ct,org[dim1+ct, dim2, dim3, channel_idx, batch_idx]))
            @inbounds curr_l = CUDA.sqrt(out[dim1, dim2, dim3, channel_idx, batch_idx])
        end
        ct += 1
    end
    ct = 1
    @inbounds while ct < curr_l && dim1-ct > 0
        @inbounds if org[dim1-ct, dim2, dim3, channel_idx, batch_idx] < out[dim1, dim2, dim3, channel_idx, batch_idx]
            @inbounds out[dim1, dim2, dim3, channel_idx, batch_idx] = min(out[dim1, dim2, dim3, channel_idx, batch_idx], muladd(ct,ct,org[dim1-ct, dim2, dim3, channel_idx, batch_idx]))
            @inbounds curr_l = CUDA.sqrt(out[dim1, dim2, dim3, channel_idx, batch_idx])
        end
        ct += 1
    end
    return
end

# ╔═╡ 1d497ec9-12fb-4289-b2f2-2e51cbf042e5
function _kernel_3D_3_batch!(out, org, dim2_l, dim3_l, l, channel_idx, batch_idx)
	i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if i > l
        return
    end 
    temp =  dim2_l*dim3_l
    dim1 = cld(i, temp)
    temp2 = (i-1)%temp+1
    dim2 = cld(temp2, dim3_l)
    dim3 = temp2 % dim3_l + 1
    # 2d DT along dim3
    ct = 1
    @inbounds curr_l = CUDA.sqrt(out[dim1, dim2, dim3, channel_idx, batch_idx])
    @inbounds while ct < curr_l && dim3+ct <= dim3_l
        @inbounds if org[dim1, dim2, dim3+ct, channel_idx, batch_idx] < out[dim1, dim2, dim3, channel_idx, batch_idx]
            @inbounds out[dim1, dim2, dim3, channel_idx, batch_idx] = min(out[dim1, dim2, dim3, channel_idx, batch_idx], muladd(ct,ct,org[dim1, dim2, dim3+ct, channel_idx, batch_idx]))
            @inbounds curr_l = CUDA.sqrt(out[dim1, dim2, dim3, channel_idx, batch_idx])
        end
        ct += 1
    end
    ct = 1
    @inbounds while ct < curr_l && ct < dim3
        @inbounds if org[dim1, dim2, dim3-ct, channel_idx, batch_idx] < out[dim1, dim2, dim3, channel_idx, batch_idx]
            @inbounds out[dim1, dim2, dim3, channel_idx, batch_idx] = min(out[dim1, dim2, dim3, channel_idx, batch_idx], muladd(ct,ct,org[dim1, dim2, dim3-ct, channel_idx, batch_idx]))
            @inbounds curr_l = CUDA.sqrt(out[dim1, dim2, dim3, channel_idx, batch_idx])
        end
        ct += 1
    end
    return
end 

# ╔═╡ 1062d2aa-902a-42e2-98d2-e560fc63e7ae
"""
```julia
get_GPU_kernels(tfm::Wenbo)
```

Returns an array with the needed kernels for GPU version of `transform(..., tfm::Wenbo)`. This function should be called before calling any GPU version of `transform(..., tfm::Wenbo)`.
"""
function get_GPU_kernels(tfm::Wenbo)
	kernels = []
	# 2D kernels:
	push!(kernels, @cuda launch=false _kernel_2D_1_1!(CuArray{Float32, 2}(undef,0,0), CuArray{Float32, 2}(undef, 0, 0),0,0)) #1
	# push!(kernels, @cuda launch=false _kernel_2D_1_2!(CuArray{Float32, 2}(undef,0,0), CuArray{Bool, 2}(undef, 0, 0),0,0)) 
	push!(kernels, @cuda launch=false _kernel_2D_2!(CuArray{Float32, 2}(undef, 0,0),CuArray{Float32, 2}(undef, 0,0),0,0,0)) #2
	# 3D kernels:
	push!(kernels, @cuda launch=false _kernel_3D_1_1!(CuArray{Float32, 3}(undef, 0, 0, 0), CuArray{Float32, 3}(undef, 0, 0, 0),0,0,0)) #3
	# push!(kernels, @cuda launch=false _kernel_3D_1_2!(CuArray{Float32, 3}(undef, 0, 0, 0), CuArray{Bool, 3}(undef, 0, 0, 0),0,0,0)) 
	push!(kernels, @cuda launch=false _kernel_3D_2!(CuArray{Float32, 3}(undef, 0, 0, 0), CuArray{Float32, 3}(undef, 0, 0, 0),0,0,0,0)) #4
	push!(kernels, @cuda launch=false _kernel_3D_3!(CuArray{Float32, 3}(undef, 0, 0, 0), CuArray{Float32, 3}(undef, 0, 0, 0),0,0,0)) #5
	# GPU_threads:
	GPU_threads = launch_configuration(kernels[1].fun).threads
	println("GPU threads = $GPU_threads.")
	push!(kernels, GPU_threads) #6
	# Batch 2D kernels:
	push!(kernels, @cuda launch=false _kernel_2D_1_1_batch!(CuArray{Float32, 4}(undef,0,0,0,0), CuArray{Float32, 4}(undef, 0,0,0,0),0,0,0,0)) #7
	# push!(kernels, @cuda launch=false _kernel_2D_1_2_batch!(CuArray{Float32, 4}(undef,0,0,0,0), CuArray{Bool, 4}(undef, 0,0,0,0),0,0,0,0)) 
	push!(kernels, @cuda launch=false _kernel_2D_2_batch!(CuArray{Float32, 4}(undef, 0,0,0,0),CuArray{Float32, 4}(undef, 0,0,0,0),0,0,0,0,0)) #8
	# Batch 3D kernels:
	push!(kernels, @cuda launch=false _kernel_3D_1_1_batch!(CuArray{Float32, 5}(undef, 0,0,0,0,0), CuArray{Float32, 5}(undef, 0,0,0,0,0),0,0,0,0,0)) #9
	# push!(kernels, @cuda launch=false _kernel_3D_1_2_batch!(CuArray{Float32, 5}(undef, 0,0,0,0,0), CuArray{Bool, 4}(undef, 0,0,0,0,0),0,0,0,0,0)) 
	push!(kernels, @cuda launch=false _kernel_3D_2_batch!(CuArray{Float32, 5}(undef, 0,0,0,0,0), CuArray{Float32, 5}(undef, 0,0,0,0,0),0,0,0,0,0,0)) #10
	push!(kernels, @cuda launch=false _kernel_3D_3_batch!(CuArray{Float32, 5}(undef, 0,0,0,0,0), CuArray{Float32, 5}(undef, 0,0,0,0,0),0,0,0,0,0)) #11
	return kernels
end

# ╔═╡ 83ded49e-77e3-49da-8c77-ac7375670b3d
function _transform_batch(f::CuArray{T, 5}, tfm::Wenbo, kernels) where T  
	d1, d2, d3, num_channels, batch_size = size(f)
	# println("size = $d1, $d2, $d3")
	l = d1 * d2 * d3
	f_new = CUDA.zeros(d1, d2, d3, num_channels, batch_size)
	@inbounds threads = min(l, kernels[6])
	blocks = cld(l, threads)
	# @inbounds k1 = T<:Bool ? kernels[13] : kernels[12]
	for batch_idx = 1:batch_size
		for channel_idx = 1:num_channels
			@inbounds kernels[9](f_new, f, d2, d3, l, channel_idx, batch_idx; threads, blocks)
			@inbounds kernels[10](f_new, copy(f_new), d1, d2, d3, l, channel_idx, batch_idx; threads, blocks)
	    	@inbounds kernels[11](f_new, copy(f_new), d2, d3, l, channel_idx, batch_idx; threads, blocks)
		end
	end
	return f_new
end

# ╔═╡ f8a18f6e-8d35-43a3-a9a8-ae2f4abfe803
"""
```julia
function transform(f::CuArray{T, 3}, tfm::Wenbo, kernels) where T
```

Applies a squared euclidean distance transform to an input 3D image using the Wenbo algorithm. Returns an array with spatial information embedded in the array elements. GPU version of `transform(..., tfm::Wenbo)`
"""
function transform(f::CuArray{T, 3}, tfm::Wenbo, kernels) where T
    d1, d2, d3 = size(f)
    l = length(f)
    f_new = CUDA.zeros(d1,d2,d3)
    threads = min(l, kernels[6])
    blocks = cld(l, threads)
	# k1 = T<:Bool ? kernels[5] : kernels[4]
    @inbounds kernels[3](f_new, f, d2, d3, l; threads, blocks)
    @inbounds kernels[4](f_new, copy(f_new), d1, d2, d3, l; threads, blocks)
    @inbounds kernels[5](f_new, copy(f_new), d2, d3, l; threads, blocks)
    return f_new
end 

# ╔═╡ 882322db-dd8e-415d-a5d4-b6cc68761f07
md"""
### GPU-Batch
"""

# ╔═╡ 14358a91-52aa-4f39-9d75-884ca53a7ce8
"""
```julia
transform(is_batched::Bool, f::CuArray, tfm::Wenbo, kernels)
```

Applies squared euclidean distance transforms to a number of batch_size N-dimension images using the Wenbo algorithm. Returns an array with spatial information embedded in the array elements. The length of last dimension of input should be equal to the batch size. GPU version of 'CPU-Batch'.
"""
function transform(is_batched::Bool, f::CuArray, tfm::Wenbo, kernels)
	return _transform_batch(f, tfm, kernels)
end 

# ╔═╡ 42a77639-403a-42a1-8d69-fc6bdbc9c613
# begin
# 	img2D_batch = CuArray(rand(Float32, 8,6,2,2))
# 	img3D_batch = CuArray(rand(Float32, 6,6,6,2,2))
# 	""
# end

# ╔═╡ 5c10f6c3-1755-4cb8-a9fd-923447291998
# ks = get_GPU_kernels(Wenbo());

# ╔═╡ ea1da50b-22b5-4323-b0d4-142e717c7e40
# Array(transform(true, img3D_batch, Wenbo(), ks))

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
transform(f::AbstractMatrix, tfm::Wenbo, ex)
```

Applies a squared euclidean distance transform to an input 2D image using the Wenbo algorithm. Returns an array with spatial information embedded in the array elements. Multi-threaded version of `transform(..., tfm::Wenbo)` but utilizes FoldsThreads.jl for different threaded executors. `ex`=(FoldsThreads.DepthFirstEx(), FoldsThreads.NonThreadedEx(), FoldsThreads.WorkStealingEx())
"""
function transform(f::AbstractMatrix, tfm::Wenbo, ex)
	f = boolean_indicator(f)
	@floop ex for i in axes(f, 1)
		@inbounds _transform1!(@view(f[i, :]))
	end
	@floop ex for j in axes(f, 2)
		@inbounds _transform2_EN_DE!(@view(f[:,j]))
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
transform(f::AbstractArray, tfm::Wenbo, ex)
```

Applies a squared euclidean distance transform to an input 3D image using the Wenbo algorithm. Returns an array with spatial information embedded in the array elements. Multi-threaded version of `transform!(..., tfm::Wenbo)` but utilizes FoldsThreads.jl for different threaded executors. `ex`=(FoldsThreads.DepthFirstEx(), FoldsThreads.NonThreadedEx(), FoldsThreads.WorkStealingEx())
"""
function transform(f::AbstractArray, tfm::Wenbo, ex)
	f = boolean_indicator(f)
	@floop ex for i in CartesianIndices(f[1,:,:])
		@inbounds _transform1!(@view(f[:, i]))
	end
	@floop ex for i in CartesianIndices(f[:,1,:])
		@inbounds _transform2_EN_DE!(@view(f[i[1], :, i[2]]))
	end
	@floop ex for i in CartesianIndices(f[:,:,1])
		@inbounds _transform2_EN_DE!(@view(f[i, :]))
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
# ╟─9db7eb7e-e47d-4e8d-81c0-f597eae51c04
# ╠═167c008e-5a5f-4ba1-b1ff-2ae137b10c98
# ╟─cf740dd8-79bb-4dd8-b40c-7efcf7844256
# ╟─16991d8b-ec84-49d0-90a9-15a78f1668bb
# ╟─e7dbc916-c5cb-4f86-8ea1-adbcb0bdf8ea
# ╟─32a4bf03-98f8-4ed9-9c12-f45c09b0b0dd
# ╟─89fed2a6-b09e-47b1-a020-efed76ba57de
# ╠═423df2ac-b9a2-4d59-b5fc-8de0e8cc6691
# ╟─dd8014b7-3960-4a2e-878c-c86bbc5e7303
# ╠═b2328983-1c71-49b8-9b43-39bb3febf54b
# ╟─58e1cdff-59b8-44d9-a1b7-ecc14b09556c
# ╟─d663cf13-4a3a-4667-8971-ddb5c455d85c
# ╟─0f0675ad-899d-4808-9757-deaae19a58a5
# ╠═7fecbf6c-59b0-4465-a7c3-c5217b3980c0
# ╟─37cccaee-053d-4f9c-81ef-58b274ec25b8
# ╠═f1977b4e-1834-449a-a8c9-f984a55eeca4
# ╟─948a0099-bc78-4707-9fa1-ad5dc59c34a5
# ╠═84555ba9-ac32-4409-81e4-1e21d02aa1a1
# ╟─8da39536-8765-40fe-a158-335c905e99e6
# ╠═1062d2aa-902a-42e2-98d2-e560fc63e7ae
# ╟─c41c40b2-e23a-4ddd-a4ae-62b37e399f5c
# ╟─ad52080b-7d59-459d-829d-2a77ddf12c5f
# ╟─b5963be3-7794-4ae0-9330-6177a82605ef
# ╟─927f25f9-1687-415f-b5bb-8a8f40afdd0f
# ╟─2a1c7734-805e-4971-9e07-a39c840457f0
# ╟─716e061a-dec8-4515-b2ce-f893ca23f99e
# ╟─aaf5a46a-67c7-457b-bc83-d1c2163583b8
# ╟─25b46272-9f45-45f1-bf81-128a4bcf041f
# ╠═58441e91-b837-496c-b1db-5dd428a6eba7
# ╟─01719cd4-f69e-47f5-9d84-36229fc3e73c
# ╟─24527d9b-1fa2-443d-ad4c-76a53b1ba4c2
# ╟─36bee155-1c27-40be-a956-30ef54ab14ef
# ╟─22c9dd53-6ae6-45f9-8a44-c3777aefef5c
# ╟─678c8a54-0b50-4419-8984-e0f0507e9b48
# ╟─a0dc7da8-54dc-43d6-a501-49ae35951563
# ╟─e4872d2f-1931-4019-993f-14c318362be6
# ╟─8d4ea8ae-1fd6-4448-9852-4beb552425a7
# ╟─1d497ec9-12fb-4289-b2f2-2e51cbf042e5
# ╟─83ded49e-77e3-49da-8c77-ac7375670b3d
# ╠═f8a18f6e-8d35-43a3-a9a8-ae2f4abfe803
# ╟─882322db-dd8e-415d-a5d4-b6cc68761f07
# ╠═14358a91-52aa-4f39-9d75-884ca53a7ce8
# ╟─42a77639-403a-42a1-8d69-fc6bdbc9c613
# ╟─5c10f6c3-1755-4cb8-a9fd-923447291998
# ╟─ea1da50b-22b5-4323-b0d4-142e717c7e40
# ╟─ebee3240-63cf-4323-9755-a135834208c8
# ╟─fccb36b9-ee1b-411f-aded-147a88b23872
# ╠═88806a34-a025-40b2-810d-b3320a137543
# ╟─91e09975-e7df-4d05-9e77-dbd6c35430f0
# ╠═da6e01c4-5ef9-4628-a77f-4b43a05aad36
