### A Pluto.jl notebook ###
# v0.19.14

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
			while pointerA <= l && @inbounds f[pointerA] == 0
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
function _encode(leftD, rightf)
	if rightf == 1f10
		return -leftD
	end
	idx = 0
	while rightf>1
		rightf  /=10
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
function _transform2!(f::AbstractVector)
	l = length(f)
	pointerA = 1
	while pointerA<=l && @inbounds f[pointerA] <= 1
		pointerA += 1
	end
	p = 0
	while pointerA<=l
		@inbounds curr = f[pointerA]
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
		temp = 1
		templ = length(f) - pointerA
		while temp <= templ && muladd(temp, temp, -curr) < 0
			@inbounds curr = min(curr, muladd(temp, temp, f[pointerA+temp]))
			temp += 1
		end
		@inbounds f[pointerA] = _encode(curr, prev)
		pointerA+=1
		while pointerA<=l && @inbounds f[pointerA] <= 1
			pointerA += 1
		end
	end
	i = 0
	while i<l
		i+=1
		f[i] = floor(abs(f[i]))
	end
end

# ╔═╡ 89fed2a6-b09e-47b1-a020-efed76ba57de
function _transform3!(f)
	for i in axes(f, 1)
		@inbounds _transform1!(@view(f[i, :]))
	end
	for j in axes(f, 2)
		@inbounds _transform2!(@view(f[:,j]))
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
	for j in axes(f, 2)
		@inbounds _transform2!(@view(f[:,j]))
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
	for i in axes(f, 3)
		@inbounds _transform3!(@view(f[:, :, i]))
	end
	for j in CartesianIndices(f[:,:,1])
		@inbounds _transform2!(@view(f[j, :]))
	end
	return f
end 

# ╔═╡ 58e1cdff-59b8-44d9-a1b7-ecc14b09556c
md"""
## Multi-Threaded
"""

# ╔═╡ d663cf13-4a3a-4667-8971-ddb5c455d85c
function _transform4!(f)
	Threads.@threads for i in axes(f, 1)
		@inbounds _transform1!(@view(f[i, :]))
	end
	Threads.@threads for j in axes(f, 2)
		@inbounds _transform2!(@view(f[:, j]))
	end
end

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
		@inbounds _transform2!(@view(f[:, j]))
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
	Threads.@threads for i in axes(f, 3)
		@inbounds _transform4!(@view(f[:, :, i]))
	end
	Threads.@threads for j in CartesianIndices(f[:,:,1])
		@inbounds _transform2!(@view(f[j, :]))
	end
	return f
end 

# ╔═╡ 8da39536-8765-40fe-a158-335c905e99e6
md"""
## GPU
"""

# ╔═╡ 6b98e312-826f-4259-8d7c-f6cb61c4c27d
ks = []

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
	@inbounds if f[row, col] == 1
		return 
	end
	ct = 1
	curr_l = min(col-1, row_l-col)
	while ct <= curr_l
		@inbounds if f[row, col-ct] == 1 || f[row, col+ct] == 1
			@inbounds out[row, col] = ct*ct
			return 
		end
		ct += 1
	end
	while ct < col
		@inbounds if f[row, col-ct] == 1
			@inbounds out[row, col] = ct*ct
			return 
		end
		ct += 1    
	end
	while col+ct <= row_l
		@inbounds if f[row, col+ct] == 1
			@inbounds out[row, col] = ct*ct
			return 
		end
		ct += 1
	end
	@inbounds out[row, col] = 1f10
	return 
end

# ╔═╡ b5963be3-7794-4ae0-9330-6177a82605ef
function _kernel_2D_1_2!(out, f, row_l, l)
	i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
	if i > l
		return
	end 
	row = cld(i, row_l)
	col = i%row_l+1
	@inbounds if f[row, col]
		return 
	end
	ct = 1
	curr_l = min(col-1, row_l-col)
	while ct <= curr_l
		@inbounds if f[row, col-ct] || f[row, col+ct]
			@inbounds out[row, col] = ct*ct
			return 
		end
		ct += 1
	end
	while ct < col
		@inbounds if f[row, col-ct]
			@inbounds out[row, col] = ct*ct
			return 
		end
		ct += 1    
	end
	while col+ct <= row_l
		@inbounds if f[row, col+ct]
			@inbounds out[row, col] = ct*ct
			return 
		end
		ct += 1
	end
	@inbounds out[row, col] = 1f10
	return 
end

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

# ╔═╡ b6b10ccd-bf6d-413c-b37a-446f5671e3d1
begin
	# k1 = @cuda launch=false _kernel_2D_1_1!(CuArray{Float32, 2}(undef,0,0),CuArray{Int64, 2}(undef, 0, 0),0,0) # 1
	# GPU_threads = launch_configuration(k1.fun).threads
	# k2 = @cuda launch=false _kernel_2D_1_2!(CuArray{Float32, 2}(undef,0,0), CuArray{Bool, 2}(undef, 0, 0),0,0) # 2
	# k3 = @cuda launch=false _kernel_2D_2!(CuArray{Float32, 2}(undef,0,0),CuArray{Float32, 2}(undef, 0,0),0,0,0) # 3
end

# ╔═╡ 58441e91-b837-496c-b1db-5dd428a6eba7
"""
```julia
transform(f::CuArray{Bool, 2}, tfm::Wenbo)
```

Applies a squared euclidean distance transform to an input 2D boolean image using the Wenbo algorithm. Returns an array with spatial information embedded in the array elements. GPU version of `transform(..., tfm::Wenbo)`
"""
function transform(f::CuArray{Bool, 2}, tfm::Wenbo)
    col_length, row_length = size(f)
    l = length(f)
    f_new = CUDA.zeros(col_length,row_length)
    threads = min(l, ks[8])
    blocks = cld(l, threads)
    # @cuda blocks=blocks threads=threads _kernel_2D_1_2!(f_new, f, row_length, l) #k2
	ks[2](f_new, f, row_length, l; threads, blocks)
	# @cuda blocks=blocks threads=threads _kernel_2D_2!(deepcopy(f_new), f_new, row_length, col_length, l) #k3
    ks[3](deepcopy(f_new), f_new, row_length, col_length, l; threads, blocks)
    CUDA.reclaim()
	return f_new
end

# ╔═╡ abee05dd-cce0-416c-b473-292fee0d3172
"""
```julia
transform(f::CuArray{Int, 2}, tfm::Wenbo)
```

Applies a squared euclidean distance transform to an input 2D image using the Wenbo algorithm. Returns an array with spatial information embedded in the array elements. GPU version of `transform(..., tfm::Wenbo)`
"""
function transform(f::CuArray{Int, 2}, tfm::Wenbo)
    col_length, row_length = size(f)
    l = length(f)
    f_new = CUDA.zeros(col_length,row_length)
    threads = min(l, ks[8])
    blocks = cld(l, threads)
	# @cuda blocks=blocks threads=threads _kernel_2D_1_1!(f_new, f, row_length, l) # k1
    ks[1](f_new, f, row_length, l; threads, blocks)
	# @cuda blocks=blocks threads=threads _kernel_2D_2!(deepcopy(f_new), f_new, row_length, col_length, l) #k3
    ks[3](deepcopy(f_new), f_new, row_length, col_length, l; threads, blocks)
    CUDA.reclaim()
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
    # 1d DT alone dim2
    @inbounds if f[dim1, dim2, dim3] != 1
        ct = 1
        curr_l = CUDA.min(dim2-1, dim2_l-dim2)
        while ct <= curr_l
            @inbounds if f[dim1, dim2-ct, dim3]==1 || f[dim1, dim2+ct, dim3]==1
                @inbounds out[dim1, dim2, dim3] = ct*ct
                return 
            end
            ct += 1
        end
        while ct < dim2
            @inbounds if f[dim1, dim2-ct, dim3]==1
                @inbounds out[dim1, dim2, dim3] = ct*ct
                return 
            end
            ct += 1    
        end
        while dim2+ct <= dim2_l
            @inbounds if f[dim1, dim2+ct, dim3]==1
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
function _kernel_3D_1_2!(out, f, dim2_l, dim3_l, l)
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
    @inbounds if !f[dim1, dim2, dim3]
        ct = 1
        curr_l = CUDA.min(dim2-1, dim2_l-dim2)
        while ct <= curr_l
            @inbounds if f[dim1, dim2-ct, dim3] || f[dim1, dim2+ct, dim3]
                @inbounds out[dim1, dim2, dim3] = ct*ct
                return 
            end
            ct += 1
        end
        while ct < dim2
            @inbounds if f[dim1, dim2-ct, dim3]
                @inbounds out[dim1, dim2, dim3] = ct*ct
                return 
            end
            ct += 1    
        end
        while dim2+ct <= dim2_l
            @inbounds if f[dim1, dim2+ct, dim3]
                @inbounds out[dim1, dim2, dim3] = ct*ct
                return 
            end
            ct += 1
        end
        @inbounds out[dim1, dim2, dim3] = 1f10
    end
    return 
end

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
    # 2d DT alone dim1
    # @inbounds out[dim1, dim2, dim3] = dim1
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
    # 2d DT alone dim3
    # @inbounds out[dim1, dim2, dim3] = dim1
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

# ╔═╡ 1062d2aa-902a-42e2-98d2-e560fc63e7ae
# """
# ```julia
# get_GPU_kernels(tfm::Wenbo)
# ```

# Returns an array with the needed kernels for GPU version of `transform(..., tfm::Wenbo)`. This function should be called before calling any GPU version of `transform(..., tfm::Wenbo)`.
# """

function initialize_GPU_kernels(tfm::Wenbo)
	push!(ks, @cuda launch=false _kernel_2D_1_1!(CuArray{Float32, 2}(undef,0,0), CuArray{Int64, 2}(undef, 0, 0),0,0)) #1
	push!(ks, @cuda launch=false _kernel_2D_1_2!(CuArray{Float32, 2}(undef,0,0), CuArray{Bool, 2}(undef, 0, 0),0,0)) #2
	push!(ks, @cuda launch=false _kernel_2D_2!(CuArray{Float32, 2}(undef, 0,0),CuArray{Float32, 2}(undef, 0,0),0,0,0)) #3
	push!(ks, @cuda launch=false _kernel_3D_1_1!(CuArray{Float32, 3}(undef, 0, 0, 0), CuArray{Int64, 3}(undef, 0, 0, 0),0,0,0)) #4
	push!(ks, @cuda launch=false _kernel_3D_1_2!(CuArray{Float32, 3}(undef, 0, 0, 0), CuArray{Bool, 3}(undef, 0, 0, 0),0,0,0)) #5
	push!(ks, @cuda launch=false _kernel_3D_2!(CuArray{Float32, 3}(undef, 0, 0, 0), CuArray{Float32, 3}(undef, 0, 0, 0),0,0,0,0)) #6
	push!(ks, @cuda launch=false _kernel_3D_3!(CuArray{Float32, 3}(undef, 0, 0, 0), CuArray{Float32, 3}(undef, 0, 0, 0),0,0,0)) #7
	GPU_threads = launch_configuration(ks[1].fun).threads
	println("$GPU_threads GPU threads.")
	push!(ks, GPU_threads) #8
	return nothing
end

# ╔═╡ 70ff72b3-4e43-40d2-8279-077801dc9ac8
# begin
# 	k4 = @cuda launch=false _kernel_3D_1_1!(CuArray{Float32, 3}(undef, 0, 0, 0), CuArray{Int64, 3}(undef, 0, 0, 0),0,0,0)
# 	k5 = @cuda launch=false _kernel_3D_1_2!(CuArray{Float32, 3}(undef, 0, 0, 0), CuArray{Bool, 3}(undef, 0, 0, 0),0,0,0)
# 	k6 = @cuda launch=false _kernel_3D_2!(CuArray{Float32, 3}(undef, 0, 0, 0), CuArray{Float32, 3}(undef, 0, 0, 0),0,0,0,0)
# 	k7 = @cuda launch=false _kernel_3D_3!(CuArray{Float32, 3}(undef, 0, 0, 0), CuArray{Float32, 3}(undef, 0, 0, 0),0,0,0)
# end

# ╔═╡ f8a18f6e-8d35-43a3-a9a8-ae2f4abfe803
"""
```julia
function transform(f::CuArray{Bool, 3}, tfm::Wenbo)
```

Applies a squared euclidean distance transform to an input 3D image using the Wenbo algorithm. Returns an array with spatial information embedded in the array elements. GPU version of `transform(..., tfm::Wenbo)`
"""
function transform(f::CuArray{Bool, 3}, tfm::Wenbo) 
	# println("GPU 3D Bool")
    d1, d2, d3 = size(f)
    l = length(f)
    f_new = CUDA.zeros(d1,d2,d3)
    threads = min(l, ks[8])
    blocks = cld(l, threads)
	# @cuda blocks=blocks threads=threads _kernel_3D_1_2!(f_new, f, d2, d3, l) #k5
    ks[5](f_new, f, d2, d3, l; threads, blocks)
	# @cuda blocks=blocks threads=threads _kernel_3D_2!(f_new, deepcopy(f_new), d1, d2, d3, l) #k6
    ks[6](f_new, deepcopy(f_new), d1, d2, d3, l; threads, blocks)
	# @cuda blocks=blocks threads=threads _kernel_3D_3!(f_new, deepcopy(f_new), d2, d3, l) # k7
    ks[7](f_new, deepcopy(f_new), d2, d3, l; threads, blocks)
	CUDA.reclaim()
    return f_new
end 

# ╔═╡ 44d9a02a-b737-4989-9852-e40515accb8b
"""
```julia
transform(f::CuArray{Int, 3}, tfm::Wenbo)
```

Applies a squared euclidean distance transform to an input 3D image using the Wenbo algorithm. Returns an array with spatial information embedded in the array elements. GPU version of `transform(..., tfm::Wenbo)`
"""
function transform(f::CuArray{Int, 3}, tfm::Wenbo)
	# println("GPU 3D T")
    d1, d2, d3 = size(f)
    l = length(f)
    f_new = CUDA.zeros(d1,d2,d3)
    threads = min(l, ks[8])
    blocks = cld(l, threads)
	# @cuda blocks=blocks threads=threads _kernel_3D_1_1!(f_new, f, d2, d3, l) #k4
    ks[4](f_new, f, d2, d3, l; threads, blocks)
	# @cuda blocks=blocks threads=threads _kernel_3D_2!(f_new, deepcopy(f_new), d1, d2, d3, l) #k6
    ks[6](f_new, deepcopy(f_new), d1, d2, d3, l; threads, blocks)
	# @cuda blocks=blocks threads=threads _kernel_3D_3!(f_new, deepcopy(f_new), d2, d3, l) # k7
    ks[7](f_new, deepcopy(f_new), d2, d3, l; threads, blocks)
	CUDA.reclaim()
    return f_new
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
		@inbounds _transform2!(@view(f[:, j]))
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
	@floop ex for k in axes(f, 3)
		@inbounds _transform4!(@view(f[:, :, k]))
	end
	@floop ex for i in CartesianIndices(f[:,:,1])
		@inbounds _transform2!(@view(f[i, :]))
	end
	return f
end 

# ╔═╡ 70762a8f-a070-4194-98c4-09096b2f3ec9


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
# ╟─8da39536-8765-40fe-a158-335c905e99e6
# ╠═6b98e312-826f-4259-8d7c-f6cb61c4c27d
# ╠═1062d2aa-902a-42e2-98d2-e560fc63e7ae
# ╟─c41c40b2-e23a-4ddd-a4ae-62b37e399f5c
# ╟─ad52080b-7d59-459d-829d-2a77ddf12c5f
# ╟─b5963be3-7794-4ae0-9330-6177a82605ef
# ╟─927f25f9-1687-415f-b5bb-8a8f40afdd0f
# ╠═b6b10ccd-bf6d-413c-b37a-446f5671e3d1
# ╠═58441e91-b837-496c-b1db-5dd428a6eba7
# ╠═abee05dd-cce0-416c-b473-292fee0d3172
# ╟─01719cd4-f69e-47f5-9d84-36229fc3e73c
# ╟─24527d9b-1fa2-443d-ad4c-76a53b1ba4c2
# ╟─36bee155-1c27-40be-a956-30ef54ab14ef
# ╟─22c9dd53-6ae6-45f9-8a44-c3777aefef5c
# ╟─678c8a54-0b50-4419-8984-e0f0507e9b48
# ╠═70ff72b3-4e43-40d2-8279-077801dc9ac8
# ╠═f8a18f6e-8d35-43a3-a9a8-ae2f4abfe803
# ╠═44d9a02a-b737-4989-9852-e40515accb8b
# ╟─ebee3240-63cf-4323-9755-a135834208c8
# ╟─fccb36b9-ee1b-411f-aded-147a88b23872
# ╠═88806a34-a025-40b2-810d-b3320a137543
# ╟─91e09975-e7df-4d05-9e77-dbd6c35430f0
# ╠═da6e01c4-5ef9-4628-a77f-4b43a05aad36
# ╠═70762a8f-a070-4194-98c4-09096b2f3ec9
