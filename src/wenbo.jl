struct Wenbo <: DistanceTransform end

"""
    transform(f, tfm::Wenbo; output=zeros(length(f)), pointerA=1, pointerB=1)

Assume length(f)>0. This is a one pass algorithm. Time complexity=O(n). Space complexity=O(1)
"""
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

"""
    function _DT1(input, output, i, j)

Helper function for 1-D Wenbo distance transform `transform(f::AbstractVector, tfm::Wenbo)`
"""
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

"""
    transform(img::AbstractMatrix, tfm::Wenbo; output=zeros(size(img)), pointerA=1, pointerB=1)

2-D Wenbo Distance Transform.
"""
function transform(img::AbstractMatrix, tfm::Wenbo; output=zeros(size(img)), pointerA=1, pointerB=1)
	# This is a worst case = O(n^3) implementation
	for i in axes(img, 1)
	    output[i, :] = transform(img[i, :], Wenbo(); output=output[i, :], pointerA=pointerA, pointerB=pointerB) 
	end

	for j in axes(img, 2)
	    output[:, j] = _DT2(output[:, j]; output=output[:, j], pointerA=pointerA) 
	end
	return output
end


"""
    _DT2(f; output=zeros(length(f)), pointerA=1)

Helper function for 2-D Wenbo distance transform `transform(f::AbstractVector, tfm::Wenbo)`
Computes the vertical operation.
"""
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

"""
    transform(f::AbstractArray, tfm::Wenbo; D=zeros(size(f)), pointerA=1, pointerB=1)

3-D Wenbo Distance Transform.
"""
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