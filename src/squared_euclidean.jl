# struct SquaredEuclidean <: DistanceTransform
# end

# function DT1(f; D=zeros(length(f)), v=ones(Int32, length(f)), z=ones(length(f)))
# 	z[1] = -Inf32
# 	z[2] = Inf32
# 	k = 1; # Index of the rightmost parabola in the lower envelope
# 	for q = 2:length(f)
# 		s = ((f[q] + q^2) - (f[v[k]] + v[k]^2)) / (2*q - 2*v[k])
# 	    while s ≤ z[k]
# 	        k = k - 1
# 	        s = ((f[q] + q^2) - (f[v[k]] + v[k]^2)) / (2*q - 2*v[k])
# 	    end
# 	    k = k + 1
# 	    v[k] = q
# 	    z[k] = s
# 		if k ≤ length(f) - 1
# 			z[k+1] = Inf32
# 		else
# 			z[k] = Inf32
# 		end
# 	end
# 	k = 1
# 	for q in 1:length(f)
# 	    while z[k+1] < q
# 	        k = k+1
# 	    end
# 	    D[q] = (q-v[k])^2 + f[v[k]]
# 	end
# 	return D
# end