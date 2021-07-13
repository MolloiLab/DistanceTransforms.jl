"""
    squared_euclidean_distance_transform(img::Array{T,1}, is_indicator=true)
    squared_euclidean_distance_transform(img::Array{T,2}, is_indicator=true)

Applies a squared euclidean distance transform to an input image.
Returns an array with spatial information embedded in the array 
elements.

# Arguments
- img: 1D, 2D, or 3D to be transformed based on location 
    to the nearest background (0) pixel
- is_indicator: whether or not the `img` is a binary array
    that needs to be transformed based on the foreground (1)
    pixels

# Citation
'Distance Transforms of Sampled Functions' [Felzenszwalb and
Huttenlocher] (DOI: 10.4086/toc.2012.v008a019)
"""
function squared_euclidean_distance_transform(img::Array{T,1}, is_indicator=true) where {T}
    f = copy(img)
    if is_indicator
        f = @. ifelse(f == 0, 1e10, 0)
    end

    n = length(f)
    k = 1
    v = ones(Int64, length(f))
    z = zeros(Float64, length(f) + 1)
    z[1] = -Inf
    z[2] = Inf
    flag = 0

    # Lower envelope operation
    for q in 2:n
        flag = 0
        while flag != 1
            s = ((f[q] + q^2) - (f[v[k]] + v[k]^2)) / (2 * q - 2 * v[k])
            if s ≤ z[k]
                k -= 1
                flag = 0
            else
                k += 1
                v[k] = q
                z[k] = s
                z[k + 1] = Inf
                flag = 1
            end
        end
    end

    # Distance transform operation
    k = 1
    dt = []
    for q in 1:n
        while z[k + 1] < q
            k = k + 1
        end
        push!(dt, (q - v[k])^2 + f[v[k]])
    end
    return dt
end

function squared_euclidean_distance_transform(img::Array{T,2}, is_indicator=true) where {T}
    rows, columns = size(img)
    dt = zeros(Float64, (rows, columns))
    for x in 1:rows
        dt[x, :] = squared_euclidean_distance_transform(img[x, :], is_indicator)
    end

    for y in 1:columns
        dt[:, y] = squared_euclidean_distance_transform(img[:, y], is_indicator)
    end

    return dt
end
