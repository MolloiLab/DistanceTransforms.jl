"""
    chamfer(img::Array{T,2})
    chamfer(img::Array{T,3})

Applies a 3-4 chamfer distance transform to an input image.
Returns an array with spatial information embedded in the
array elements.

# Arguments
- img: 2D or 3D array to be transformed based on location 
    to the nearest background (0) pixel

# Citation
'Distance transformations in digital images,
Computer Vision, Graphics, and Image Processing,
Volume 34, Issue 3, 1986, Pages 344-371' 
[Gunilla Borgefors] (DOI: https://doi.org/10.1016/S0734-189X(86)80047-0.)
"""
function chamfer(img::Array{T,2}, dt=zeros(Int32, size(img)[1], size(img)[2])) where {T}
    w, h = size(img)
    # Forward pass
    x = 1
    y = 1
    if img[x, y] == 0
        dt[x, y] = 65535 # some large value
    end
    for x in 1:(w - 1)
        if img[x + 1, y] == 0
            dt[x + 1, y] = 3 + dt[x, y]
        end
    end
    for y in 1:(h - 1)
        x = 1
        if img[x, y + 1] == 0
            dt[x, y + 1] = min(3 + dt[x, y], 4 + dt[x + 1, y])
        end
        for x in 1:(w - 2)
            if img[x + 1, y + 1] == 0
                dt[x + 1, y + 1] = min(
                    4 + dt[x, y], 3 + dt[x + 1, y], 4 + dt[x + 2, y], 3 + dt[x, y + 1]
                )
            end
        end
        x = w

        if img[x, y + 1] == 0
            dt[x, y + 1] = min(4 + dt[x - 1, y], 3 + dt[x, y], 3 + dt[x - 1, y + 1])
        end
    end

    # Backward pass
    for x in (w - 1):-1:1
        y = h
        if img[x, y] == 0
            dt[x, y] = min(dt[x, y], 3 + dt[x + 1, y])
        end
    end
    for y in (h - 1):-1:1
        x = w
        if img[x, y] == 0
            dt[x, y] = min(dt[x, y], 3 + dt[x, y + 1], 4 + dt[x - 1, y + 1])
        end
        for x in 1:(w - 2)
            if img[x + 1, y] == 0
                dt[x + 1, y] = min(
                    dt[x + 1, y],
                    4 + dt[x + 2, y + 1],
                    3 + dt[x + 1, y + 1],
                    4 + dt[x, y + 1],
                    3 + dt[x + 2, y],
                )
            end
        end
        x = 1
        if img[x, y] == 0
            dt[x, y] = min(
                dt[x, y], 4 + dt[x + 1, y + 1], 3 + dt[x, y + 1], 3 + dt[x + 1, y]
            )
        end
    end
    return dt
end

function chamfer(img::Array{T,3}, dt=zeros(Int32, size(img))) where {T}
    for z in 1:size(img)[3]
        dt[:, :, z] = chamfer(img[:, :, z])
    end
    return dt
end
