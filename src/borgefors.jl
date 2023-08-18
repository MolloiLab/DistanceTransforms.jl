
"""
## DistanceTransform

```julia
abstract type DistanceTransform end
```
Main type for all distance transforms
"""
abstract type DistanceTransform end


"""
## Borgefors

```julia
struct Borgefors{T} <: DistanceTransform end
```

Prepares an array to be `transform`ed using the 3-4 chamfer algorithm laid out in 'Distance transformations in digital images, Computer Vision, Graphics, and Image Processing' [Gunilla Borgefors](https://studentportalen.uu.se/uusp-filearea-tool/download.action?nodeId=214320&toolAttachmentId=64777)
"""
struct Borgefors <: DistanceTransform end


"""
## transform (Borgefors)
```julia
transform(img::AbstractMatrix, dt::AbstractMatrix, tfm::Borgefors)

transform(img::AbstractArray, dt::AbstractArray, tfm::Borgefors)
```

2D chamfer distance transform using the 3-4 chamfer algorithm laid out in 'Distance transformations in digital images, Computer Vision, Graphics, and Image Processing' [Gunilla Borgefors](https://studentportalen.uu.se/uusp-filearea-tool/download.action?nodeId=214320&toolAttachmentId=64777)

3D chamfer distance transform using the 3-4 chamfer algorithm laid out in 'Distance transformations in digital images, Computer Vision, Graphics, and Image Processing' [Gunilla Borgefors](https://studentportalen.uu.se/uusp-filearea-tool/download.action?nodeId=214320&toolAttachmentId=64777)
"""
function transform(img::AbstractMatrix, dt::AbstractMatrix, tfm::Borgefors)
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

function transform(img::AbstractArray, dt::AbstractArray, tfm::Borgefors)
    for z in 1:size(img)[3]
        dt[:, :, z] = transform(img[:, :, z], dt[:, :, z], tfm)
    end
    return dt
end
