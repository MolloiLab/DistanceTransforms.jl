### A Pluto.jl notebook ###
# v0.19.11

using Markdown
using InteractiveUtils

# ╔═╡ fe947108-2f07-425d-8ded-5a2d8322a0a7
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

# ╔═╡ 32dcf5b9-fb91-4891-8be0-a578947aa484
TableOfContents()

# ╔═╡ ec56a36e-72b3-42d2-9f8f-6a332401b9b9
"""
```julia
abstract type DistanceTransform end
```
Main type for all distance transforms
"""
abstract type DistanceTransform end

# ╔═╡ f6dd7123-0069-4154-a3c3-b9f95c49d21d
md"""
# `Borgefors`
"""

# ╔═╡ aafc9419-3105-45ff-905e-610843528e04
"""

```julia
struct Borgefors{T} <: DistanceTransform end
```

Prepares an array to be `transform`ed using the 3-4 chamfer algorithm laid out in 'Distance transformations in digital images, Computer Vision, Graphics, and Image Processing' [Gunilla Borgefors](https://studentportalen.uu.se/uusp-filearea-tool/download.action?nodeId=214320&toolAttachmentId=64777)
"""
struct Borgefors <: DistanceTransform end

# ╔═╡ e68e45ae-fbc1-403e-bc45-9d4f227a933f
md"""
## Regular
"""

# ╔═╡ e9afda48-bdbf-4e6b-8fb8-94324a76a7e7
md"""
### 2D
"""

# ╔═╡ 408fa845-a280-47ce-aedd-a53ffe3376f7
"""
```julia
transform(img::AbstractMatrix, dt::AbstractMatrix, tfm::Borgefors)
```

2D chamfer distance transform using the 3-4 chamfer algorithm laid out in 'Distance transformations in digital images, Computer Vision, Graphics, and Image Processing' [Gunilla Borgefors](https://studentportalen.uu.se/uusp-filearea-tool/download.action?nodeId=214320&toolAttachmentId=64777)
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

# ╔═╡ 3616fedc-53d6-4eec-90b5-14c1c98a83ba
md"""
### 3D
"""

# ╔═╡ ab0b89ad-8cf1-4cb0-9666-487e6973e414
"""
```julia
transform(img::AbstractArray, dt::AbstractArray, tfm::Borgefors)
```

3D chamfer distance transform using the 3-4 chamfer algorithm laid out in 'Distance transformations in digital images, Computer Vision, Graphics, and Image Processing' [Gunilla Borgefors](https://studentportalen.uu.se/uusp-filearea-tool/download.action?nodeId=214320&toolAttachmentId=64777)
"""
function transform(img::AbstractArray, dt::AbstractArray, tfm::Borgefors)
    for z in 1:size(img)[3]
        dt[:, :, z] = transform(img[:, :, z], dt[:, :, z], tfm)
    end
    return dt
end

# ╔═╡ Cell order:
# ╠═fe947108-2f07-425d-8ded-5a2d8322a0a7
# ╠═32dcf5b9-fb91-4891-8be0-a578947aa484
# ╠═ec56a36e-72b3-42d2-9f8f-6a332401b9b9
# ╠═f6dd7123-0069-4154-a3c3-b9f95c49d21d
# ╠═aafc9419-3105-45ff-905e-610843528e04
# ╟─e68e45ae-fbc1-403e-bc45-9d4f227a933f
# ╟─e9afda48-bdbf-4e6b-8fb8-94324a76a7e7
# ╠═408fa845-a280-47ce-aedd-a53ffe3376f7
# ╟─3616fedc-53d6-4eec-90b5-14c1c98a83ba
# ╠═ab0b89ad-8cf1-4cb0-9666-487e6973e414
