```@meta
CurrentModule = DistanceTransforms
```

# DistanceTransforms

```@index
```

## Function documentation
```@docs
transform(img::AbstractMatrix{T}, tfm::Chamfer)
transform(img::AbstractArray{T,3}, tfm::Chamfer)
transform(f::AbstractVector{T}, tfm::SquaredEuclidean)
transform(img::AbstractMatrix{T}, tfm::SquaredEuclidean)
transform!(img::AbstractMatrix{T}, tfm::SquaredEuclidean, nthreads)
transform!(img::CuArray{T,2}, tfm::SquaredEuclidean)
euclidean(img)
euclidean(img::BitArray)
boolean_indicator(f)
```

```@autodocs
Modules = [DistanceTransforms]
```
