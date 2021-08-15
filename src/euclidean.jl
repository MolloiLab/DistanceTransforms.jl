"""
    euclidean(img)

Wrapper function for `ImageMorphology.feature_transform` and
`ImageMorphology.distance_transform`. Applies a true Euclidean 
distance transform to the array elements and returns an array 
with spatial information embedded in the elements.

# Arguments
- img: N-dimensional array to be transformed based on location
    to the nearest background (0) pixel

# Citation
'A Linear Time Algorithm for Computing Exact Euclidean Distance
Transforms of Binary Images in Arbitrary Dimensions' [Maurer et al.,
2003] (DOI: 10.1109/TPAMI.2003.1177156)
"""
euclidean(img) = distance_transform(feature_transform(.!(Bool.(img))))
euclidean(img::BitArray) = distance_transform(feature_transform(.!(img)))
