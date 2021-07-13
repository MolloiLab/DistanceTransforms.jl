"""
    euclidean_distance_transform(img)

Wrapper function for `ImageMorphology.feature_transform` and
`ImageMorphology.distance_transform`. Returns a true Euclidean
distance transform.

# Arguments
- img: N-dimensional array to be transformed based on location
    to the nearest background (0) pixel

# Citation
'A Linear Time Algorithm for Computing Exact Euclidean Distance
Transforms of Binary Images in Arbitrary Dimensions' [Maurer et al.,
2003] (DOI: 10.1109/TPAMI.2003.1177156)
"""
function euclidean_distance_transform(img)
    f = ImageMorphology.feature_transform(.!(Bool.(img)))
    return foreground_dtm = ImageMorphology.distance_transform(f)
end
