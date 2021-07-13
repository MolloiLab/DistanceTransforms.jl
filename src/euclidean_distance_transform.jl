function euclidean_distance_transform(img)
    f = ImageMorphology.feature_transform(.!(Bool.(img)))
    return foreground_dtm = ImageMorphology.distance_transform(f)
end
