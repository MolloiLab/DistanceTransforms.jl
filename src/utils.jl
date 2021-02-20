"
    detect_edges_3D(img, f)

Modifies `detect_edges` to work with 3D images.
"

function detect_edges_3D(img, f)
	container = Array{Int64}(undef, size(img))
	for k in 1:(size(img)[3])
		container[:, :, k] = detect_edges(img[:, :, k], f)
	end
	return container
end

function compute_dtm(img)
    f = feature_transform(.!(Bool.(img)))
    foreground_dtm = distance_transform(f)
end