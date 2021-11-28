### A Pluto.jl notebook ###
# v0.17.2

using Markdown
using InteractiveUtils

# ╔═╡ d9f9bd70-508d-11ec-1d5d-b3592d653992
begin
	let
		using Pkg
		Pkg.activate(mktempdir())
		Pkg.Registry.update()
		Pkg.add("PlutoUI")
		Pkg.add(url="https://github.com/Dale-Black/DistanceTransforms.jl")
		Pkg.add(url="https://github.com/Dale-Black/Losers.jl")
	end
	
	using PlutoUI
	using DistanceTransforms
	using Losers
end

# ╔═╡ 5398d829-ad81-4971-94b5-f87eeaaa381e
md"""
## Import packages
"""

# ╔═╡ 44119707-2397-4e7f-83f7-f362eae6004f
TableOfContents()

# ╔═╡ 6e03547c-d9b6-4972-9574-7cb4f30948f9
md"""
## Distance transforms + deep learning
One area of recent interest in deep learning has been distance transform-based loss functions to directly reduce Hausdorff loss and other boundary specific loss metrics [D. Karimi and S. E. Salcudean](http://arxiv.org/abs/1904.10030).

This package can be utilized directly in various training loops, along with [Losers.jl](https://github.com/Dale-Black/Losers.jl) to train segmentation networks, like the common UNet architecture and directly reduce Hausdorff distance using the `hausdorff()` loss function.

In practice, it is common to combine `hausdorff` with the `dice` loss function to help stabalize during training.

Below we will show how one might set this up, without actually training on anything.
"""

# ╔═╡ 2810025c-9940-4da2-a587-938c425f511e
y1 = [1 1 1 0; 1 1 1 0; 1 1 1 0; 1 1 1 0]

# ╔═╡ dfc81bb5-29fa-46f2-a335-66f541205d22
y = cat(y1, y1; dims=3)

# ╔═╡ 89347efa-ee13-4c61-a4e6-3d4d1a77955d
ŷ = copy(y)

# ╔═╡ 96d6cb3f-a500-490b-9bc9-500731af2651
md"""
Given some fake three dimensional array, lets apply a distance transform to this array for use in the `hausdorff` loss function
"""

# ╔═╡ 646d3c5b-e72b-49ac-b1e1-74d2d89c57c2
y_dtm = euclidean(y)

# ╔═╡ 10f64f57-e052-4895-9fe3-58e1c4fb2d88
ŷ_dtm = euclidean(ŷ)

# ╔═╡ 6e52eae8-e1cd-4d07-a000-d591b7bfa3ff
hausdorff(ŷ, y, ŷ_dtm, y_dtm)

# ╔═╡ b7c539d1-0e65-4716-ad23-cf4400bf4bf7
md"""
As one would expect, the difference between two identical arrays is zero
"""

# ╔═╡ 2780485d-ce45-4d5f-a88c-d4823bcf4273
begin
	y2 = rand([0, 1], 10, 10, 10)
	ŷ2 = rand([0, 1], 10, 10, 10)

	y2_dtm = euclidean(y2)
	ŷ2_dtm = euclidean(ŷ2)
end

# ╔═╡ 423704a8-926e-4fd3-98c4-7da8374a35a5
hausdorff(ŷ2, y2, ŷ2_dtm, y2_dtm)

# ╔═╡ 5d1ac21a-8073-4817-869b-5500649e486f
md"""
## GPU considerations
When using distance transform-based loss functions for deep learning, one must consider GPUs. The benefit of DistanceTransforms.jl is that GPU compatible distance transform algorithms are ready to use, out-of-the-box. 

We will look at how one would go about this, without actually calling a CUDA array from the CUDA.jl library.
"""

# ╔═╡ cfd9c96d-1ce7-4edc-954c-4d844aff6629
begin
	y3 = rand([0, 1], 10, 10, 10)
	ŷ3 = rand([0, 1], 10, 10, 10)
end

# ╔═╡ 6345df57-dba7-4608-9e08-9c9d981810e5
begin
	y3_bool = boolean_indicator(y3)
	ŷ3_bool = boolean_indicator(ŷ3)
end

# ╔═╡ dde2505d-1a69-4201-a75a-e29a62197163
begin
	tfm1 = SquaredEuclidean(y3_bool)
	tfm2 = SquaredEuclidean(ŷ3_bool)
end

# ╔═╡ d2a987a3-f44c-4239-9e96-d83ae16c265e
begin
	y3_dtm = transform(y3_bool, tfm1)
	ŷ3_dtm = transform(ŷ3_bool, tfm1)
end

# ╔═╡ 90467d7a-785d-44d3-92bc-9d05bcb5f5b4
hausdorff(ŷ3, y3, ŷ3_dtm, y3_dtm)

# ╔═╡ 43e694a4-bbd5-4072-a4de-0ea65b16c906
md"""
If we were to be running an actual training loop it might look something like this...

```julia
# Begin training loop
for epoch in 1:max_epochs
	step = 0
	@show epoch
	
	# Loop through training data
	for (xs, ys) in train_loader
		step += 1
		@show step

		# Send data to GPU
		xs, ys = xs |> gpu, ys |> gpu		
		gs = Flux.gradient(ps) do
			ŷs = model(xs)

			# Apply distance transform using GPU compatible `SquaredEuclidean`
			# Data will usually be 4D or 5D [x, y, (z), channel, batch]
			ŷs, ys = boolean_indicator(ŷs), boolean_indicator(ys)
			tfm1, tfm2 = SquaredEuclidean(ŷs), SquaredEuclidean(ys)
			ŷs_dtm, ys_dtm = transform!(ŷs_dtm), transform!(ys_dtm)
			loss = hausdorff(ŷs, ys, ŷs_dtm, ys_dtm)
			return loss
		end
		Flux.update!(optimizer, ps, gs)
	end
end
```
"""

# ╔═╡ Cell order:
# ╟─5398d829-ad81-4971-94b5-f87eeaaa381e
# ╠═d9f9bd70-508d-11ec-1d5d-b3592d653992
# ╠═44119707-2397-4e7f-83f7-f362eae6004f
# ╟─6e03547c-d9b6-4972-9574-7cb4f30948f9
# ╠═2810025c-9940-4da2-a587-938c425f511e
# ╠═dfc81bb5-29fa-46f2-a335-66f541205d22
# ╠═89347efa-ee13-4c61-a4e6-3d4d1a77955d
# ╟─96d6cb3f-a500-490b-9bc9-500731af2651
# ╠═646d3c5b-e72b-49ac-b1e1-74d2d89c57c2
# ╠═10f64f57-e052-4895-9fe3-58e1c4fb2d88
# ╠═6e52eae8-e1cd-4d07-a000-d591b7bfa3ff
# ╟─b7c539d1-0e65-4716-ad23-cf4400bf4bf7
# ╠═2780485d-ce45-4d5f-a88c-d4823bcf4273
# ╠═423704a8-926e-4fd3-98c4-7da8374a35a5
# ╟─5d1ac21a-8073-4817-869b-5500649e486f
# ╠═cfd9c96d-1ce7-4edc-954c-4d844aff6629
# ╠═6345df57-dba7-4608-9e08-9c9d981810e5
# ╠═dde2505d-1a69-4201-a75a-e29a62197163
# ╠═d2a987a3-f44c-4239-9e96-d83ae16c265e
# ╠═90467d7a-785d-44d3-92bc-9d05bcb5f5b4
# ╟─43e694a4-bbd5-4072-a4de-0ea65b16c906
