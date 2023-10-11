### A Pluto.jl notebook ###
# v0.19.26

#> [frontmatter]
#> title = "Deep Learning Usage"
#> category = "Tutorials"

using Markdown
using InteractiveUtils

# ╔═╡ d9f9bd70-508d-11ec-1d5d-b3592d653992
# ╠═╡ show_logs = false
begin
	using Pkg
	Pkg.activate(".")
	Pkg.instantiate()
	
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

# ╔═╡ f134f1c2-503d-41cf-ab9a-eb8c8fca37c4
md"""
## Distance transforms + deep learning
One area of recent interest in deep learning has been distance transform-based loss functions to directly reduce Hausdorff loss and other boundary-specific loss metrics [D. Karimi and S. E. Salcudean](http://arxiv.org/abs/1904.10030).
This package can be utilized directly in various training loops, along with [Losers.jl](https://github.com/Dale-Black/Losers.jl) to train segmentation networks, like the common UNet architecture, to directly reduce Hausdorff distance using the `hausdorff()` loss function.
In practice, it is common to combine `hausdorff` with the `dice` loss function to help stabilize during training.
Below we will show how one might set this up, without actually training on anything.
"""

# ╔═╡ 18003aaf-fadb-4c13-8d86-8efb47135cde
ground_truth = rand([0, 1], 5, 5, 3)

# ╔═╡ a23b92c1-ceb9-4943-8916-42fd3e43cc29
prediction = copy(ground_truth)

# ╔═╡ ded6a37c-bfde-42cf-b075-87289694d6fd
md"""
Given some fake three dimensional array, lets apply a distance transform to this array for use in the `hausdorff` loss function
"""

# ╔═╡ ac4e2b4d-c6ab-4342-bb2a-afd05078ef04
tfm = Felzenszwalb()

# ╔═╡ 135f2c9a-a3d9-4477-b5af-3c95154b48f2
ground_truth_dtm = transform(boolean_indicator(ground_truth), tfm)

# ╔═╡ 2daa28b5-ddf1-4c2a-bf1e-e273a7f8fad0
prediction_dtm = transform(boolean_indicator(prediction), tfm)

# ╔═╡ bcc150cf-ab58-4f69-90bc-db72660805d7
hausdorff(ground_truth, prediction, ground_truth_dtm, prediction_dtm)

# ╔═╡ 3e59daab-e71c-46df-896b-764dce9f2b42
md"""
As one would expect, the difference between two identical arrays is zero. Below we see that two randomly generated arrays should have a non-zero Hausdorff distance. The `hausdorff` loss function approximates this Hausdorff distance, by utilizing distance transform maps of the original arrays.
"""

# ╔═╡ ea235007-1fab-4ed4-bf60-b77e04d38460
begin
	prediction2 = rand([0, 1], 5, 5, 3)
	ground_truth2 = rand([0, 1], 5, 5, 3)

	prediction2_dtm = transform(boolean_indicator(prediction2), tfm)
	ground_truth2_dtm = transform(boolean_indicator(ground_truth2), tfm)
end

# ╔═╡ ba878e0d-1102-488a-a9e5-b2e12ab4358d
hausdorff(prediction2, ground_truth2, prediction2_dtm, ground_truth2_dtm)

# ╔═╡ f9ca3701-f897-4c93-b694-41a6e5134bfb
md"""
## GPU considerations
When using distance transform-based loss functions for deep learning, one must consider GPUs for many real-world deep learning tasks. The benefit of DistanceTransforms.jl is that GPU compatible distance transform algorithms are ready to use, out-of-the-box. 
We will look at how one would go about this, without actually calling a CUDA array from the CUDA.jl library.
"""

# ╔═╡ 34360230-7961-4b4a-9b85-8969d8c17d59
md"""
If we were to be running an actual training loop it might look something like this...
```julia
# Begin training loop
for epoch in 1:max_epochs
	step = 0
    α = some_constant .+ 0.01
	@info epoch
	
	# Loop through training data
	for (xs, ys) in train_loader
		step += 1
		@info step
		# Send data to GPU
		xs, ys = xs |> gpu, ys |> gpu		
		gs = Flux.gradient(ps) do
			ŷs = model(xs)
			# Apply distance transform using GPU compatible `Felzenszwalb`
			# Data will usually be 4D or 5D [x, y, (z), channel, batch]
			ys_dtm = CuArray{Float32}(undef, size(ys))
			ŷs_dtm = CuArray{Float32}(undef, size(ŷs))
			for b in size(ys, 5)
				for c in size(ys, 4)
					for z in size(ys, 3)
						bool_arr_gt = boolean_indicator(ys[:, :, z, c, b])
						bool_arr_pred = boolean_indicator(ŷs[:, :, z, c, b])
						tfm = Felzenszwalb()
						ys_dtm[:, :, z, c, b] = transform!(bool_arr_gt, tfm)
						ŷs_dtm[:, :, z, c, b] = transform!(bool_arr_pred, tfm)
					end
				end
			end
			hd_loss = hausdorff(ŷs, ys, ŷs_dtm, ys_dtm)
			dice_loss = dice(ŷs, ys)
			loss = α*dice_loss + (1-α)*hausdorff_loss
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
# ╟─f134f1c2-503d-41cf-ab9a-eb8c8fca37c4
# ╠═18003aaf-fadb-4c13-8d86-8efb47135cde
# ╠═a23b92c1-ceb9-4943-8916-42fd3e43cc29
# ╟─ded6a37c-bfde-42cf-b075-87289694d6fd
# ╠═ac4e2b4d-c6ab-4342-bb2a-afd05078ef04
# ╠═135f2c9a-a3d9-4477-b5af-3c95154b48f2
# ╠═2daa28b5-ddf1-4c2a-bf1e-e273a7f8fad0
# ╠═bcc150cf-ab58-4f69-90bc-db72660805d7
# ╟─3e59daab-e71c-46df-896b-764dce9f2b42
# ╠═ea235007-1fab-4ed4-bf60-b77e04d38460
# ╠═ba878e0d-1102-488a-a9e5-b2e12ab4358d
# ╟─f9ca3701-f897-4c93-b694-41a6e5134bfb
# ╟─34360230-7961-4b4a-9b85-8969d8c17d59
