### A Pluto.jl notebook ###
# v0.19.11

using Markdown
using InteractiveUtils

# ╔═╡ 96a0c69e-7462-4b7f-8e27-a0eba5c43ea1
# ╠═╡ show_logs = false
begin
	using Pkg
	Pkg.activate(".")
	using Revise
	using PlutoUI
	using DistanceTransforms
	using Losers
end

# ╔═╡ d7df155f-dc09-4c90-99b6-2997cddfbe5c
md"""
## Import packages
"""

# ╔═╡ e75cddb3-5fd7-4b63-9091-02d30d5e303b
TableOfContents()

# ╔═╡ 8090db7e-ef92-4960-87e3-fc4be97b1dba
md"""
## Distance transforms + deep learning
One area of recent interest in deep learning has been distance transform-based loss functions to directly reduce Hausdorff loss and other boundary-specific loss metrics [D. Karimi and S. E. Salcudean](http://arxiv.org/abs/1904.10030).

This package can be utilized directly in various training loops, along with [Losers.jl](https://github.com/Dale-Black/Losers.jl) to train segmentation networks, like the common UNet architecture, to directly reduce Hausdorff distance using the `hausdorff()` loss function.

In practice, it is common to combine `hausdorff` with the `dice` loss function to help stabilize during training.

Below we will show how one might set this up, without actually training on anything.
"""

# ╔═╡ 17970c09-b3c6-4c3b-84b2-1196147df97b
y = rand([0, 1], 5, 5, 3)

# ╔═╡ d4400da1-98ab-4dc7-a527-ab3064905612
ŷ = copy(y)

# ╔═╡ f472ca2d-babc-4c69-b1b3-0653f8618ded
md"""
Given some fake three dimensional array, lets apply a distance transform to this array for use in the `hausdorff` loss function
"""

# ╔═╡ 47296dd7-7318-4501-983b-8c89e79b4899
tfm = SquaredEuclidean()

# ╔═╡ 6e542e89-995a-4261-9c27-0ddf5a2037f3
y_dtm = transform(boolean_indicator(y), tfm)

# ╔═╡ f6b1b811-9c09-4ee2-8d9f-f20f21ac65f4
ŷ_dtm = transform(boolean_indicator(ŷ), tfm)

# ╔═╡ 6084377d-1178-47cf-91d1-becab14d0440
hausdorff(ŷ, y, ŷ_dtm, y_dtm)

# ╔═╡ 3fa61868-3be8-4c37-acef-0c630305156b
md"""
As one would expect, the difference between two identical arrays is zero. Below we see that two randomly generated arrays should have a non-zero Hausdorff distance. The `hausdorff` loss function approximates this.
"""

# ╔═╡ 330cc292-0b97-433e-9e0b-3bc3d1bea9a3
begin
	y2 = rand([0, 1], 5, 5, 3)
	ŷ2 = rand([0, 1], 5, 5, 3)

	y2_dtm = euclidean(y2)
	ŷ2_dtm = euclidean(ŷ2)
end

# ╔═╡ 4b21785f-e69b-4817-89e7-ff5fb58c5575
hausdorff(ŷ2, y2, ŷ2_dtm, y2_dtm)

# ╔═╡ d2392f57-f27b-42e4-9855-02077d8c2644
md"""
## GPU considerations
When using distance transform-based loss functions for deep learning, one must consider GPUs for many real-world deep learning tasks. The benefit of DistanceTransforms.jl is that GPU compatible distance transform algorithms are ready to use, out-of-the-box. 

We will look at how one would go about this, without actually calling a CUDA array from the CUDA.jl library.
"""

# ╔═╡ a82a987c-f13b-4233-8c31-657565d2641a
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

			# Apply distance transform using GPU compatible `SquaredEuclidean`
			# Data will usually be 4D or 5D [x, y, (z), channel, batch]

			ys_dtm = CuArray{Float32}(undef, size(ys))
			ŷs_dtm = CuArray{Float32}(undef, size(ŷs))
			for b in size(ys, 5)
				for c in size(ys, 4)
					for z in size(ys, 3)
						bool_arr_gt = boolean_indicator(ys[:, :, z, c, b])
						bool_arr_pred = boolean_indicator(ŷs[:, :, z, c, b])
						tfm = SquaredEuclidean()
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
# ╠═96a0c69e-7462-4b7f-8e27-a0eba5c43ea1
# ╟─d7df155f-dc09-4c90-99b6-2997cddfbe5c
# ╠═e75cddb3-5fd7-4b63-9091-02d30d5e303b
# ╟─8090db7e-ef92-4960-87e3-fc4be97b1dba
# ╠═17970c09-b3c6-4c3b-84b2-1196147df97b
# ╠═d4400da1-98ab-4dc7-a527-ab3064905612
# ╟─f472ca2d-babc-4c69-b1b3-0653f8618ded
# ╠═47296dd7-7318-4501-983b-8c89e79b4899
# ╠═6e542e89-995a-4261-9c27-0ddf5a2037f3
# ╠═f6b1b811-9c09-4ee2-8d9f-f20f21ac65f4
# ╠═6084377d-1178-47cf-91d1-becab14d0440
# ╟─3fa61868-3be8-4c37-acef-0c630305156b
# ╠═330cc292-0b97-433e-9e0b-3bc3d1bea9a3
# ╠═4b21785f-e69b-4817-89e7-ff5fb58c5575
# ╟─d2392f57-f27b-42e4-9855-02077d8c2644
# ╟─a82a987c-f13b-4233-8c31-657565d2641a
