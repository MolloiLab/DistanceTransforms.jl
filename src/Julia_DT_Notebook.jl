### A Pluto.jl notebook ###
# v0.16.1

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : missing
        el
    end
end

# ╔═╡ 123c3f17-4a4a-478d-ae0e-5ec7f9d4ad44
begin
    let
        using Pkg
		Pkg.activate(mktempdir())
        Pkg.Registry.update()
        Pkg.add("PlutoUI")
        Pkg.add("Tar")
        Pkg.add("MLDataPattern")
        Pkg.add("Glob")
        Pkg.add("NIfTI")
        Pkg.add("DataAugmentation")
        Pkg.add("CairoMakie")
        Pkg.add("ImageCore")
        Pkg.add("DataLoaders")
        # Pkg.add("CUDA")
        Pkg.add("FastAI")
        Pkg.add("StaticArrays")
    end

    using PlutoUI
    using Tar
    using MLDataPattern
    using Glob
    using NIfTI
    using DataAugmentation
    using DataAugmentation: OneHot, Image
    using CairoMakie
    using ImageCore
    using DataLoaders
    # using CUDA
    using FastAI
    using StaticArrays
end

# ╔═╡ d781f284-16c1-4505-9f5a-dbd90c0334d2
TableOfContents()

# ╔═╡ 6e2a1536-e659-4366-93de-ac7d2e7a27a4
md"""
## Load data
Part of the [Medical Decathlon Dataset](http://medicaldecathlon.com/)
"""

# ╔═╡ 855407b1-b8ce-465f-8531-1af5d014abe9
data_dir = raw"/Users/daleblack/Google Drive/Datasets/Task02_Heart"

# ╔═╡ 4cefe3ca-d7f1-4b2a-b63c-e39d734d2514
function loadfn_label(p)
    a = NIfTI.niread(string(p)).raw
    convert_a = convert(Array{UInt8}, a)
    convert_a = convert_a .+ 1
    return convert_a
end

# ╔═╡ 838a17df-7479-42c2-a2c5-ce626a4016fe
function loadfn_image(p)
    a = NIfTI.niread(string(p)).raw
    convert_a = convert(Array{Float32}, a)
    convert_a = convert_a / max(convert_a...)
    return convert_a
end

# ╔═╡ 11d5a9b4-fdc3-48e4-a6ad-f7dec5fabb8b
begin
    niftidata_image(dir) = mapobs(loadfn_image, Glob.glob("*.nii*", dir))
    niftidata_label(dir) =  mapobs(loadfn_label, Glob.glob("*.nii*", dir))
    data = (
        niftidata_image(joinpath(data_dir, "imagesTr")),
        niftidata_label(joinpath(data_dir, "labelsTr")),
    )
end

# ╔═╡ 778cc0f9-127c-4d4b-a7de-906dfcc29cae
train_files, val_files = MLDataPattern.splitobs(data, 0.8)

# ╔═╡ 349d843a-4a5f-44d7-9371-38c140b9972d
md"""
## Create learning method
"""

# ╔═╡ 019e666e-e2e4-4e0b-a225-b346c7c70939
struct ImageSegmentationSimple <: DLPipelines.LearningMethod
    imagesize
end

# ╔═╡ f7274fa9-8231-44fd-8d00-1c7ab7fc855c
image_size = (112, 112, 96)

# ╔═╡ 9ac18928-9fe4-46ed-ab9c-916791739157
method = ImageSegmentationSimple(image_size)

# ╔═╡ edf2b37a-2775-44c0-8d2e-5e2350b454c4
md"""
### Set up `encode` pipelines
"""

# ╔═╡ 7cf0cc6b-8ff9-4198-9646-9d0787e1013d
begin
  function DLPipelines.encode(
          method::ImageSegmentationSimple,
          context::Training,
          (image, target)::Union{Tuple, NamedTuple}
          )

      tfm_proj = RandomResizeCrop(method.imagesize)
      tfm_im = DataAugmentation.compose(
			ImageToTensor(),
			NormalizeIntensity()
          )
      tfm_mask = OneHot()

      items = Image(Gray.(image)), MaskMulti(target)
      item_im, item_mask = apply(tfm_proj, (items))

      return itemdata(apply(tfm_im, item_im)), itemdata(apply(tfm_mask, item_mask))
  end

  function DLPipelines.encode(
          method::ImageSegmentationSimple,
          context::Validation,
          (image, target)::Union{Tuple, NamedTuple}
          )

      tfm_proj = CenterResizeCrop(method.imagesize)
      tfm_im = DataAugmentation.compose(
          ImageToTensor(),
          NormalizeIntensity()
          )
      tfm_mask = OneHot()

      items = Image(Gray.(image)), MaskMulti(target)
      item_im, item_mask = apply(tfm_proj, (items))

      return itemdata(apply(tfm_im, item_im)), itemdata(apply(tfm_mask, item_mask))
  end
end

# ╔═╡ f2a92ff7-0f94-44d9-aba7-4b7db9fa4a56
begin
	methoddata_train = DLPipelines.MethodDataset(train_files, method, Training())
	methoddata_valid = DLPipelines.MethodDataset(val_files, method, Validation())
end

# ╔═╡ b8b18728-cb5a-445a-809c-986e3965aad3
let
    x, y = MLDataPattern.getobs(methoddata_valid, 1)
    @assert size(x) == (image_size..., 1)
    @assert size(y) == (image_size..., 2)
end

# ╔═╡ bfe051e5-66a3-4b4b-b446-48195d8f3868
md"""
## Visualize
"""

# ╔═╡ 4b7bff28-ec60-4a9d-8b38-648ab871ed16
begin
    x, y = MLDataPattern.getobs(methoddata_valid, 3)
end;

# ╔═╡ 6d1e9ca8-da31-4143-b46f-44b459a2cfc3
size(x)

# ╔═╡ 792a709c-6740-4fc0-b2a3-43ad2ae8035a
size(y)

# ╔═╡ 08be9516-a44c-4fe5-ac46-f586600d586a
@bind b PlutoUI.Slider(1:size(x)[3], default=50, show_value=true)

# ╔═╡ e2670a05-2c09-4bd2-b40c-0cc46fef2344
heatmap(x[:, :, b, 1], colormap=:grays)

# ╔═╡ 427ff0c3-d773-4099-a483-bb8f7d4b8ba1
heatmap(y[:, :, b, 2], colormap=:grays)

# ╔═╡ 616aa780-cbcf-4bf2-b6c5-a984f2530482
md"""
## Dataloader
"""

# ╔═╡ bda7309e-ae97-4ac0-96b6-36a776e9215e
begin
    train_loader = DataLoaders.DataLoader(methoddata_train, 2)
    val_loader = DataLoaders.DataLoader(methoddata_valid, 4)
end

# ╔═╡ 3da0e60e-0196-4538-ab08-49f21b46679a
train_loader

# ╔═╡ e70816b8-4597-4a54-b4f7-735880df6132
val_loader

# ╔═╡ ff962aea-2501-42f5-90bc-72f29deb42af
md"""
## Model
"""

# ╔═╡ 5987b24a-a1da-4d13-89d1-c854de2c3ba0
begin
    # 3D layer utilities
    conv = (stride, in, out) -> Flux.Conv((3, 3, 3), in=>out, stride=stride, pad=Flux.SamePad())
    tran = (stride, in, out) -> Flux.ConvTranspose((3, 3, 3), in=>out, stride=stride, pad=Flux.SamePad())

    conv1 = (in, out) -> Flux.Chain(conv(1, in, out), Flux.BatchNorm(out), x -> leakyrelu.(x))
    conv2 = (in, out) -> Flux.Chain(conv(2, in, out), Flux.BatchNorm(out), x -> leakyrelu.(x))
    tran2 = (in, out) -> Flux.Chain(tran(2, in, out), Flux.BatchNorm(out), x -> leakyrelu.(x))
end

# ╔═╡ f451e763-d7d0-4de7-a1f3-76776a194022
begin
    function unet3D(in_chs, lbl_chs)
        # Contracting layers
        l1 = Flux.Chain(conv1(in_chs, 4))
        l2 = Flux.Chain(l1, conv1(4, 4), conv2(4, 16))
        l3 = Flux.Chain(l2, conv1(16, 16), conv2(16, 32))
        l4 = Flux.Chain(l3, conv1(32, 32), conv2(32, 64))
        l5 = Flux.Chain(l4, conv1(64, 64), conv2(64, 128))

        # Expanding layers
        l6 = Flux.Chain(l5, tran2(128, 64), conv1(64, 64))
        l7 = Flux.Chain(Flux.Parallel(+, l6, l4), tran2(64, 32), conv1(32, 32))
        l8 = Flux.Chain(Flux.Parallel(+, l7, l3), tran2(32, 16), conv1(16, 16))
        l9 = Flux.Chain(Flux.Parallel(+, l8, l2), tran2(16, 4), conv1(4, 4))
        l10 = Flux.Chain(l9, conv1(4, lbl_chs))
    end
end

# ╔═╡ ad342428-920a-4414-a4cc-cab281a681dc
model = unet3D(1, 2);

# ╔═╡ d073cbac-18c8-4682-a508-307a619e84fc
md"""
## Helper functions
"""

# ╔═╡ 5726b8bd-e1e7-44eb-8872-a4a8d26be0f9
function dice_metric(ŷ, y)
    dice = 2 * sum(ŷ .& y) / (sum(ŷ) + sum(y))
    return dice
end

# ╔═╡ 7b974fc5-7fc3-45e7-bf6c-48ea4f8eff16
function as_discrete(array, logit_threshold)
    array = array .>= logit_threshold
    return array
end

# ╔═╡ b87d764e-dd31-48b7-bcf1-0538ef5c5e6a
md"""
## Loss functions
"""

# ╔═╡ 6dee15db-5cf9-4bfb-ab42-48f1c2777c04
function dice_loss(ŷ, y)
    ϵ = 1e-5
    return loss = 1 - ((2 * sum(ŷ .* y) + ϵ) / (sum(ŷ .* ŷ) + sum(y .* y) + ϵ))
end

# ╔═╡ bdc1ca32-4847-440f-8a44-98bf7e822803
function hd_loss(ŷ, y, ŷ_dtm, y_dtm)
    M = (ŷ .- y) .^ 2 .* (ŷ_dtm .^ 2 .+ y_dtm .^ 2)
    return loss = mean(M)
end

# ╔═╡ aafb8a3f-6e6c-4475-92db-550eb6999741
md"""
## Training
"""

# ╔═╡ 3d0f1399-54c3-45b6-840a-8e287513bbe7
ps = Flux.params(model);

# ╔═╡ 06061dc5-eae9-47cb-99aa-d521c5cd37dd
loss_function = dice_loss

# ╔═╡ 98a49f59-dfba-46f0-851a-4c83f0b53183
begin
	max_epochs = 30
	val_interval = 2
	epoch_loss_values = []
	val_epoch_loss_values = []
	metric_values = []
	metric_count = 0
	metric_sum = 0.0
	new_loss = 0
	best_metric_epoch = -1
	best_metric = -1
end

# ╔═╡ f8634d1c-0e67-4f6b-ab29-8ae1962a95e1
begin
	iter_num = 0
	alpha = 1.00
end

# ╔═╡ ad1bf410-f187-4811-b752-09252e1b3a32
for epoch in range(max_epochs)
    epoch_loss = 0
    step = 0
    println("Epoch: ", epoch)

    # Loop through training data
    for (xs, ys) in train_loader
        step += 1
        println("Train Step: ", step)

        xs, ys = xs |> gpu, ys |> gpu
        ŷs = model(xs)

        outputs_soft = softmax(ŷs; dims = 2)
        loss_seg_dice = dice_loss(outputs_soft[:, 2, :, :, :], ys== 1)

        gt_dtm = compute_dtm(ys)
        seg_dtm = compute_dtm(outputs_soft[:, 2, :, :, :] .> 0.5)
        
        gs = Flux.gradient(ps) do
            loss_hd = hd_loss(outputs_soft, seg_dtm, gt_dtm)
            loss = alpha*(loss_seg_dice) + (1 - alpha) * loss_hd
            return loss
        end
        Flux.update!(optimizer, ps, gs)
        local ŷs = model(xs)
        local loss = loss_function(ŷs[:, :, :, 2, :], ys[:, :, :, 2, :])
        epoch_loss += loss_function(ŷs[:, :, :, 2, :], ys[:, :, :, 2, :])

    end
    epoch_loss = (epoch_loss / step)
    push!(epoch_loss_values, epoch_loss)

    println("epoch $(epoch_num + 1) average loss: (epoch_loss:.4f)")
    alpha -= 0.001
    if alpha <= 0.001
        alpha = 0.001
    end
    # Loop through validation data
    if (epoch + 1) % val_interval == 0
        val_step = 0
        val_epoch_loss = 0
        metric_step = 0
        dice = 0
        for (val_xs, val_ys) in val_loader
            val_step += 1
            println("val step: ", val_step)

            val_xs, val_ys = val_xs |> gpu, val_ys |> gpu
            local val_ŷs = model(val_xs)
            local val_ŷs = model(val_xs)

            local outputs_soft = softmax(val_ŷs; dims = 2)
            local loss_seg_dice = dice_loss(outputs_soft[:, 2, :, :, :], ys== 1)

            local gt_dtm = compute_dtm(val_ys)
            local seg_dtm = compute_dtm(outputs_soft[:, 2, :, :, :] .> 0.5)
        
            local loss_hd = hd_loss(outputs_soft, seg_dtm, gt_dtm)
            local val_loss = alpha*(loss_seg_dice) + (1 - alpha) * loss_hd

            val_epoch_loss += val_loss

            val_ŷs, val_ys = val_ŷs |> cpu, val_ys |> cpu
            val_ŷs, val_ys = as_discrete(val_ŷs, 0.5), as_discrete(val_ys, 0.5)
            metric_step += 1
            metric = dice_metric(val_ŷs[:, :, :, 2, :], val_ys[:, :, :, 2, :])
            dice += metric
        end

        val_epoch_loss = (val_epoch_loss / val_step)
        push!(val_epoch_loss_values, val_epoch_loss)

        dice = dice / metric_step
        push!(dice_metric_values, dice)
    end
end

# ╔═╡ 36501d5d-d29a-4281-9e75-cebc97bb68cd
begin
	base_lr = 0.001
	max_iterations = 20000
	optimizer = Flux.ADAM(0.001)
end

# ╔═╡ 3a13a9a9-380a-4613-b137-99fdef8cb92f
optimizer = Flux.ADAM(0.001)

# ╔═╡ Cell order:
# ╠═123c3f17-4a4a-478d-ae0e-5ec7f9d4ad44
# ╠═d781f284-16c1-4505-9f5a-dbd90c0334d2
# ╟─6e2a1536-e659-4366-93de-ac7d2e7a27a4
# ╠═855407b1-b8ce-465f-8531-1af5d014abe9
# ╠═4cefe3ca-d7f1-4b2a-b63c-e39d734d2514
# ╠═838a17df-7479-42c2-a2c5-ce626a4016fe
# ╠═11d5a9b4-fdc3-48e4-a6ad-f7dec5fabb8b
# ╠═778cc0f9-127c-4d4b-a7de-906dfcc29cae
# ╟─349d843a-4a5f-44d7-9371-38c140b9972d
# ╠═019e666e-e2e4-4e0b-a225-b346c7c70939
# ╠═f7274fa9-8231-44fd-8d00-1c7ab7fc855c
# ╠═9ac18928-9fe4-46ed-ab9c-916791739157
# ╟─edf2b37a-2775-44c0-8d2e-5e2350b454c4
# ╠═7cf0cc6b-8ff9-4198-9646-9d0787e1013d
# ╠═f2a92ff7-0f94-44d9-aba7-4b7db9fa4a56
# ╠═b8b18728-cb5a-445a-809c-986e3965aad3
# ╟─bfe051e5-66a3-4b4b-b446-48195d8f3868
# ╠═4b7bff28-ec60-4a9d-8b38-648ab871ed16
# ╠═6d1e9ca8-da31-4143-b46f-44b459a2cfc3
# ╠═792a709c-6740-4fc0-b2a3-43ad2ae8035a
# ╟─08be9516-a44c-4fe5-ac46-f586600d586a
# ╠═e2670a05-2c09-4bd2-b40c-0cc46fef2344
# ╠═427ff0c3-d773-4099-a483-bb8f7d4b8ba1
# ╟─616aa780-cbcf-4bf2-b6c5-a984f2530482
# ╠═bda7309e-ae97-4ac0-96b6-36a776e9215e
# ╠═3da0e60e-0196-4538-ab08-49f21b46679a
# ╠═e70816b8-4597-4a54-b4f7-735880df6132
# ╟─ff962aea-2501-42f5-90bc-72f29deb42af
# ╠═5987b24a-a1da-4d13-89d1-c854de2c3ba0
# ╠═f451e763-d7d0-4de7-a1f3-76776a194022
# ╠═ad342428-920a-4414-a4cc-cab281a681dc
# ╟─d073cbac-18c8-4682-a508-307a619e84fc
# ╠═5726b8bd-e1e7-44eb-8872-a4a8d26be0f9
# ╠═7b974fc5-7fc3-45e7-bf6c-48ea4f8eff16
# ╟─b87d764e-dd31-48b7-bcf1-0538ef5c5e6a
# ╠═6dee15db-5cf9-4bfb-ab42-48f1c2777c04
# ╠═bdc1ca32-4847-440f-8a44-98bf7e822803
# ╟─aafb8a3f-6e6c-4475-92db-550eb6999741
# ╠═3d0f1399-54c3-45b6-840a-8e287513bbe7
# ╠═06061dc5-eae9-47cb-99aa-d521c5cd37dd
# ╠═3a13a9a9-380a-4613-b137-99fdef8cb92f
# ╠═36501d5d-d29a-4281-9e75-cebc97bb68cd
# ╠═98a49f59-dfba-46f0-851a-4c83f0b53183
# ╠═f8634d1c-0e67-4f6b-ab29-8ae1962a95e1
# ╠═ad1bf410-f187-4811-b752-09252e1b3a32
