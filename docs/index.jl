### A Pluto.jl notebook ###
# v0.19.26

#> [frontmatter]
#> title = "DistanceTransforms.jl"
#> sidebar = "false"

using Markdown
using InteractiveUtils

# ╔═╡ a73b3508-903c-4863-ad05-6966cb055102
using HypertextLiteral

# ╔═╡ b07dcdc1-495e-45f7-b298-dcb61172c84b
html"""
<head>
	<link rel="preconnect" href="https://fonts.googleapis.com">
	<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
	<link href="https://fonts.googleapis.com/css2?family=Alegreya+Sans:ital,wght@0,400;0,700;1,400&family=Vollkorn:ital,wght@0,400;0,700;1,400;1,700&display=swap" rel="stylesheet">
	<link href="https://cdn.jsdelivr.net/npm/daisyui@3.7.4/dist/full.css" rel="stylesheet" type="text/css" />
	<script src="https://cdn.tailwindcss.com"></script>
</head>

<div data-theme="pastel" class="bg-transparent dark:bg-[#1f1f1f]">
	<div id="ComputerVisionMetricsHeader" class="flex justify-center items-center">
		<div class="card card-bordered border-accent text-center w-full">
			<div class="card-body flex flex-col justify-center items-center">
				<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 128 128" style="enable-background:new 0 0 128 128" width="150px" height="150px" xml:space="preserve">
					<path style="fill:#303030" d="M128 122.674H28.697L0 104.398V5.235h99.305L128 23.511v99.163zm-91.264-8.028h73.413l-17.851-10.043.132-.206H36.736v10.249zm-23.038-10.248 14.999 8.234v-8.234H13.698zm85.607-4.861 20.656 13.094V31.539H99.305v67.998zM36.736 96.37h54.53V31.539h-54.53V96.37zm-28.697 0h20.658V29.647L8.039 16.551V96.37zm91.266-72.859h14.919l-14.919-8.949v8.949zm-65.745 0h57.706V13.262H17.851l15.841 10.043-.132.206z"/>
				</svg>
				<h1 class="card-title text-5xl font-serif">DistanceTransforms.jl</h1>
				<p class="card-text text-lg font-serif">Fast Distance Transforms in Julia</p>
			</div>
		</div>
	</div>
</div>
"""

# ╔═╡ b8e3330c-a384-4091-98db-82646f080f40
function ArticleTile(article)
	@htl("""
	<a href="$(article.path)" data-theme="pastel" class="card card-bordered bg-transparent border-primary hover:shadow-lg">
		<div class="card-body">
			<h2 class="card-title">$(article.title)</h2>
			<p>Click to open the notebook.</p>
		</div>
		<figure>
			<img src="$(article.image_url)" alt="$(article.title)">
		</figure>
	</a>
	""")
end;

# ╔═╡ f6f7f39c-d9b3-4f9a-9ba3-4833e2f5419f
struct Article
	title::String
	path::String
	image_url::String
end

# ╔═╡ c230bcc0-f1bf-4530-a5ec-d23b76770b01
article_list = Article[
	Article("Getting Started", "01_getting_started.jl", "https://images.pexels.com/photos/45718/pexels-photo-45718.jpeg?auto=compress&cs=tinysrgb&w=800"),
	Article("Multi-threading", "02_multi_threading.jl", "https://images.pexels.com/photos/3091200/pexels-photo-3091200.jpeg?auto=compress&cs=tinysrgb&w=800"),
	Article("GPU", "03_gpu.jl", "https://images.pexels.com/photos/8622911/pexels-photo-8622911.jpeg?auto=compress&cs=tinysrgb&w=800"),
	Article("Deep Learning Usage", "04_loss_functions.jl", "https://images.pexels.com/photos/8386440/pexels-photo-8386440.jpeg?auto=compress&cs=tinysrgb&w=800"),
	Article("Benchmarks", "05_benchmarks.jl", "https://images.pexels.com/photos/39396/hourglass-time-hours-sand-39396.jpeg?auto=compress&cs=tinysrgb&w=800"),
	Article("API", "06_api.jl", "https://images.unsplash.com/photo-1503789146722-cf137a3c0fea?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=2562&q=80"),
];

# ╔═╡ e62bfcc3-4745-43c2-9bae-d356d2148beb
@htl("""
<div class="grid grid-cols-2 gap-4">
	$([ArticleTile(article) for article in article_list])
</div>
""")

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
HypertextLiteral = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"

[compat]
HypertextLiteral = "~0.9.4"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.9.3"
manifest_format = "2.0"
project_hash = "fc304fba520d81fb78ea25b98f5762b4591b1182"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "c47c5fa4c5308f27ccaac35504858d8914e102f9"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.4"

[[deps.Tricks]]
git-tree-sha1 = "aadb748be58b492045b4f56166b5188aa63ce549"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.7"
"""

# ╔═╡ Cell order:
# ╟─b07dcdc1-495e-45f7-b298-dcb61172c84b
# ╟─e62bfcc3-4745-43c2-9bae-d356d2148beb
# ╟─c230bcc0-f1bf-4530-a5ec-d23b76770b01
# ╟─b8e3330c-a384-4091-98db-82646f080f40
# ╟─f6f7f39c-d9b3-4f9a-9ba3-4833e2f5419f
# ╟─a73b3508-903c-4863-ad05-6966cb055102
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
