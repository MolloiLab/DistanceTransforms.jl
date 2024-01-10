### A Pluto.jl notebook ###
# v0.19.36

#> [frontmatter]
#> title = "DistanceTransforms.jl"
#> sidebar = "false"

using Markdown
using InteractiveUtils

# ╔═╡ 8a70f0c0-c70e-4022-8a06-ea8fec31a7c7
# ╠═╡ show_logs = false
begin
	using Pkg
	Pkg.activate(joinpath(pwd(), "docs"))
	Pkg.instantiate()

	using HTMLStrings: to_html, head, link, script, divv, h1, img, p, span, a, figure, hr
	using PlutoUI
end

# ╔═╡ 5e70dc3d-85b6-4149-9962-70215d3c7f1e
md"""
## Tutorials
"""

# ╔═╡ 8b5d431a-f91c-422b-86cf-3ace891e742f
md"""
## Python Usage
"""

# ╔═╡ b3349485-1a5f-4da6-be79-965fec5fbc13
md"""
## API
"""

# ╔═╡ d0ec67d4-26bd-4960-b4d9-75d9c0b740b0
to_html(hr())

# ╔═╡ 4195fc74-92a0-4ad4-8c36-9d652ac36fd5
TableOfContents()

# ╔═╡ febd9959-70d6-409d-b216-762ec0b751dc
data_theme = "cupcake";

# ╔═╡ 0af240e8-ccf7-4bf8-b21c-08e2972d2459
function index_title_card(title::String, subtitle::String, image_url::String; data_theme::String = "pastel", border_color::String = "primary")
	return to_html(
	    divv(
	        head(
				link(:href => "https://cdn.jsdelivr.net/npm/daisyui@3.7.4/dist/full.css", :rel => "stylesheet", :type => "text/css"),
	            script(:src => "https://cdn.tailwindcss.com")
	        ),
			divv(:data_theme => "$data_theme", :class => "card card-bordered flex justify-center items-center border-$border_color text-center w-full dark:text-[#e6e6e6]",
				divv(:class => "card-body flex flex-col justify-center items-center",
					img(:src => "$image_url", :class => "h-24 w-24 md:h-52 md:w-52 rounded-md", :alt => "$title Logo"),
					divv(:class => "text-5xl font-bold bg-gradient-to-r from-accent to-primary inline-block text-transparent bg-clip-text py-10", "$title"),
					p(:class => "card-text text-md font-serif", "$subtitle"
					)
				)
			)
	    )
	)
end;

# ╔═╡ 18be66cd-28c4-4137-b16b-428430b7f996
index_title_card(
	"DistanceTransforms.jl",
	"Fast Distance Transforms in Julia",
	"https://img.freepik.com/free-vector/global-communication-background-business-network-vector-design_53876-151122.jpg";
	data_theme = data_theme
)

# ╔═╡ 09b3d1e1-b44f-461d-abaf-2e5a1f6f6cc2
struct Article
	title::String
	path::String
	image_url::String
end

# ╔═╡ 0d222d9d-7576-40a7-acb4-b0e27f9a9ef8
article_list_tutorials = Article[
	Article("Getting Started", "docs/01_getting_started.jl", "https://img.freepik.com/free-photo/futuristic-spaceship-takes-off-into-purple-galaxy-fueled-by-innovation-generated-by-ai_24640-100023.jpg"),
	Article("Advanced Usage", "docs/02_advanced_usage.jl", "https://images.pexels.com/photos/3091200/pexels-photo-3091200.jpeg?auto=compress&cs=tinysrgb&w=800")
];

# ╔═╡ e0ef1b4d-0076-410c-b325-32a7a274f487
article_list_python = Article[
	Article("Python", "https://colab.research.google.com/drive/1-CDqQgrBHoxNqs2IbMebMRxsp0m21jSa?usp=sharing", "https://img.freepik.com/free-vector/code-typing-concept-illustration_114360-3581.jpg?size=626&ext=jpg&ga=GA1.1.1694943658.1700350224&semt=sph"),
];

# ╔═╡ 8d11bae1-d232-4c03-981a-cd9f1db24e6b
article_list_api = Article[
	Article("API", "docs/99_api.jl", "https://img.freepik.com/free-photo/modern-technology-workshop-creativity-innovation-communication-development-generated-by-ai_188544-24548.jpg"),
];

# ╔═╡ 3bcd40f0-e8dd-48bd-8249-51b89f5766de
function article_card(article::Article, color::String; data_theme = "pastel")
    a(:href => article.path, :class => "w-1/2 p-2",
		divv(:data_theme => "$data_theme", :class => "card card-bordered border-$color text-center dark:text-[#e6e6e6]",
			divv(:class => "card-body justify-center items-center",
				p(:class => "card-title", article.title),
				p("Click to open the notebook")
			),
			figure(
				img(:class =>"w-full h-40", :src => article.image_url, :alt => article.title)
			)
        )
    )
end;

# ╔═╡ ace65d98-fe6b-45b7-990d-df8f59f6a5b3
to_html(
    divv(:class => "flex flex-wrap justify-center items-start",
        [article_card(article, "secondary"; data_theme = data_theme) for article in article_list_tutorials]...
    )
)

# ╔═╡ 75f02433-7ef3-45f4-ad2a-d610923824af
to_html(
    divv(:class => "flex flex-wrap justify-center items-start",
        [article_card(article, "secondary"; data_theme = data_theme) for article in article_list_python]...
    )
)

# ╔═╡ 0c5f21fe-80fc-4810-b4a8-bb1c290c5d81
to_html(
    divv(:class => "flex flex-wrap justify-center items-start",
        [article_card(article, "secondary"; data_theme = data_theme) for article in article_list_api]...
    )
)

# ╔═╡ Cell order:
# ╟─18be66cd-28c4-4137-b16b-428430b7f996
# ╟─5e70dc3d-85b6-4149-9962-70215d3c7f1e
# ╟─ace65d98-fe6b-45b7-990d-df8f59f6a5b3
# ╟─8b5d431a-f91c-422b-86cf-3ace891e742f
# ╟─75f02433-7ef3-45f4-ad2a-d610923824af
# ╟─b3349485-1a5f-4da6-be79-965fec5fbc13
# ╟─0c5f21fe-80fc-4810-b4a8-bb1c290c5d81
# ╟─d0ec67d4-26bd-4960-b4d9-75d9c0b740b0
# ╟─4195fc74-92a0-4ad4-8c36-9d652ac36fd5
# ╟─febd9959-70d6-409d-b216-762ec0b751dc
# ╟─8a70f0c0-c70e-4022-8a06-ea8fec31a7c7
# ╟─0af240e8-ccf7-4bf8-b21c-08e2972d2459
# ╟─09b3d1e1-b44f-461d-abaf-2e5a1f6f6cc2
# ╟─0d222d9d-7576-40a7-acb4-b0e27f9a9ef8
# ╟─e0ef1b4d-0076-410c-b325-32a7a274f487
# ╟─8d11bae1-d232-4c03-981a-cd9f1db24e6b
# ╟─3bcd40f0-e8dd-48bd-8249-51b89f5766de
