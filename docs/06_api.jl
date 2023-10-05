### A Pluto.jl notebook ###
# v0.19.26

#> [frontmatter]
#> title = "API"
#> category = "API"

using Markdown
using InteractiveUtils

# ╔═╡ 94428604-9725-417c-a3c4-174b32b7cfd6
# ╠═╡ show_logs = false
begin
	using Pkg
	Pkg.activate(temp = true)
	Pkg.add(url = "https://github.com/Dale-Black/DistanceTransforms.jl")
	Pkg.add("PlutoUI")

	using DistanceTransforms
	using PlutoUI
end

# ╔═╡ 0e3a840c-aebe-47e6-bdba-be8b54af2e55
TableOfContents()

# ╔═╡ cb588ce4-3d28-4337-891a-999ebe9ccc79
md"""
# API
"""

# ╔═╡ 5cb2767d-7ab0-4fec-ad61-163204d8913a
all_names = [name for name in names(DistanceTransforms)]

# ╔═╡ cbb16a36-41b5-450d-b0ce-6bd2b2601706
exported_functions = filter(x -> x != :DistanceTransforms, all_names)

# ╔═╡ d4b66280-c9e4-4c27-9cd3-6aaf179b8e0d
function generate_docs(exported_functions)
    PlutoUI.combine() do Child
        md"""
        $([md" $(Docs.doc(eval(name)))" for name in exported_functions])
        """
    end
end

# ╔═╡ d7559c62-1c0a-4e86-a642-74171d742c41
generate_docs(exported_functions)

# ╔═╡ Cell order:
# ╠═94428604-9725-417c-a3c4-174b32b7cfd6
# ╠═0e3a840c-aebe-47e6-bdba-be8b54af2e55
# ╟─cb588ce4-3d28-4337-891a-999ebe9ccc79
# ╠═d7559c62-1c0a-4e86-a642-74171d742c41
# ╠═5cb2767d-7ab0-4fec-ad61-163204d8913a
# ╠═cbb16a36-41b5-450d-b0ce-6bd2b2601706
# ╠═d4b66280-c9e4-4c27-9cd3-6aaf179b8e0d
