using DistanceTransforms
using Documenter

DocMeta.setdocmeta!(DistanceTransforms, :DocTestSetup, :(using DistanceTransforms); recursive=true)

makedocs(;
    modules=[PracticePackage],
    authors="Dale-Black <djblack@uci.edu> and contributors",
    repo="https://github.com/Dale-Black/DistanceTransforms.jl/blob/{commit}{path}#{line}",
    sitename="DistanceTransforms.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://Dale-Black.github.io/DistanceTransforms.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/Dale-Black/DistanceTransforms.jl",
)