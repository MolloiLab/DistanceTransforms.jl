## -- Vanilla loss functions --##

function dice_loss(ŷ, y)
    ϵ = 1e-5
    loss = 1 - ((2 * sum(ŷ .* y) + ϵ) / (sum(ŷ .* ŷ) + sum(y .* y) + ϵ))
end

function hd_loss(ŷ, y, ŷ_dtm, y_dtm)
    M = (ŷ .- y).^2 .* (ŷ_dtm .^ 2 .+ y_dtm .^ 2)
    loss = mean(M)
end

## -- Parallel loss functions --##

function dice_lossP(ŷ, y) 
    ϵ = 1e-5
    @tullio loss := 1 - ((2 * sum(ŷ[i,j,k] .* y[i,j,k]) + ϵ) / (sum(ŷ[i,j,k] .* ŷ[i,j,k]) + sum(y[i,j,k] .* y[i,j,k]) + ϵ))
end

function hd_lossP(ŷ, y, ŷ_dtm, y_dtm)
    @tullio tot := (ŷ[i,j,k] .- y[i,j,k])^2 * (ŷ_dtm[i,j,k] ^ 2 + y_dtm[i,j,k] ^ 2)
    loss = tot / length(y)
end