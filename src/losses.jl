## -- Vanilla loss functions --##

function dice_loss(ŷ, y)
    ϵ = 1e-5
    return loss = 1 - ((2 * sum(ŷ .* y) + ϵ) / (sum(ŷ .* ŷ) + sum(y .* y) + ϵ))
end

function hd_loss(ŷ, y, ŷ_dtm, y_dtm)
    M = (ŷ .- y) .^ 2 .* (ŷ_dtm .^ 2 .+ y_dtm .^ 2)
    return loss = mean(M)
end

## -- Parallel loss functions --##

function dice_lossP(ŷ, y)
    ϵ = 1e-5
    @tullio loss :=
        1 - (
            (2 * sum(ŷ[i, j, k, c, b] .* y[i, j, k, c, b]) + ϵ) / (
                sum(ŷ[i, j, k, c, b] .* ŷ[i, j, k, c, b]) +
                sum(y[i, j, k, c, b] .* y[i, j, k, c, b]) +
                ϵ
            )
        )
end

function hd_lossP(ŷ, y, ŷ_dtm, y_dtm)
    @tullio tot :=
        (ŷ[i, j, k, c, b] .- y[i, j, k, c, b])^2 *
        (ŷ_dtm[i, j, k, c, b]^2 + y_dtm[i, j, k, c, b]^2)
    return loss = tot / length(y)
end
