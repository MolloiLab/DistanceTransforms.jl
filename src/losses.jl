## -- Vanilla loss functions --##

function dice_loss(ŷ, y)
    ϵ = 1e-5
    intersect = sum(ŷ .* y)
    y_sum = sum(y .* y)
    z_sum = sum(ŷ .* ŷ)
    loss = (2 * intersect + ϵ) / (z_sum + y_sum + ϵ)
    loss = 1 - loss
end


function hd_loss(ŷ, y, ŷ_dtm, y_dtm)
    Δ = (ŷ .- y) .^ 2
    dtm = (ŷ_dtm .^ 2) + (y_dtm .^ 2)

    @tullio M[i,j,k] := Δ[i,j,k] * dtm[i,j,k]
    return mean(M)
end

## -- Parallel loss functions --##

function hd_lossP(ŷ, y, ŷ_dtm, y_dtm)
    @tullio Δ[i,j,k] := (ŷ[i,j,k] - y[i,j,k]) .^ 2
    @tullio dtm[i,j,k] := (ŷ_dtm[i,j,k] .^ 2) + (y_dtm[i,j,k] .^ 2)
    @tullio M[i,j,k] := Δ[i,j,k] * dtm[i,j,k]
    hd_loss = mean(M)
end