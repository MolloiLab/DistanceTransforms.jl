## -- Vanilla loss functions --##

function dice_loss(ŷ, y)
    ϵ = 1e-5
    intersect = sum(ŷ .* y)
    y_sum = sum(y .* y)
    z_sum = sum(ŷ .* ŷ)
    loss = (2 * intersect + ϵ) / (z_sum + y_sum + ϵ)
    loss = 1 - loss
end


function hd_loss(ŷ, y)
    ŷ_dtm = compute_dtm(ŷ)
    y_dtm = compute_dtm(y)

    ŷ_dtm = ŷ_dtm .^ 2
    y_dtm = y_dtm .^ 2

    Δ = (ŷ .- y) .^ 2
    dtm = ŷ_dtm + y_dtm

    @tullio multipled[x, y, z] := Δ[x, y, z] * dtm[x, y, z]
    hd_loss = mean(multipled)
end

## -- Parallel loss functions --##