function dice_loss(score, target)
    smooth = 1e-5
    intersect = sum(score .* target)
    y_sum = sum(target .* target)
    z_sum = sum(score .* score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss
end