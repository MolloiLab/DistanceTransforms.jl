""" 
    boolean_indicator(f)

If `f` is a boolean indicator where 0's correspond to
background and 1s correspond to foreground then mark
background pixels with large number `1e10`
"""
boolean_indicator(f) = @. ifelse(f == 0, 1.0f10, 0f0)