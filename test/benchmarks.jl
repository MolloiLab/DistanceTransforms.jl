using BenchmarkTools

@testset "Benchmarks" begin
    s = 10 # number of samples
    e = 1 # number of evals

    # Create an array of input sizes to benchmark for 2D
    sizes_2D = [2^i for i in 3:4]

    # Create DataFrames to store the benchmark results
    df_dt_2D = DataFrame(
        os_info = os_info,
        gpu_info = gpu_info,
        sizes = sizes_2D,
        dt_proposed = Float64[]
    )

    for n in sizes_2D
        f = Float32.(rand([0, 1], n, n))

        if dev != Array
            f_dev = dev(f)
            dt = @benchmark(transform($boolean_indicator($f_dev)); samples=s, evals=e)
            push!(df_dt_2D, [os_info, gpu_info, n, minimum(dt).time])
        end
    end

    # Create an array of input sizes to benchmark for 3D
    sizes_3D = [2^i for i in 0:2]

    # Create DataFrames to store the benchmark results
    df_dt_3D = DataFrame(
        os_info = os_info,
        gpu_info = gpu_info,
        sizes = sizes_3D,
        dt_proposed = Float64[]
    )

    for n in sizes_3D
        f = Float32.(rand([0, 1], n, n, n))

        if dev != Array
            f_dev = dev(f)
            dt = @benchmark(transform($boolean_indicator($f_dev)); samples=s, evals=e)
            push!(df_dt_3D, [os_info, gpu_info, n, minimum(dt).time])
        end
    end

    # Show the dataframes
    @info df_dt_2D
    @info df_dt_3D
end