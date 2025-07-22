using Distributed
addprocs()  # You can specify number of workers if needed, e.g., addprocs(8)
using Dates

@everywhere begin
    using AltermagneticHubbardModel
    using BSON: @save, @load


    # Define the calculation at one point
    function compute_δm_at(n::Float64, U::Float64, α::Float64)
        params = ModelParams(
            lattice = ALPHA_T3,
            t       = 1.0,
            t_prime = 0.0,
            δ       = 0.0,
            U       = U,
            λ       = 0.0,
            n       = n,
            β       = 10000.0,
            α       = α,
            kpoints = 20,
            mixing  = 0.4,
            tol     = 1e-6
        )

        try
            δm = run_scf(params)
            return ((n, U) => δm)
        catch e
            return ((n, U) => "error: $(e)")
        end
    end
end

function main()
    fixed_α = π/4
    n_vals  = 0.8:0.1:1.2
    U_vals  = 0.0:1.0:4.0

    param_list = [(n, U, fixed_α) for n in n_vals, U in U_vals]
    param_list = vec(param_list)  # flatten

    # Use pmap to run in parallel
    results_list = pmap(p -> compute_δm_at(p[1], p[2], p[3]), param_list)

    # Merge into a Dict
    results = Dict(results_list...)
    
    # Save results using BSON
    timestamp = Dates.format(now(), "yyyy-mm-dd_HHMM")
    filename = "phaseDiagram_n_U_alpha$(round(fixed_α, digits=2))_$timestamp.bson"
    @save filename results n_vals U_vals fixed_α
    println("Saved results to $filename")
    
end

main()


#=

NOTE: 

rsults ---> # unordered! due to parallel execution

Solution: 

n_grid = collect(n_vals)
U_grid = collect(U_vals)

Z = [get(results, (n, U), NaN) for n in n_grid, U in U_grid]

=#