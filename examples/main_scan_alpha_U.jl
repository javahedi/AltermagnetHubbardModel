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
            #return ("α=$(round(α, digits=3)), U=$(U)" => δm)
            return ((α, U) => δm)  # key is a Tuple{Float64, Float64}
        catch e
            return ((α, U) => "error: $(e)")
        end
    end
end

function main()
    fixed_n = 1.0
    α_vals = 0.0:π/16:π/2
    U_vals = 0.0:1.0:4.0

    param_list = [(fixed_n, U, α) for α in α_vals, U in U_vals]
    param_list = vec(param_list)  # flatten

    # Use pmap to run in parallel
    results_list = pmap(p -> compute_δm_at(p[1], p[2], p[3]), param_list)

    # Merge into a Dict
    results = Dict(results_list...)

    
    # Save results using BSON
    timestamp = Dates.format(now(), "yyyy-mm-dd_HHMM")
    filename = "phaseDiagram_alpha_U_n$(round(fixed_n, digits=2))_$timestamp.bson"
    @save filename results α_vals U_vals fixed_n
    println("Saved results to $filename")
    
end

main()


#=

NOTE: 

rsults ---> # unordered! due to parallel execution

Solution: 

α_grid = collect(α_vals)
U_grid = collect(U_vals)

Z = [get(results, (α, U), NaN) for α in α_grid, U in U_grid]

=#