using Distributed
addprocs()  # Add workers as needed
@everywhere using Dates, BSON

@everywhere begin
    using AltermagneticHubbardModel
    using LinearAlgebra

    # Enhanced computation function with better error handling
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
            kpoints = 100,
            mixing  = 0.4,
            tol     = 1e-6
        )

        try
            δm, m_plus = run_scf(params; verbose=false)
            return (n=n, U=U, α=α, δm=δm, m_plus=m_plus, error=nothing)
        catch e
            @warn "Failed at n=$n, U=$U, α=$α: $e"
            return (n=n, U=U, α=α, δm=NaN, m_plus=NaN, error=string(e))
        end
    end
end

function main()
    fixed_α = π/4
    n_vals = 0.6:0.02:1.4
    U_vals = 0.0:0.1:5.0

    # Generate parameter grid
    param_list = [(n, U, fixed_α) for n in n_vals, U in U_vals]
    
    # Parallel computation with progress tracking
    println("Starting parallel computation with $(nworkers()) workers...")
    results = pmap(p -> compute_δm_at(p[1], p[2], p[3]), param_list)
    
    # Reorganize results into structured format
    results_dict = Dict()
    for r in results
        key = (r.n, r.U)
        results_dict[key] = (δm=r.δm, m_plus=r.m_plus, error=r.error)
    end

    # Create result matrices for easy plotting
    n_grid = collect(n_vals)
    U_grid = collect(U_vals)
    δm_matrix = [get(results_dict, (n, U), (δm=NaN, m_plus=NaN, error="")).δm for n in n_grid, U in U_grid]
    m_plus_matrix = [get(results_dict, (n, U), (δm=NaN, m_plus=NaN, error="")).m_plus for n in n_grid, U in U_grid]

    # Save comprehensive results
    timestamp = Dates.format(now(), "yyyy-mm-dd_HHMMSS")
    save_data = Dict(
        :results => results_dict,
        :δm_matrix => δm_matrix,
        :m_plus_matrix => m_plus_matrix,
        :n_vals => n_grid,
        :U_vals => U_grid,
        :α => fixed_α,
        :timestamp => timestamp
    )
    
    filename = "phase_diagram_alpha_$(round(fixed_α, digits=3))_$timestamp.bson"
    BSON.@save filename save_data
    println("Results saved to $filename")

    return save_data
end

# Execute
main()