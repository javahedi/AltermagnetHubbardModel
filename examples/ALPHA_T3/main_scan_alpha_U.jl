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
            β       = 1000.0,
            α       = α,
            kpoints = 100,
            mixing  = 0.4,
            tol     = 1e-6
        )

        try
            δm, m_plus, μ = run_scf(params; verbose=false)
            return (α=α, U=U, n=n, δm=δm, m_plus=m_plus, μ=μ, error=nothing)
        catch e
            @warn "Failed at α=$(round(α, digits=3)), U=$U: $e"
            return (α=α, U=U, n=n, δm=NaN, m_plus=NaN, μ=NaN, error=string(e))
        end
    end
end

function main()
    fixed_n = 1.0
    α_vals = 0.0:π/32:π/2
    U_vals = 0.0:0.5:4.0

    # Generate parameter grid
    param_list = [(fixed_n, U, α) for α in α_vals, U in U_vals]
    
    # Parallel computation with progress tracking
    println("Starting parallel computation with $(nworkers()) workers...")
    results = pmap(p -> compute_δm_at(p[1], p[2], p[3]), param_list)
    
    # Reorganize results into structured format
    results_dict = Dict()
    for r in results
        key = (r.α, r.U)
        results_dict[key] = (δm=r.δm, m_plus=r.m_plus, μ=r.μ, error=r.error)
    end

    # Create result matrices for easy plotting
    α_grid = collect(α_vals)
    U_grid = collect(U_vals)
    δm_matrix = [get(results_dict, (α, U), (δm=NaN, m_plus=NaN, μ=NaN, error="")).δm for α in α_grid, U in U_grid]
    m_plus_matrix = [get(results_dict, (α, U), (δm=NaN, m_plus=NaN, μ=NaN, error="")).m_plus for α in α_grid, U in U_grid]
    μ_matrix = [get(results_dict, (α, U), (δm=NaN, m_plus=NaN, μ=NaN, error="")).μ for α in α_grid, U in U_grid]

    # Save comprehensive results
    timestamp = Dates.format(now(), "yyyy-mm-dd_HHMMSS")
    save_data = Dict(
        :results => results_dict,
        :δm_matrix => δm_matrix,
        :m_plus_matrix => m_plus_matrix,
        :μ_matrix => μ_matrix,
        :α_vals => α_grid,
        :U_vals => U_grid,
        :n => fixed_n,
        :timestamp => timestamp
    )
    
    filename = "phase_diagram_n$(fixed_n)_$timestamp.bson"
    BSON.@save filename save_data
    println("Results saved to $filename")

    return save_data
end

# Execute
main()


