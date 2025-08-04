using LinearAlgebra
using AltermagneticHubbardModel  # For ModelParams
using Logging
using Dates, BSON 

fixed_n = 1.0
α_vals = 0.0:π/32:π/2

Uc_zz_vals_α = Float64[]
Uc_pm_vals_α = Float64[]

# Initialize arrays to store all susceptibility data if needed
all_qpoints = []
all_chi_zz = []
all_chi_pm = []

for (i, α) in enumerate(α_vals)
    # Choose small U (noninteracting) to compute χ₀ at that α
    U_test = 0.1
    params = ModelParams(
        lattice = ALPHA_T3,
        t       = 1.0,
        t_prime = 0.0,
        δ       = 0.0,
        U       = 0.0,  # irrelevant at this stage
        λ       = 0.0,
        n       = fixed_n,
        β       = 1000.0,
        α       = α,
        kpoints = 10,
        mixing  = 0.4,
        tol     = 1e-6
    )

    # Get chemical potential for consistency
    δm = 0.0
    μ = find_chemical_potential(params, δm; μ_min=-3.0, μ_max=3.0)

    # Compute susceptibility
    qpoints, chi_zz, chi_pm, _ = compute_chi_q0(params, μ, δm; η=1e-3)

    # Store all susceptibility data if needed
    push!(all_qpoints, qpoints)
    push!(all_chi_zz, chi_zz)
    push!(all_chi_pm, chi_pm)

    # Extract max eigenvalue over all q-points
    λmax_zz = maximum(chi_zz)
    λmax_pm = maximum(chi_pm)

    # Estimate Uc
    push!(Uc_zz_vals_α, 1.0 / λmax_zz)
    push!(Uc_pm_vals_α, 1.0 / λmax_pm)
    @show α, Uc_zz_vals_α[end], Uc_pm_vals_α[end]
end

# Prepare data for saving
save_data = Dict(
    :Uc_zz_vals_α => Uc_zz_vals_α,
    :Uc_pm_vals_α => Uc_pm_vals_α,
    :n => fixed_n,
    :α_vals => collect(α_vals),  # Convert range to array
    # Optional: include susceptibility data from last iteration
    :last_qpoints => all_qpoints[end],
    :last_chi_zz => all_chi_zz[end],
    :last_chi_pm => all_chi_pm[end],
    # Or include all susceptibility data
    # :all_qpoints => all_qpoints,
    # :all_chi_zz => all_chi_zz,
    # :all_chi_pm => all_chi_pm
)

filename = "Susceptibility_n$(fixed_n)_$(round(π/4, digits=3)).bson"
BSON.@save filename save_data