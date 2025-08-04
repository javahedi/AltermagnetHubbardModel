using Distributed
using Plots
addprocs()  # In SLURM, use -p flag. Leave empty here.

@everywhere begin
    using LinearAlgebra
    using AltermagneticHubbardModel
    using Logging
    using Dates, BSON

    function rpa_susceptibility_qω(
            q::Tuple{Float64, Float64},
            ωlist::Vector{Float64},
            params::ModelParams,
            μ::Float64,
            δm::Float64,
            U::Float64,
            η::Float64 = 0.01,
            reduction::Function = χ -> maximum(real.(eigvals(Hermitian(imag(χ)))))
        )

        χ_zz_ω_scalar = Float64[]
        χ_pm_ω_scalar = Float64[]
        χ_mp_ω_scalar = Float64[]

        for ω in ωlist
            χ0_zz, χ0_pm, χ0_mp = chi_qω(q, ω, params, μ, δm; η=η)
            dim = size(χ0_zz)[1]
            #χ_zz_rpa = χ0_zz ./ (I(dim) - U .* χ0_zz)
            #χ_pm_rpa = χ0_pm ./ (I(dim) - U .* χ0_pm)
            #χ_mp_rpa = χ0_mp ./ (I(dim) - U .* χ0_mp)
            Umax = [1.0 0.0 0.0; 0.0 -1.0 0.0; 0.0 0.0 0.0]
            χ_zz_rpa = χ0_zz / (I(dim) - Umax * χ0_zz)
            χ_pm_rpa = χ0_pm / (I(dim) - Umax * χ0_pm)
            χ_mp_rpa = χ0_mp / (I(dim) - Umax * χ0_mp)

            push!(χ_zz_ω_scalar, reduction(χ_zz_rpa))
            push!(χ_pm_ω_scalar, reduction(χ_pm_rpa))
            push!(χ_mp_ω_scalar, reduction(χ_mp_rpa))
        end

        return (
            ωlist = ωlist,
            χzz   = χ_zz_ω_scalar,
            χpm   = χ_pm_ω_scalar,
            χmp   = χ_mp_ω_scalar
        )
    end
end

function compute_rpa_susceptibility_along_path(
        qpath::Vector{Tuple{Float64, Float64}},
        ωlist::Vector{Float64},
        params::ModelParams,
        μ::Float64,
        δm::Float64,
        U::Float64,
        η::Float64 = 0.01
    )
    @info "Launching RPA susceptibility calculation..."

    results = pmap(q -> rpa_susceptibility_qω(q, ωlist, params, μ, δm, U, η), qpath)

    data = Dict(
        :qpath => qpath,
        :ωlist => ωlist,
        :χzz   => [r.χzz for r in results],
        :χpm   => [r.χpm for r in results],
        :χmp   => [r.χmp for r in results],
    )

    # Save to file
    fname = "rpa_susceptibility_U$(U)_α$(round(α, digits=3))_time_$(Dates.format(now(), "yyyymmdd_HHMM")).bson"
    BSON.@save fname data
    @info "Saved results to $fname"

    return data
end


# --- Function to get μ, δm given (U, α) ---
function get_parameters(U_target, α_target, U_vals, α_vals, μ_matrix, δm_matrix)
    iU = findfirst(≈(U_target), U_vals)
    iα = findfirst(≈(α_target), α_vals)
    if iU === nothing || iα === nothing
        error("Requested (U=$U_target, α=$α_target) not found in data grid.")
    end
    return μ_matrix[iU, iα], δm_matrix[iU, iα]
end


function main(U::Float64, α::Float64)

    
    # --- Load the BSON file ---
   
    BSON.@load "examples/ALPHA_T3/phase_diagram_n1.0_2025-07-25_003225.bson" save_data  # Adjust filename accordingly

    # Unpack what you need
    μ_matrix      = save_data[:μ_matrix]
    δm_matrix     = save_data[:δm_matrix]
    α_vals        = save_data[:α_vals]
    U_vals        = save_data[:U_vals]
    fixed_n       = save_data[:n]


    μ, δm = get_parameters(U, α, U_vals, α_vals, μ_matrix, δm_matrix)

    params = ModelParams(
        lattice = ALPHA_T3,
        t       = 1.0,
        t_prime = 0.0,
        δ       = 0.0,
        U       = U,       # now used
        λ       = 0.0,
        n       = fixed_n,
        β       = 1000.0,
        α       = α,
        kpoints = 60,
        mixing  = 0.4,
        tol     = 1e-6
    )

    @show params

    qpath, labels, ticks = get_high_symmetry_path(params.lattice, params.kpoints)
    ωlist = 0.0:0.1:3.0
   
    data = compute_rpa_susceptibility_along_path(qpath, collect(ωlist), params, μ, δm, U, 0.005)

    #@show data[:χzz]

    χzz_matrix = hcat(data[:χzz]...)  # Transpose because hcat stacks columns

    heatmap(
        1:length(data[:qpath]),              # x-axis: q-index
        data[:ωlist],                        # y-axis: ω
        χzz_matrix,                          # z values
        xlabel="q index",
        ylabel="ω",
        title="χ_zz at α=$(α)",
        colorbar_title="χ_zz"
    )

    savefig("heatmap_alpha_U.png")

    



end


# --- Example: extract values and construct ModelParams ---
U = 2.0
α = π/4
main(U,α)
