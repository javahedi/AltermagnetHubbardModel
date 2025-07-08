module ConductivityTensor

using LinearAlgebra
using AltermagneticHubbardModel
using QuadGK
using Statistics
using Base.Threads

export calculate_conductivity

"""
    calculate_velocity(H::Matrix{Float64}, k::Tuple{Float64,Float64}, params::ModelParams, δm::Float64, α::Int)

Compute velocity operator ∂H/∂kₐ using centered differences for high accuracy (ℏ = 1). 
α=1 corresponds to x-direction, α=2 to y-direction.
"""
function calculate_velocity(k::Tuple{Float64,Float64}, params::ModelParams, δm::Float64, α::Int)
    Δk = 1e-4
    k_plus  = α == 1 ? (k[1]+Δk, k[2]) : (k[1], k[2]+Δk)
    k_minus = α == 1 ? (k[1]-Δk, k[2]) : (k[1], k[2]-Δk)
    
    H_plus  = build_hamiltonian(k_plus, params, δm)
    H_minus = build_hamiltonian(k_minus, params, δm)
    
    # Same size blocks for spin projection
    return (H_plus - H_minus) / (2Δk)
end



"""
    project_spin_block(H::AbstractMatrix, spin::Symbol)

Extract the spin block (:up or :dn) from the Hamiltonian matrix.
For 6×6 matrices (3-sublattice system), returns 3×3 blocks.
For 4×4 matrices (2-sublattice system), returns 2×2 blocks.

# Arguments
- `H`: Hamiltonian matrix (4×4 or 6×6)
- `spin`: :up or :dn to select spin sector

# Returns
Submatrix for the requested spin sector
"""
function project_spin_block(H::AbstractMatrix, spin::Symbol)
    size_H = size(H)
    
    if size_H == (4,4)
        # 2-sublattice case (e.g., graphene)
        return spin == :up ? @view(H[1:2, 1:2]) : @view(H[3:4, 3:4])
    elseif size_H == (6,6)
        # 3-sublattice case (hextriangular)
        return spin == :up ? @view(H[1:3, 1:3]) : @view(H[4:6, 4:6])
    else
        error("Unsupported Hamiltonian size: $size_H. Expected 4×4 or 6×6")
    end
    
    # Validate spin argument
    spin ∉ (:up, :dn) && error("Invalid spin: $spin. Must be :up or :dn")
end

"""
    compute_conductivity_components(ϵ, ψ, vₓ, vᵧ, μ, β, Γ)

Compute the full conductivity tensor (σₓₓ, σᵧᵧ, σₓᵧ, σᵧₓ) for given eigenstructure.
"""
function compute_conductivity_components(ϵ::Vector{Float64}, ψ::AbstractMatrix,
                                         vₓ::AbstractMatrix, vᵧ::AbstractMatrix,
                                         μ::Float64, β::Float64, Γ::Float64)

    # Velocity matrix elements in eigenbasis
    vₓ_mn = [ψ[:,m]' * vₓ * ψ[:,n] for m in eachindex(ϵ), n in eachindex(ϵ)]
    vᵧ_mn = [ψ[:,m]' * vᵧ * ψ[:,n] for m in eachindex(ϵ), n in eachindex(ϵ)]

    if β == Inf
        σ_xx = σ_yy = σ_xy = σ_yx = 0.0
        for m in eachindex(ϵ), n in eachindex(ϵ)
            if isapprox(ϵ[m], μ; atol=1e-4) && isapprox(ϵ[n], μ; atol=1e-4)
                prefactor = 1 / (4π * Γ)
                σ_xx += abs2(vₓ_mn[m,n]) * prefactor
                σ_yy += abs2(vᵧ_mn[m,n]) * prefactor
                σ_xy += real(vₓ_mn[m,n] * conj(vᵧ_mn[n,m])) * prefactor
                σ_yx += real(vᵧ_mn[m,n] * conj(vₓ_mn[n,m])) * prefactor
            end
        end
        return (σ_xx, σ_yy, σ_xy, σ_yx)
    else
        # Integrate each conductivity component separately
        function component_integrand(ϵ_val)
            f = 1 / (1 + exp(β * (ϵ_val - μ)))
            df = β * f * (1 - f)

            sum_xx = sum_yy = sum_xy = sum_yx = 0.0
            for m in eachindex(ϵ), n in eachindex(ϵ)
                Aₘ = Γ / (π * ((ϵ_val - ϵ[m])^2 + Γ^2))
                Aₙ = Γ / (π * ((ϵ_val - ϵ[n])^2 + Γ^2))
                weight = Aₘ * Aₙ * df

                sum_xx += abs2(vₓ_mn[m,n]) * weight
                sum_yy += abs2(vᵧ_mn[m,n]) * weight
                sum_xy += real(vₓ_mn[m,n] * conj(vᵧ_mn[n,m])) * weight
                sum_yx += real(vᵧ_mn[m,n] * conj(vₓ_mn[n,m])) * weight
            end
            return (sum_xx, sum_yy, sum_xy, sum_yx)
        end

        # Integrate each component over energy
        limits = (μ - 10/β, μ + 10/β)
        σ_xx = quadgk(ϵ -> component_integrand(ϵ)[1], limits...; rtol=1e-5)[1]
        σ_yy = quadgk(ϵ -> component_integrand(ϵ)[2], limits...; rtol=1e-5)[1]
        σ_xy = quadgk(ϵ -> component_integrand(ϵ)[3], limits...; rtol=1e-5)[1]
        σ_yx = quadgk(ϵ -> component_integrand(ϵ)[4], limits...; rtol=1e-5)[1]

        return (σ_xx, σ_yy, σ_xy, σ_yx)
    end
end

"""
    calculate_spin_conductivity(params, δm, spin; Γ=0.01, nk=50)

Calculate (σ_longitudinal, σ_transverse, σ_Hall) for a given spin.
"""
function calculate_spin_conductivity(params::ModelParams, 
                                    δm::Float64, spin::Symbol; Γ=0.01, nk=50)
                                    
    μ       = find_chemical_potential(params, δm)
    kpoints = generate_kpoints(params.lattice, nk)

    # Initialize atomic variables for thread-safe accumulation
    σ_xx = Threads.Atomic{Float64}(0.0)
    σ_yy = Threads.Atomic{Float64}(0.0)
    σ_xy = Threads.Atomic{Float64}(0.0)
    σ_yx = Threads.Atomic{Float64}(0.0)

    @threads for k in kpoints
        H = build_hamiltonian(k, params, δm)
        H_spin = project_spin_block(H, spin)
        ϵ, ψ = eigen(Hermitian(H_spin))

        vfullₓ = calculate_velocity(k, params, δm, 1)
        vfullᵧ = calculate_velocity(k, params, δm, 2)

        vₓ = project_spin_block(vfullₓ, spin)
        vᵧ = project_spin_block(vfullᵧ, spin)

        xx, yy, xy, yx = compute_conductivity_components(ϵ, ψ, vₓ, vᵧ, μ, params.β, Γ)

        atomic_add!(σ_xx, xx)
        atomic_add!(σ_yy, yy)
        atomic_add!(σ_xy, xy)
        atomic_add!(σ_yx, yx)
    end

    norm = length(kpoints)
    # Extract Atomic Values Before Division
    σ_xx_val = σ_xx.value
    σ_yy_val = σ_yy.value
    σ_xy_val = σ_xy.value
    σ_yx_val = σ_yx.value

    σ_longitudinal = [σ_xx_val, σ_yy_val] ./ norm
    σ_transverse   = (σ_xy_val + σ_yx_val) / (2 * norm)
    σ_Hall         = (σ_xy_val - σ_yx_val) / (2im * norm)



    return (
        longitudinal = real.(σ_longitudinal),
        transverse   = real(σ_transverse),
        hall         = real(σ_Hall)
    )
end

"""
    calculate_conductivity(params::ModelParams, δm::Float64; Γ=0.01, nk=50)

Compute spin-resolved conductivity tensor for both spins.
Returns a named tuple with up/down components.
"""
function calculate_conductivity(params::ModelParams, δm::Float64; Γ=0.01, nk=50)
    up = calculate_spin_conductivity(params, δm, :up; Γ, nk)
    dn = calculate_spin_conductivity(params, δm, :dn; Γ, nk)

    return (
        up_longitudinal = up.longitudinal,
        up_transverse   = up.transverse,
        up_Hall         = up.hall,
        down_longitudinal = dn.longitudinal,
        down_transverse   = dn.transverse,
        down_Hall         = dn.hall,
        spin_Hall       = up.hall - dn.hall
    )
end

end
