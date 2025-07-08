module ConductivityTensorFull

using LinearAlgebra
using AltermagneticHubbardModel
using Statistics
using Base.Threads

export calculate_conductivity

"""
    calculate_velocity(k::Tuple{Float64,Float64}, params::ModelParams, δm::Float64, α::Int)

Compute velocity operator ∂H/∂kₐ using centered differences for high accuracy (ℏ = 1).
α=1 corresponds to x-direction, α=2 to y-direction.
Returns a ComplexF64 matrix.
"""
function calculate_velocity(k::Tuple{Float64,Float64}, params::ModelParams, δm::Float64, α::Int)
    Δk = 1e-4 # Finite difference step size
    
    k_plus  = α == 1 ? (k[1]+Δk, k[2]) : (k[1], k[2]+Δk)
    k_minus = α == 1 ? (k[1]-Δk, k[2]) : (k[1], k[2]-Δk)
    
    H_plus  = build_hamiltonian(k_plus, params, δm)
    H_minus = build_hamiltonian(k_minus, params, δm)

    # Add validation:
    v = (H_plus - H_minus) / (2Δk)
    @assert all(isfinite.(v)) "Non-finite velocity matrix"
    return v
end

"""
    compute_conductivity_components(ϵ, ψ, vₓ, vᵧ, μ, β, Γ)

Compute the DC conductivity tensor (σₓₓ, σᵧᵧ, σₓᵧ, σᵧₓ) for given eigenstructure
using the Kubo-Greenwood formula with broadening Γ.
Returns complex values.
"""
function compute_conductivity_components(ϵ::Vector{Float64}, ψ::AbstractMatrix{<:ComplexF64},
                                         vₓ::AbstractMatrix{<:ComplexF64}, vᵧ::AbstractMatrix{<:ComplexF64},
                                         μ::Float64, β::Float64, Γ::Float64)

    # Velocity matrix elements in eigenbasis
    vₓ_mn = [ψ[:,m]' * vₓ * ψ[:,n] for m in eachindex(ϵ), n in eachindex(ϵ)]
    vᵧ_mn = [ψ[:,m]' * vᵧ * ψ[:,n] for m in eachindex(ϵ), n in eachindex(ϵ)]

    σ_xx = 0.0 + 0.0im
    σ_yy = 0.0 + 0.0im
    σ_xy = 0.0 + 0.0im
    σ_yx = 0.0 + 0.0im

    for m in eachindex(ϵ), n in eachindex(ϵ)
        fm = fermi(ϵ[m], μ, β)
        fn = fermi(ϵ[n], μ, β)

        if m == n # Intraband (Drude) contribution
            # The df/dE term is -β * f(1-f). We want (f_m - f_n)/(E_m - E_n) which -> -df/dE.
            # So, for m=n, the term (f_m - f_n)/(E_m - E_n) becomes -beta * fm * (1-fm)
            # And the denominator (E_m - E_n - iGamma) becomes -iGamma.
            # Overall: (-beta * fm * (1-fm)) / (-iGamma) = (beta * fm * (1-fm)) / (iGamma)
            # This is complex. For real part, often take Γ for the scattering time.
            # Real part of Drude (often given by: n * e^2 * tau / m):
            # In Kubo, this comes from (df/dE) * |v|^2 / Gamma
            
            # For m=n, the formula usually simplifies to:
            # Re[σ_αβ] = - (∂f/∂ϵ)_m |v_α_{mm}|^2 / Γ_m
            # Im[σ_αβ] = 0 (for diagonal terms)
            
            # Using the limit of (f_m - f_n) / (ϵ_m - ϵ_n) as m->n:
            df_dE = -β * fm * (1 - fm) # This is -∂f/∂ϵ

            # Standard intraband contribution for DC real conductivity.
            # It's (e^2/hbar) * sum_k sum_m (-df/dE)_m * <m|v_x|m><m|v_x|m> / Gamma
            # The (e^2/hbar) is outside. V is handled by k-point sum norm.
             # Add small threshold to avoid /0
            σ_xx += df_dE * abs2(vₓ_mn[m,m]) / (Γ + 1e-10)
            σ_yy += df_dE * abs2(vᵧ_mn[m,m]) / (Γ + 1e-10)

        else # Interband contribution
            delta_E = ϵ[m] - ϵ[n]
            
            # Skip if effectively degenerate
            if abs(delta_E) < 1e-10
                continue
            end
            
            # The full complex prefactor for interband terms
            prefactor = (fm - fn) / (delta_E - 1im * Γ) / delta_E

            σ_xx += vₓ_mn[m,n] * vₓ_mn[n,m] * prefactor
            σ_yy += vᵧ_mn[m,n] * vᵧ_mn[n,m] * prefactor
            σ_xy += vₓ_mn[m,n] * vᵧ_mn[n,m] * prefactor
            σ_yx += vᵧ_mn[m,n] * vₓ_mn[n,m] * prefactor
        end
    end
    return (σ_xx, σ_yy, σ_xy, σ_yx)
end


"""
    calculate_total_charge_conductivity(params::ModelParams, δm::Float64; Γ=0.01, nk=50)

Compute the total charge conductivity tensor (complex values) by summing over k-points.
"""
# In calculate_total_charge_conductivity function:
function calculate_total_charge_conductivity(params::ModelParams, 
                                            δm::Float64; Γ=0.02, nk=100)

    μ       = find_chemical_potential(params, δm)
    @assert isfinite(μ) "Chemical potential is not finite: $μ"
    kpoints = generate_kpoints(params.lattice, nk)

    # Initialize atomic variables for real and imaginary parts
    σ_xx_real_atomic = Threads.Atomic{Float64}(0.0)
    σ_xx_imag_atomic = Threads.Atomic{Float64}(0.0)
    σ_yy_real_atomic = Threads.Atomic{Float64}(0.0)
    σ_yy_imag_atomic = Threads.Atomic{Float64}(0.0)
    σ_xy_real_atomic = Threads.Atomic{Float64}(0.0)
    σ_xy_imag_atomic = Threads.Atomic{Float64}(0.0)
    σ_yx_real_atomic = Threads.Atomic{Float64}(0.0)
    σ_yx_imag_atomic = Threads.Atomic{Float64}(0.0)

    @threads for k in kpoints
        H = build_hamiltonian(k, params, δm)
        @assert all(isfinite.(H)) "Hamiltonian contains NaN/Inf at k=$k"
        ϵ, ψ = eigen(Hermitian(H))

        vₓ = calculate_velocity(k, params, δm, 1)
        vᵧ = calculate_velocity(k, params, δm, 2)

        xx_k, yy_k, xy_k, yx_k = compute_conductivity_components(ϵ, ψ, vₓ, vᵧ, μ, params.β, Γ)

        atomic_add!(σ_xx_real_atomic, real(xx_k))
        atomic_add!(σ_xx_imag_atomic, imag(xx_k))
        atomic_add!(σ_yy_real_atomic, real(yy_k))
        atomic_add!(σ_yy_imag_atomic, imag(yy_k))
        atomic_add!(σ_xy_real_atomic, real(xy_k))
        atomic_add!(σ_xy_imag_atomic, imag(xy_k))
        atomic_add!(σ_yx_real_atomic, real(yx_k))
        atomic_add!(σ_yx_imag_atomic, imag(yx_k))
    end

    norm_factor = length(kpoints)
    
    # Reconstruct complex values from atomic parts
    σ_xx_val = (σ_xx_real_atomic.value + σ_xx_imag_atomic.value * im) / norm_factor
    σ_yy_val = (σ_yy_real_atomic.value + σ_yy_imag_atomic.value * im) / norm_factor
    σ_xy_val = (σ_xy_real_atomic.value + σ_xy_imag_atomic.value * im) / norm_factor
    σ_yx_val = (σ_yx_real_atomic.value + σ_yx_imag_atomic.value * im) / norm_factor

    return (
        xx = σ_xx_val,
        yy = σ_yy_val,
        xy = σ_xy_val,
        yx = σ_yx_val
    )
end

"""
    calculate_conductivity(params::ModelParams, δm::Float64; Γ=0.01, nk=50)

Compute relevant charge conductivity components from the full conductivity tensor.
Returns a named tuple with relevant real components.
"""
function calculate_conductivity(params::ModelParams, δm::Float64; Γ=0.01, nk=50)
    # Calculate the full complex conductivity tensor
    total_cond_tensor = calculate_total_charge_conductivity(params, δm; Γ, nk)

    # Extract real parts for physical interpretation
    σ_xx_real = real(total_cond_tensor.xx)
    σ_yy_real = real(total_cond_tensor.yy)
    σ_xy_real = real(total_cond_tensor.xy)
    σ_yx_real = real(total_cond_tensor.yx)

    # Longitudinal conductivity components (diagonal, real part)
    σ_longitudinal_charge = [σ_xx_real, σ_yy_real]
    
    # Transverse conductivity (symmetric part of off-diagonal, real part)
    σ_transverse_charge   = (σ_xy_real + σ_yx_real) / 2.0
    
    # Hall conductivity (antisymmetric part of off-diagonal, real part of imaginary part, or just imaginary part)
    # Conventionally, AHE is real(σ_xy - σ_yx)/2 or just σ_xy if it's the only off-diagonal part.
    # The Berry curvature part of Hall conductivity is often Im[σ_xy].
    # Let's return the full complex values and then the real/imaginary parts based on standard definitions.
    # For anomalous Hall, it's typically Re[sigma_xy - sigma_yx]/2 (odd under T, even under P)
    # or sometimes Im[sigma_xy] for intrinsic Berry curvature mechanism.
    # Given that `sigma_xy` is complex from `compute_conductivity_components`, the Hall effect related to `Im(sigma_xy)` is more likely.
    # The Hall conductivity is generally Re[sigma_xy]. If it arises from time-reversal symmetry breaking, it's real.
    # If it's Berry phase (intrinsic), it's often related to Im[sigma_xy] if we use Green's function, or a purely real formula.
    # Let's define AHE as real part of the anti-symmetric off-diagonal component.
    σ_Hall_charge = (σ_xy_real - σ_yx_real) / 2.0

    # You might also want the intrinsic Berry curvature Hall, which is often related to the imaginary part of sigma_xy.
    # For altermagnets, AHE is expected to be real and could be anisotropic.

    return (
        charge_xx = σ_xx_real,
        charge_yy = σ_yy_real,
        charge_xy = σ_xy_real, # The full real sigma_xy (including both symmetric and antisymmetric parts)
        charge_yx = σ_yx_real, # The full real sigma_yx
        charge_longitudinal = σ_longitudinal_charge,
        charge_transverse_symmetric = σ_transverse_charge, # (sigma_xy_real + sigma_yx_real)/2
        charge_Hall_antisymmetric   = σ_Hall_charge,       # (sigma_xy_real - sigma_yx_real)/2
        # If you need Berry curvature related Hall, it's typically a separate calc or Im[σ_xy] if formula implies it.
        # Im_sigma_xy = imag(total_cond_tensor.xy)
    )
end

# This part for spin conductivity is fundamentally flawed with SOC.
# If you want spin currents, you need to define spin current operators.
# Example: J_x^s = 1/2 {v_x, S_z} where S_z is the spin operator (4x4 matrix in your basis)
# You would then calculate spin conductivity using these operators.
# For now, commenting out and advising to rethink if truly needed.
# function calculate_spin_conductivity(...)
#     error("Spin conductivity calculation is not valid with SOC by simply projecting Hamiltonians.
#            You need to define spin current operators and compute correlation functions.")
# end

end # module