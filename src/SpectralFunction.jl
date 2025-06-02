module SpectralFunction

using PyPlot
const plt = PyPlot  # Create an alias for PyPlot
using LinearAlgebra
using AltermagneticHubbardModel
using LaTeXStrings
export plot_spectral_function

"""
    spectral_function(params::ModelParams, δm::Float64, k::Tuple{Float64,Float64}, ω::Float64; η=0.05)

Calculate A(ω,k) = -1/π Im[Tr(G(ω,k))] where G is the retarded Green's function.
η is the broadening parameter (default 0.05).
"""
function spectral_function(params::ModelParams, δm::Float64, k::Tuple{Float64,Float64}, ω::Float64; η=0.05)
    H = build_hamiltonian(k, params, δm)
    G = inv((ω + im*η)*I - H)  # Retarded Green's function
    return -1/π * imag(tr(G))  # Total spectral function
end

"""
    spin_resolved_spectral_function(params::ModelParams, δm::Float64, k::Tuple{Float64,Float64}, ω::Float64; η=0.05)

Calculate spin-up and spin-down spectral functions separately.
Returns (A_up, A_dn).
"""
function spin_resolved_spectral_function(params::ModelParams, δm::Float64, k::Tuple{Float64,Float64}, ω::Float64; η=0.05)
    H = build_hamiltonian(k, params, δm)
    G = inv((ω + im*η)*I - H)
    
    # Project onto spin components
    A_up = -1/π * imag(tr(G[1:2,1:2]))  # Spin-up block
    A_dn = -1/π * imag(tr(G[3:4,3:4]))  # Spin-down block
    
    return (A_up, A_dn)
end

"""
    plot_spectral_function(params::ModelParams, δm::Float64; nω=100, nk=50, η=0.05, k_path=nothing)

Plot the spectral function along a high-symmetry k-path.
"""
function plot_spectral_function(params::ModelParams, δm::Float64; nω=200, nk=100, η=0.05, k_path=nothing)
    μ = find_chemical_potential(params, δm)
    
    # Default k-path: Γ-X-M-Γ for square lattice
    if k_path === nothing && params.lattice == SQUARE
        # High-symmetry points for square lattice
        Γ = (0.0, 0.0)
        X = (-π/2, π/2)
        M = (0, π)
        X2 = (π/2, π/2)
        k_path = [Γ, X, M, X2, Γ]
        xticklabels = ["Γ", "X", "M", "X2'", "Γ"]
    elseif k_path === nothing && params.lattice == HONEYCOMB
        # Γ-K-M-Γ for honeycomb lattice
        Γ = (0.0, 0.0)
        M = (π, 2π/(2√3))  # K point in honeycomb
        K = (4π/3, 0.0)
        
        k_path = [Γ, K, M, Γ]
        xticklabels = ["Γ", "K", "M", "Γ"]
    elseif k_path === nothing
        error("No k-path provided and default path not defined for lattice: $(params.lattice)")
    end
    
    # Interpolate k-points along path
    k_interp = interpolate_kpath(k_path, nk)
    
    # Energy range (relative to μ)
    ω_range = range(-4.0, 4.0, length=nω)  # ±3t
    
    # Calculate A(ω,k)
    A_kω = zeros(length(k_interp), nω)
    A_up_kω = zeros(length(k_interp), nω)
    A_dn_kω = zeros(length(k_interp), nω)
    
    for (i,k) in enumerate(k_interp), (j,ω) in enumerate(ω_range)
        A_kω[i,j] = spectral_function(params, δm, k, μ + ω, η=η)
        A_up_kω[i,j], A_dn_kω[i,j] = spin_resolved_spectral_function(params, δm, k, μ + ω, η=η)
    end
    
    # Plotting
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(9,8), sharex=true)
    
    # Total spectral function
    im1 = ax1.pcolormesh(1:length(k_interp), ω_range, A_kω', shading="auto", cmap="gnuplot")
    fig.colorbar(im1, ax=ax1, label="A(ω,k)")
    ax1.set_title("Total Spectral Function")
    ax1.set_ylabel(L"$\omega - \mu[t]$")
    
    # Spin-up
    im2 = ax2.pcolormesh(1:length(k_interp), ω_range, A_up_kω', shading="auto", cmap="Reds")
    fig.colorbar(im2, ax=ax2, label=L"A↑(ω,k)")
    ax2.set_title("")
    ax2.set_ylabel(L"$\omega - \mu[t]$")
    
    # Spin-down
    im3 = ax3.pcolormesh(1:length(k_interp), ω_range, A_dn_kω', shading="auto", cmap="Blues")
    fig.colorbar(im3, ax=ax3, label=L"A_↓(ω,k)")
    ax3.set_title("")
    ax3.set_ylabel(L"$\omega - \mu[t]$")
    
    # Set k-path labels
    tick_positions = range(1, length(k_interp), length=length(k_path))
    ax3.set_xticks(tick_positions)
    ax3.set_xticklabels(xticklabels)
    ax3.set_xlabel("Wave Vector (k)")
    
    plt.tight_layout()
    return fig
end

"""
    interpolate_kpath(k_points, nk_per_segment)

Create interpolated k-path through high-symmetry points.
"""
function interpolate_kpath(k_points, nk_per_segment)
    k_interp = Tuple{Float64,Float64}[]
    for i in 1:length(k_points)-1
        k1 = k_points[i]
        k2 = k_points[i+1]
        for t in range(0, 1, length=nk_per_segment)
            push!(k_interp, (k1[1] + t*(k2[1]-k1[1]), k1[2] + t*(k2[2]-k1[2])))
        end
    end
    return k_interp
end

end