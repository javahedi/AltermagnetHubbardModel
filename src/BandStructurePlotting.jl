module BandStructurePlotting

using PyPlot
const plt = PyPlot  # Create an alias for PyPlot

using AltermagneticHubbardModel
using LinearAlgebra
using Statistics


export plot_band_structure, get_high_symmetry_path

"""
    get_high_symmetry_path(lattice::Symbol, npoints::Int=100) -> (kpath, labels, ticks)

Generate high-symmetry k-path for common lattices.
Returns tuple of:
- kpath: Vector of Tuples (kx, ky)
- labels: High-symmetry point names
- ticks: Positions for labels along the path
"""
function get_high_symmetry_path(lattice::Symbol, npoints::Int=100)
    if lattice == SQUARE
        # Γ -> X -> M -> Γ for square lattice
        Γ = (0.0, 0.0)
        X = (-π/2, π/2)
        M = (0, π)
        X2 = (π/2, π/2)

        # Define the k-path
        kpath = vcat(
            [(Γ[1] + t*(X[1]-Γ[1]), Γ[2] + t*(X[2]-Γ[2])) for t in range(0, 1, length=npoints)],
            [(X[1] + t*(M[1]-X[1]), X[2] + t*(M[2]-X[2])) for t in range(0, 1, length=npoints)],
            [(M[1] + t*(X2[1]-M[1]), M[2] + t*(X2[2]-M[2])) for t in range(0, 1, length=npoints)],
            [(X2[1] + t*(Γ[1]-X2[1]), X2[2] + t*(Γ[2]-X2[2])) for t in range(0, 1, length=npoints)]
        )
        
        labels = ["Γ", "X", "M", "X2", "Γ"]
        ticks = [1, npoints, 2npoints, 3npoints, 4npoints]
        
    elseif lattice == HEXATRIANGULAR
        # Γ -> K -> M -> Γ for honeycomb
        Γ = [0.0, 0.0]
        K = [4π/3, 0.0]
        M = [π, π/√3]
        
        kpath = vcat(
            [(Γ[1] + t*(K[1]-Γ[1]), Γ[2] + t*(K[2]-Γ[2])) for t in range(0, 1, length=npoints)],
            [(K[1] + t*(M[1]-K[1]), K[2] + t*(M[2]-K[2])) for t in range(0, 1, length=npoints)],
            [(M[1] + t*(Γ[1]-M[1]), M[2] + t*(Γ[2]-M[2])) for t in range(0, 1, length=npoints)]
        )
        
        labels = ["Γ", "K", "M", "Γ"]
        ticks = [1, npoints, 2npoints, 3npoints]
        
    else
        error("Unsupported lattice for k-path: $lattice")
    end
    
    return (kpath, labels, ticks)
end

"""
    plot_band_structure(params::ModelParams, δm::Float64; npoints::Int=100)

Plot spin-resolved band structure along high-symmetry path.
"""




function plot_band_structure(params::ModelParams, δm::Float64; npoints=100)
    kpath, labels, ticks = get_high_symmetry_path(params.lattice, npoints)

    # Determine system size based on lattice
    if params.lattice == HEXATRIANGULAR
        matrix_size = 6  # 3 sublattices × 2 spins
        nbands = 3       # 3 bands per spin
    else
        matrix_size = 4  # 2 sublattices × 2 spins
        nbands = 2       # 2 bands per spin
    end
    
    # Storage for 2 spin-up and 2 spin-down bands
    ϵ_up = zeros(length(kpath), nbands)
    ϵ_dn = zeros(length(kpath), nbands)
    
    for (i,k) in enumerate(kpath)
        H = build_hamiltonian(k, params, δm)
        
        # Extract and diagonalize spin blocks
        H_up = H[1:matrix_size÷2, 1:matrix_size÷2]
        H_dn = H[matrix_size÷2+1:end, matrix_size÷2+1:end]
        
        ϵ_up[i,:] = sort(real(eigvals(Hermitian(H_up)))[1:nbands])
        ϵ_dn[i,:] = sort(real(eigvals(Hermitian(H_dn)))[1:nbands])

    end
    
    # Plotting (same as before)
    fig, ax = plt.subplots(figsize=(7,3))
    for b in 1:nbands
        ax.plot(1:length(kpath), ϵ_up[:,b], "r-", label=b==1 ? "Spin Up" : "")
        ax.plot(1:length(kpath), ϵ_dn[:,b], "b-", label=b==1 ? "Spin Down" : "")
    end
    
    
    # Formatting
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.set_xlim(1, length(kpath))
    ax.axhline(0, color="black", linestyle=":", alpha=0.5)  # Fermi level
    ax.set_xlabel("Wave Vector (k)")
    ax.set_ylabel("Energy (t)")
    ax.set_title("Spin-Resolved Band Structure\n(δm = $(round(δm, digits=4)))")
    ax.legend()
    ax.grid(alpha=0.3)
    
    return fig
end
    
end # module