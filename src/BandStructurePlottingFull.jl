module BandStructurePlottingFull

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
        # Ensure these k-points match the conventions used in your build_hamiltonian
        # For a=1, (pi, pi) is the M-point for a simple square lattice.
        # Your previous notes used magnetic BZ with a1=(1,1), a2=(1,-1).
        # The k-points should be in the magnetic BZ.
        # Let's assume these k-points are defined in the context of your magnetic BZ.
        # If your kx, ky are in the conventional BZ, then (pi,pi) is the M-point.
        # Given your H_s(k) terms with cos(dot(k,a1)) etc., your k-points are likely in the magnetic BZ.
        # Let's use the k-points from the Das et al. paper if possible, or common ones for magnetic BZ.
        # For a d-wave altermagnet, the magnetic BZ is rotated.
        # Let's stick with your current k-points, assuming they are consistent with your magnetic BZ.

        Γ = (0.0, 0.0)
        X = (-π/2, π/2) # This is a common X point in a rotated BZ
        M = (0.0, π)    # This is a common M point in a rotated BZ
        X2 = (π/2, π/2) # Another X point for the path

        # Define the k-path
        kpath = vcat(
            [(Γ[1] + t*(X[1]-Γ[1]), Γ[2] + t*(X[2]-Γ[2])) for t in range(0, 1, length=npoints)],
            [(X[1] + t*(M[1]-X[1]), X[2] + t*(M[2]-X[2])) for t in range(0, 1, length=npoints)],
            [(M[1] + t*(X2[1]-M[1]), M[2] + t*(X2[2]-M[2])) for t in range(0, 1, length=npoints)],
            [(X2[1] + t*(Γ[1]-X2[1]), X2[2] + t*(Γ[2]-X2[2])) for t in range(0, 1, length=npoints)]
        )
        
        labels = ["Γ", "X", "M", "X'", "Γ"] # Changed X2 to X' for common notation
        ticks = [1, npoints, 2npoints, 3npoints, 4npoints]
        
   elseif lattice == HEXATRIANGULAR || lattice == ALPHA_T3
        # Γ -> K -> M -> Γ for honeycomb
        Γ = [0.0, 0.0]
        K = [4π/3, 0.0]
        Kp = [2π/3, 2π/√3]  # K' point in hexagonal BZ
        M = [π, π/√3]
        
        kpath = vcat(
            [(K[1] + t*(M[1]-K[1]), K[2] + t*(M[2]-K[2])) for t in range(0, 1, length=npoints)],
            [(M[1] + t*(Γ[1]-M[1]), M[2] + t*(Γ[2]-M[2])) for t in range(0, 1, length=npoints)]
        )
        
        labels = ["K", "M", "Γ"]
        ticks = [1, npoints, 2npoints]
        
    else
        error("Unsupported lattice for k-path: $lattice")
    end
    
    return (kpath, labels, ticks)
end

"""
    plot_band_structure(params::ModelParams, δm::Float64; npoints::Int=100)

Plot the band structure along high-symmetry path.
With SOC, bands are generally mixed-spin, so all bands are plotted together.
Optionally, spin polarization can be visualized by color.
"""
function plot_band_structure(params::ModelParams, δm::Float64; npoints=100)
    kpath, labels, ticks = get_high_symmetry_path(params.lattice, npoints)

    # Determine system size based on lattice
    if params.lattice == HEXATRIANGULAR || params.lattice == ALPHA_T3
        matrix_size = 6  # 3 sublattices × 2 spins
        nbands = 3       # 3 bands per spin
        # Define the Sz operator in your (A↑, B↑, A↓, B↓) basis
        # Sz = 1/2 * diag(1, 1, 0, -1, -1, 0)
        Sz_op = diagm([0.5, 0.5, 0.0, -0.5, -0.5, 0.0]) # Julia's diagm creates a diagonal matrix

    else
        matrix_size = 4  # 2 sublattices × 2 spins
        nbands = 2       # 2 bands per spin
        # Define the Sz operator in your (A↑, B↑, A↓, B↓) basis
        # Sz = 1/2 * diag(1, 1, -1, -1)
        Sz_op = diagm([0.5, 0.5, -0.5, -0.5]) # Julia's diagm creates a diagonal matrix

    end
    
    # Storage for all 4 bands (eigenvalues)
    all_ϵ = zeros(length(kpath), matrix_size)
    
    # Optional: Storage for average spin polarization (e.g., <Sz>) for each band
    # This allows coloring the bands based on their spin character
    avg_Sz = zeros(length(kpath), matrix_size) # Stores the average Sz for each band
    
    
    for (i,k) in enumerate(kpath)
        H = build_hamiltonian(k, params, δm)
        
        # Diagonalize the full 4x4 complex Hamiltonian
        vals, vecs = eigen(Hermitian(H)) # vals are eigenvalues (real), vecs are eigenvectors (complex)
        
        # Sort eigenvalues for consistent plotting (optional but good practice)
        # Julia's eigen usually returns sorted eigenvalues, but if not, sort them.
        perm = sortperm(vals)
        all_ϵ[i,:] = vals[perm]
        
        # Calculate average <Sz> for each band (eigenvector)
        for b in 1:matrix_size
            # For each eigenvector (column `vecs[:, perm[b]]`), calculate <ψ|Sz_op|ψ>
            # <ψ|Sz_op|ψ> = ψ_dagger * Sz_op * ψ
            avg_Sz[i,b] = real(vecs[:, perm[b]]' * Sz_op * vecs[:, perm[b]])
        end
    end
    
    # Plotting
    fig, ax = plt.subplots(figsize=(8,5)) # Adjusted figure size for better visualization

    # Plot all 4 bands. Color them based on average Sz.
    # We'll normalize Sz to be between -1 and 1 for coloring.
    # A simple colormap can be used, e.g., 'bwr' (blue-white-red)
    # Blue for spin-down like, Red for spin-up like.
    
    # Find min/max Sz for colormap scaling
    min_sz = minimum(avg_Sz)
    max_sz = maximum(avg_Sz)
    
    # Create a colormap
    cmap = plt.cm.get_cmap("bwr") # Blue-White-Red colormap

    for b in 1:matrix_size # Iterate through the 4 bands
        # Normalize Sz for coloring: map from [min_sz, max_sz] to [0, 1]
        # Avoid division by zero if min_sz == max_sz (e.g., if all Sz are 0)
        norm_sz = (max_sz - min_sz) > 1e-6 ? (avg_Sz[:,b] .- min_sz) ./ (max_sz - min_sz) : 0.5 * ones(length(kpath))
        
        # Plot each band segment by segment to apply varying color
        for j in 1:(length(kpath)-1)
            color_val = norm_sz[j] # Use the Sz value at current k-point for color
            ax.plot([j, j+1], [all_ϵ[j,b], all_ϵ[j+1,b]], color=cmap(color_val), linewidth=2)
        end
    end
    
    # Add a colorbar to explain the spin polarization
    # This is a bit tricky with `plot` directly, usually done with `scatter` or `pcolormesh`.
    # For a line plot, a proxy artist for the colorbar is needed.
    # A simpler way for a colorbar is to create a dummy scatter plot.
    mappable = plt.cm.ScalarMappable(cmap=cmap)
    mappable.set_array([min_sz, max_sz]) # Set the data range for the colorbar
    cbar = fig.colorbar(mappable, ax=ax, orientation="vertical", pad=0.02)
    cbar.set_label(L"$\langle S_z \rangle$") # Use LaTeX for label

    # Formatting
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.set_xlim(1, length(kpath))
    ax.axhline(0, color="black", linestyle=":", alpha=0.5)  # Fermi level
    ax.set_xlabel("Wave Vector (k)")
    ax.set_ylabel("Energy (t)")
    ax.set_title("Band Structure with Rashba SOC\n(δm = $(round(δm, digits=4)), λ = $(round(params.λ, digits=4)))")
    ax.grid(alpha=0.3)
    # Remove the legend for "Spin Up" / "Spin Down" as it's now represented by color
    # ax.legend() 
    
    plt.tight_layout() # Adjust layout to prevent labels overlapping
    
    return fig
end
    
end # module