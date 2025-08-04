module SpectralFunctionFull

using PyPlot
const plt = PyPlot  # Create an alias for PyPlot
using LinearAlgebra
using AltermagneticHubbardModel # Assuming this includes ModelParams, build_hamiltonian, find_chemical_potential
using LaTeXStrings # For proper LaTeX labels in plots
# Ensure get_high_symmetry_path is available, either by importing or defining it here.
# For now, I'll assume it's imported or defined, as you want to modify this module.
# Let's assume it's available from BandStructurePlotting if that's where it truly lives.

export plot_spectral_function

"""
    spectral_function(params::ModelParams, δm::Float64, k::Tuple{Float64,Float64}, ω::Float64; η=0.05)

Calculate A(ω,k) = -1/π Im[Tr(G(ω,k))] where G is the retarded Green's function.
η is the broadening parameter (default 0.05).
"""
function spectral_function(params::ModelParams, δm::Float64, k::Tuple{Float64,Float64}, ω::Float64; η=0.05)
    H = build_hamiltonian(k, params, δm)
    # The matrix `(ω + im*η)*I - H` should be robust.
    # It will correctly handle complex H from SOC.
    G = inv((ω + im*η)*I - H)  # Retarded Green's function (will be complex)
    return -1/π * imag(tr(G))  # Total spectral function (will be real)
end

"""
    spin_resolved_spectral_function(params::ModelParams, δm::Float64, k::Tuple{Float64,Float64}, ω::Float64; η=0.05)

Calculate spin-up and spin-down spectral functions separately using projection onto spin components.
This correctly accounts for mixed-spin eigenstates by summing diagonal elements of the Green's function
in the relevant spin-sublattice block.
Returns (A_up, A_dn).
"""
function spin_resolved_spectral_function(params::ModelParams, 
                                        δm::Float64, k::Tuple{Float64,Float64}, 
                                        ω::Float64; η=0.05)

    H = build_hamiltonian(k, params, δm)
    G = inv((ω + im*η)*I - H)
    
    # These are indeed the correct ways to get spin-resolved spectral functions
    # even with SOC, as they sum the local density of states for up/down components.
    if params.lattice == SQUARE
        A_up = -1/π * imag(tr(G[1:2,1:2]))  # Spin-up block trace (A↑, B↑)
        A_dn = -1/π * imag(tr(G[3:4,3:4]))  # Spin-down block trace (A↓, B↓)
    if params.lattice == HEXATRIANGULAR
        A_up = -1/π * imag(tr(G[1:3,1:3]))  # Spin-up block trace (A↑, B↑, C↑)
        A_dn = -1/π * imag(tr(G[4:6,4:6]))  # Spin-down block trace (A↓, B↓, C↓)

        
    return (A_up, A_dn)
end

"""
    plot_spectral_function(params::ModelParams, δm::Float64; nω=200, nk=100, η=0.05, k_path_override=nothing)

Plot the total and spin-resolved spectral functions along a high-symmetry k-path.
`k_path_override` should be a vector of high-symmetry k-points (e.g., `[(0.0,0.0), (π,0.0), ...]`).
If `k_path_override` is `nothing`, `get_high_symmetry_path` is used based on `params.lattice`.
"""
function plot_spectral_function(params::ModelParams, δm::Float64; nω=200, nk=100, η=0.05, k_path_override=nothing)
    μ = find_chemical_potential(params, δm)
    
    local k_interp_path, labels, ticks
    
    if k_path_override === nothing
        # Use the shared get_high_symmetry_path function
        k_interp_path, labels, ticks = get_high_symmetry_path(params.lattice, nk) # nk here is for npoints in get_high_symmetry_path
    else
        # If k_path_override is provided, interpolate it manually for plot.
        # Note: 'labels' and 'ticks' would need to be provided by the user too for clean plotting
        # if using custom k_path_override and you want labels.
        # For simplicity, if k_path_override is given, we assume it's just raw k-points to interpolate.
        # This will need a better way to handle `labels` and `ticks` for custom paths.
        # For now, let's keep it simple: assume k_path_override provides the exact interpolated path if used.
        # Or, refine `interpolate_kpath` to take labels/ticks.
        # For this module, let's just make `k_interp` a sequence derived from `k_path_override` if provided.

        # Let's adjust this to mirror the BandStructurePlotting's get_high_symmetry_path logic.
        # If the user provides `k_path_override` this function will need to know the labels and ticks too.
        # For now, I'll simplify: if k_path_override is passed, assume it's already interpolated, and no labels/ticks.
        # Or better: make plot_spectral_function accept labels and ticks directly too.

        # To align with plot_band_structure, let's pass a high-symmetry path object.
        # I'll update the signature to accept `(kpath_data, labels, ticks)` if pre-computed.
        # If `k_path_override` is just a list of (kx,ky) tuples, it's ambiguous.
        # Let's rename k_path_override to `high_symmetry_path_info` and expect a tuple `(k_path, labels, ticks)`.

        @warn "Custom k_path_override not fully supported for automatic labels/ticks. Using default high-symmetry path."
        k_interp_path, labels, ticks = get_high_symmetry_path(params.lattice, nk)
    end
    
    # Energy range (relative to μ)
    # The range should be based on the actual band width for better visualization
    # A rough estimate is 2 * (max_t + U/2 + lambda)
    # For now, fixed range is okay.
    ω_min_plot, ω_max_plot = -4.0, 4.0 # Range around Fermi level
    ω_values = range(ω_min_plot, ω_max_plot, length=nω)
    
    # Calculate A(ω,k)
    A_kω = zeros(length(k_interp_path), nω)
    A_up_kω = zeros(length(k_interp_path), nω)
    A_dn_kω = zeros(length(k_interp_path), nω)
    
    for (i,k) in enumerate(k_interp_path)
        for (j,ω_rel) in enumerate(ω_values)
            A_kω[i,j] = spectral_function(params, δm, k, μ + ω_rel, η=η)
            A_up_kω[i,j], A_dn_kω[i,j] = spin_resolved_spectral_function(params, δm, k, μ + ω_rel, η=η)
        end
    end
    
    # Plotting
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(9,8), sharex=true)
    
    # Total spectral function
    # Note on extent: The x-axis for pcolormesh usually takes `x`, `y` coordinates.
    # To map to 1:length(k_interp_path) and `ω_values`, the `extent` argument is useful for proper labeling.
    im1 = ax1.pcolormesh(1:length(k_interp_path), ω_values, A_kω', shading="auto", cmap="gnuplot")
    fig.colorbar(im1, ax=ax1, label=L"$A(\omega,k)$")
    ax1.set_title("Total Spectral Function (δm = $(round(δm, digits=4)), η = $(η))")
    ax1.set_ylabel(L"$\omega - \mu\ (t)$") # Add units to label
    ax1.axhline(0, color="black", linestyle=":", alpha=0.5) # Add Fermi level line
    
    # Spin-up
    im2 = ax2.pcolormesh(1:length(k_interp_path), ω_values, A_up_kω', shading="auto", cmap="Reds")
    fig.colorbar(im2, ax=ax2, label=L"$A_\uparrow(\omega,k)$")
    ax2.set_title("") # No title for subplots
    ax2.set_ylabel(L"$\omega - \mu\ (t)$")
    ax2.axhline(0, color="black", linestyle=":", alpha=0.5)
    
    # Spin-down
    im3 = ax3.pcolormesh(1:length(k_interp_path), ω_values, A_dn_kω', shading="auto", cmap="Blues")
    fig.colorbar(im3, ax=ax3, label=L"$A_\downarrow(\omega,k)$")
    ax3.set_title("")
    ax3.set_ylabel(L"$\omega - \mu\ (t)$")
    ax3.axhline(0, color="black", linestyle=":", alpha=0.5)
    
    # Set k-path labels (using ticks and labels from get_high_symmetry_path)
    ax3.set_xticks(ticks)
    ax3.set_xticklabels(labels)
    ax3.set_xlabel("Wave Vector (k)")
    ax3.set_xlim(1, length(k_interp_path)) # Ensure x-limits match the data range
    
    plt.tight_layout()
    return fig
end



"""
    interpolate_kpath(k_points, nk_per_segment)

Create interpolated k-path through high-symmetry points.
This function is a helper. It might be better to share it from BandStructurePlotting.
"""
function interpolate_kpath(k_points, nk_per_segment)
    k_interp = Tuple{Float64,Float64}[]
    for i in 1:length(k_points)-1
        k1 = k_points[i]
        k2 = k_points[i+1]
        # range(0,1,length=N) creates N points including 0 and 1.
        # We want to avoid duplicating the end point of one segment as the start of the next.
        # So, for the first segment, use `length=nk_per_segment`.
        # For subsequent segments, use `length=nk_per_segment+1` and skip the first point (t=0).
        # Or, just use `length=nk_per_segment` for all, and then manually concatenate with the last point.
        # A simpler way is to use `range(0, 1, length=nk_per_segment)` and then filter unique points.
        segment_k_points = [(k1[1] + t*(k2[1]-k1[1]), k1[2] + t*(k2[2]-k1[2])) for t in range(0, 1, length=nk_per_segment)]
        append!(k_interp, segment_k_points)
    end
    # Ensure unique points. `unique` might reorder, so be careful.
    # It's better to manage the overlaps manually if precise tick positions are needed.
    # For now, your original `get_high_symmetry_path` logic should work fine.
    # This `interpolate_kpath` is actually a re-implementation of part of `get_high_symmetry_path`.
    # It's cleaner to just rely on `get_high_symmetry_path` to produce the full path.
    # I've modified `plot_spectral_function` to call `get_high_symmetry_path` directly.
    return k_interp # This function might become redundant if get_high_symmetry_path is used directly.
end

end # module