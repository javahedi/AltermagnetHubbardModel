module BandStructurePlotting

using PyPlot
const plt = PyPlot  # Create an alias for PyPlot

using AltermagneticHubbardModel
using LinearAlgebra
using Statistics
using StatsBase
using CSV, DataFrames
using Interpolations

export plot_band_structure, plot_fermi_surface, compute_dos
export plot_color_coded_band_structure, plot_spectral_function2, plot_chi_0_fbz
export plot_fermi_surface_imshow




function export_band_data(path::String, kpath::Vector{Tuple{Float64,Float64}},
                          ε_up::Matrix{Float64}, ε_dn::Matrix{Float64})

    df = DataFrame(kx = [k[1] for k in kpath],
                   ky = [k[2] for k in kpath])

    for b in 1:size(ε_up, 2)
        df[!, "ε_up_$b"] = ε_up[:, b]
        df[!, "ε_dn_$b"] = ε_dn[:, b]
    end

    CSV.write("$path.csv", df)
    
end




"""
    plot_band_structure(params::ModelParams, δm::Float64; npoints::Int=100)

Plot spin-resolved band structure along high-symmetry path.
"""

function plot_band_structure(params::ModelParams, δm::Float64, μ::Float64=0.0; 
                            npoints::Int=100, savepath=nothing, showlegend=true)
    kpath, labels, ticks = get_high_symmetry_path(params.lattice, npoints)


    # Determine system size based on lattice
    
    if params.lattice == HEXATRIANGULAR || params.lattice == ALPHA_T3 
        matrix_size = 6  # 3 sublattices × 2 spins
        nbands = 3       # 2 bands per spin
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

    # Shift energies so μ is at zero
    #ϵ_up_shifted = ϵ_up .- μ
    #ϵ_dn_shifted = ϵ_dn .- μ
    
    #println(" size ϵ_up: ", size(ϵ_up))
    #println(" size ϵ_dn: ", size(ϵ_dn))

    # Plotting (same as before)
    fig, ax = plt.subplots(figsize=(5,4))
    for b in 1:nbands
        ax.plot(1:length(kpath), ϵ_up[:,b], "r-", label=b==1 ? "Spin ↑" : "")
        ax.plot(1:length(kpath), ϵ_dn[:,b], "b-", label=b==1 ? "Spin ↓" : "") 
    end
    


    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.set_xlim(1, length(kpath))

    ax.axhline(μ, color="black", linestyle=":", alpha=0.5)  # Fermi level
    ax.set_xlabel(L"$k$")
    ax.set_ylabel(L"$E(k)$")
    ax.set_title("U = $(round(params.U, digits=2)), α = $(round(params.α, digits=2)), δm = $(round(δm, digits=4))")
    showlegend && ax.legend()
    ax.grid(alpha=0.3)

   
    export_band_data("band_data", kpath, ϵ_up, ϵ_dn)



    # ΔE(k) plot: band-resolved spin splitting
    fig2, ax2 = plt.subplots(figsize=(5,3.5))
    #for b in 1:nbands
    #    ΔE_b = ϵ_up[:, b] .- ϵ_dn[:, b]
    #    ax2.plot(1:length(kpath), ΔE_b, label="Band $b")
    #end

    #ax2.axhline(0, color="gray", linestyle="--", alpha=0.5)
    #ax2.set_xticks(ticks)
    #ax2.set_xticklabels(labels)
    #ax2.set_xlim(1, length(kpath))

    #ax2.set_xlabel(L"$k$")
    #ax2.set_ylabel(L"$\Delta E(k)$")
    #ax2.set_title("Spin Splitting ΔE(k) per Band")
    #ax2.legend()
    #ax2.grid(alpha=0.3)

  
    # --- Save both figures ---
    if savepath !== nothing
        fig.savefig(savepath, dpi=300, bbox_inches="tight")
        #fig2.savefig(replace(savepath, ".png" => "_splitting.png"), dpi=300, bbox_inches="tight")
    end 

    
    return fig, fig2
end
    


"""
    compute_dos(params::ModelParams, δm::Float64; nk=100, nbins=200)
Compute and plot the density of states (DOS) for the given model parameters and magnetization.
"""
function compute_dos(params::ModelParams, δm::Float64; nk=100, nbins=200)
    kpoints = generate_kpoints(params.lattice, nk)
    E_all = Float64[]
    
    for k in kpoints
        H = build_hamiltonian(k, params, δm)
        append!(E_all, real(eigvals(Hermitian(H))))
    end
    
    hist, edges = StatsBase.fit(Histogram, E_all, nbins)
    centers = 0.5 .* (edges[1][1:end-1] .+ edges[1][2:end])
    
    figure()
    plot(centers, hist.weights ./ length(kpoints), label="DOS")
    xlabel("Energy (t)")
    ylabel("DOS")
    title("Density of States")
    axvline(0.0, linestyle="--", color="gray", label="Fermi Level")
    legend()
end



function plot_color_coded_band_structure(params::ModelParams, δm::Float64, μ::Float64=0.0; 
                                       npoints::Int=100, nk_dos::Int=100, nbins::Int=200, 
                                       savepath=nothing, showlegend=true)
    # Get high symmetry path for band structure
    kpath, labels, ticks = get_high_symmetry_path(params.lattice, npoints)

    # Determine system size based on lattice
    if params.lattice == HEXATRIANGULAR 
        matrix_size = 6  # 3 sublattices × 2 spins
        nbands = 3       # 3 bands per spin
    else
        matrix_size = 4  # 2 sublattices × 2 spins
        nbands = 2       # 2 bands per spin
    end

    # Compute DOS
    kpoints = generate_kpoints(params.lattice, nk_dos)
    E_all = Float64[]
    for k in kpoints
        H = build_hamiltonian(k, params, δm)
        append!(E_all, real(eigvals(Hermitian(H))))
    end
    
    # Define bin edges for histogram
    E_min, E_max = minimum(E_all), maximum(E_all)
    edges = range(E_min, E_max, length=nbins+1)
    hist = fit(Histogram{Float64}, E_all, edges)
    centers = 0.5 .* (hist.edges[1][1:end-1] .+ hist.edges[1][2:end])
    dos = hist.weights ./ length(kpoints)
    
    # Normalize DOS for coloring
    dos_max = maximum(dos)
    dos_normalized = dos ./ dos_max

    # Storage for band energies
    ϵ_up = zeros(length(kpath), nbands)
    ϵ_dn = zeros(length(kpath), nbands)
    
    # Compute band energies
    for (i,k) in enumerate(kpath)
        H = build_hamiltonian(k, params, δm)
        H_up = H[1:matrix_size÷2, 1:matrix_size÷2]
        H_dn = H[matrix_size÷2+1:end, matrix_size÷2+1:end]
        ϵ_up[i,:] = sort(real(eigvals(Hermitian(H_up)))[1:nbands])
        ϵ_dn[i,:] = sort(real(eigvals(Hermitian(H_dn)))[1:nbands])
    end

    # Create figure
    fig, ax = plt.subplots(figsize=(6,4))
    
    # Color-code bands based on DOS with separate colormaps
    cmap_up = plt.get_cmap("Reds")   # Red colormap for spin-up
    cmap_dn = plt.get_cmap("Blues")  # Blue colormap for spin-down
    for b in 1:nbands
        # Interpolate DOS values for band energies
        dos_up = zeros(length(kpath))
        dos_dn = zeros(length(kpath))
        for i in 1:length(kpath)
            # Find nearest DOS value for each energy
            idx_up = argmin(abs.(centers .- ϵ_up[i,b]))
            idx_dn = argmin(abs.(centers .- ϵ_dn[i,b]))
            dos_up[i] = dos_normalized[idx_up]
            dos_dn[i] = dos_normalized[idx_dn]
        end
        
        # Plot spin-up bands with Reds colormap
        points = hcat(collect(1:length(kpath)), ϵ_up[:,b])
        segments = [points[i:i+1,:] for i in 1:size(points,1)-1]
        lc_up = matplotlib.collections.LineCollection(segments, cmap=cmap_up, norm=plt.matplotlib.colors.Normalize(0,1))
        lc_up.set_array(dos_up[1:end-1])
        ax.add_collection(lc_up)
        
        # Plot spin-down bands with Blues colormap
        points = hcat(collect(1:length(kpath)), ϵ_dn[:,b])
        segments = [points[i:i+1,:] for i in 1:size(points,1)-1]
        lc_dn = matplotlib.collections.LineCollection(segments, cmap=cmap_dn, norm=plt.matplotlib.colors.Normalize(0,1))
        lc_dn.set_array(dos_dn[1:end-1])
        ax.add_collection(lc_dn)
        
        # Plot invisible lines for legend
        ax.plot([], [], color=cmap_up(0.5), label=b==1 ? "Spin ↑" : "")
        ax.plot([], [], color=cmap_dn(0.5), linestyle="--", label=b==1 ? "Spin ↓" : "")
    end

    # Add colorbars for both colormaps
    sm_up = plt.cm.ScalarMappable(cmap=cmap_up, norm=plt.matplotlib.colors.Normalize(0, dos_max))
    cbar_up = fig.colorbar(sm_up, ax=ax, label="DOS Spin ↑ (arb. units)", pad=0.01)
    
    sm_dn = plt.cm.ScalarMappable(cmap=cmap_dn, norm=plt.matplotlib.colors.Normalize(0, dos_max))
    cbar_dn = fig.colorbar(sm_dn, ax=ax, label="DOS Spin ↓ (arb. units)", pad=0.08)
    
    # Set plot properties
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.set_xlim(1, length(kpath))
    ax.axhline(μ, color="black", linestyle=":", alpha=0.5)
    ax.set_xlabel(L"$k$")
    ax.set_ylabel(L"$E(k)$")
    ax.set_title("U = $(round(params.U, digits=2)), α = $(round(params.α, digits=2)), δm = $(round(δm, digits=4))")
    showlegend && ax.legend()
    ax.grid(alpha=0.3)

    # Save figure if requested
    if savepath !== nothing
        fig.savefig(savepath, dpi=300, bbox_inches="tight")
    end

    return fig
end




function plot_spectral_function2(params::ModelParams, δm::Float64, μ::Float64=0.0; 
                               npoints::Int=100, nomega::Int=200, eta::Float64=0.05, 
                               savepath=nothing, showlegend=true)
    # Get high symmetry path for band structure
    kpath, labels, ticks = get_high_symmetry_path(params.lattice, npoints)

    # Determine system size based on lattice
    if params.lattice == HEXATRIANGULAR || params.lattice == ALPHA_T3
        matrix_size = 6  # 3 sublattices × 2 spins
        nbands = 3       # 3 bands per spin
    else
        matrix_size = 4  # 2 sublattices × 2 spins
        nbands = 2       # 2 bands per spin
    end

    # Storage for band energies
    ϵ_up = zeros(length(kpath), nbands)
    ϵ_dn = zeros(length(kpath), nbands)
    
    # Compute band energies
    for (i, k) in enumerate(kpath)
        H = build_hamiltonian(k, params, δm)
        H_up = H[1:matrix_size÷2, 1:matrix_size÷2]
        H_dn = H[matrix_size÷2+1:end, matrix_size÷2+1:end]
        ϵ_up[i, :] = sort(real(eigvals(Hermitian(H_up)))[1:nbands])
        ϵ_dn[i, :] = sort(real(eigvals(Hermitian(H_dn)))[1:nbands])
    end

    # Define energy range for spectral function
    E_min = minimum([minimum(ϵ_up), minimum(ϵ_dn)]) - 2 * eta
    E_max = maximum([maximum(ϵ_up), maximum(ϵ_dn)]) + 2 * eta
    ω = range(E_min, E_max, length=nomega)
    
    # Compute spectral function A(k, ω) using Lorentzian broadening
    A_up = zeros(length(kpath), nomega)
    A_dn = zeros(length(kpath), nomega)
    for i in 1:length(kpath)
        for b in 1:nbands
            for j in 1:nomega
                # Lorentzian: A(k, ω) = (1/π) * (η / ((ω - ϵ(k))^2 + η^2))
                A_up[i, j] += (1 / π) * (eta / ((ω[j] - ϵ_up[i, b])^2 + eta^2))
                A_dn[i, j] += (1 / π) * (eta / ((ω[j] - ϵ_dn[i, b])^2 + eta^2))
            end
        end
    end

    # Normalize spectral functions for visualization
    A_up_max = maximum(A_up)
    A_dn_max = maximum(A_dn)
    A_up ./= A_up_max
    A_dn ./= A_dn_max

    # Create figure with two subplots
    fig, (ax_up, ax_dn) = plt.subplots(1, 2, figsize=(10, 4), sharey=true)

    # Plot spin-up spectral function (Reds)
    cmap_up = plt.get_cmap("Reds")
    im_up = ax_up.imshow(A_up', aspect="auto", origin="lower", 
                         extent=[1, length(kpath), E_min, E_max], 
                         cmap=cmap_up, interpolation="bilinear")
    ax_up.set_xticks(ticks)
    ax_up.set_xticklabels(labels)
    ax_up.set_xlim(1, length(kpath))
    ax_up.set_ylabel(L"$\omega$ (t)")
    ax_up.set_title("Spin ↑ (U = $(round(params.U, digits=2)), α = $(round(params.α, digits=2)), δm = $(round(δm, digits=4)))")
    ax_up.axhline(0, color="white", linestyle=":", alpha=0.5)  # Fermi level
    fig.colorbar(im_up, ax=ax_up, label="A(k, ω) Spin ↑ (norm.)")

    # Plot spin-down spectral function (Blues)
    cmap_dn = plt.get_cmap("Blues")
    im_dn = ax_dn.imshow(A_dn', aspect="auto", origin="lower", 
                         extent=[1, length(kpath), E_min, E_max], 
                         cmap=cmap_dn, interpolation="bilinear")
    ax_dn.set_xticks(ticks)
    ax_dn.set_xticklabels(labels)
    ax_dn.set_xlim(1, length(kpath))
    ax_dn.set_title("Spin ↓")
    ax_dn.axhline(μ, color="white", linestyle=":", alpha=0.5)  # Fermi level
    fig.colorbar(im_dn, ax=ax_dn, label="A(k, ω) Spin ↓ (norm.)")

    # Shared x-label
    fig.text(0.5, 0.01, L"$k$", ha="center", va="center")

    # Adjust layout
    plt.tight_layout()

    # Save figure if requested
    if savepath !== nothing
        fig.savefig(savepath, dpi=300, bbox_inches="tight")
    end

    return fig
end


function plot_fermi_surface(params::ModelParams, δm::Float64, μ::Float64=0.0; 
                            nk::Int=100, savepath=nothing)
    # Determine system size based on lattice
    if params.lattice == HEXATRIANGULAR || params.lattice == ALPHA_T3
        matrix_size = 6  # 3 sublattices × 2 spins
        nbands = 3       # 3 bands per spin
    else
        matrix_size = 4  # 2 sublattices × 2 spins
        nbands = 2       # 2 bands per spin
    end

    # Generate k-points using provided function
    kpoints = generate_kpoints(params.lattice, nk)
    # Reshape k-points into a 2D grid for contour plotting
    kx = zeros(nk, nk)
    ky = zeros(nk, nk)
    for idx in eachindex(kpoints)
        i = div(idx-1, nk) + 1
        j = mod(idx-1, nk) + 1
        kx[i,j] = kpoints[idx][1]
        ky[i,j] = kpoints[idx][2]
    end

    # Define FBZ boundary using reciprocal lattice vectors
    b1, b2 = get_reciprocal_vectors(params.lattice)
    if params.lattice == HEXATRIANGULAR || params.lattice == ALPHA_T3
        # Hexagonal FBZ: vertices at ±b1/2, ±b2/2, ±(b1-b2)/2, etc.
        fbz_vertices = [
            0.5*b1, 0.5*(b1+b2), 0.5*b2,
            -0.5*b1, -0.5*(b1+b2), -0.5*b2,
            0.5*b1  # Close the loop
        ]
    else
        # Square FBZ: vertices at (±π, ±π)
        fbz_vertices = [
            [-π, -π], [-π, π], [π, π], [π, -π], [-π, -π]
        ]
    end

    # Compute band energies over the k-grid
    ϵ_up = zeros(nk, nk, nbands)
    ϵ_dn = zeros(nk, nk, nbands)
    for idx in eachindex(kpoints)
        i = div(idx-1, nk) + 1
        j = mod(idx-1, nk) + 1
        k = (kx[i,j], ky[i,j])
        H = build_hamiltonian(k, params, δm)
        H_up = H[1:matrix_size÷2, 1:matrix_size÷2]
        H_dn = H[matrix_size÷2+1:end, matrix_size÷2+1:end]
        ϵ_up[i,j,:] = sort(real(eigvals(Hermitian(H_up)))[1:nbands])
        ϵ_dn[i,j,:] = sort(real(eigvals(Hermitian(H_dn)))[1:nbands])
    end

    # Create figure
    fig, ax = plt.subplots(figsize=(5, 3))

    # Plot Fermi surface contours for each band
    for b in 1:nbands
        # Spin-up contours (red)
        ax.contour(kx, ky, ϵ_up[:,:,b], levels=[μ], colors="red", linestyles="-", linewidths=1.5)
        # Spin-down contours (blue)
        ax.contour(kx, ky, ϵ_dn[:,:,b], levels=[μ], colors="blue", linestyles="--", linewidths=1.5)
    end

    # Plot FBZ boundary
    #fbz_x = [v[1] for v in fbz_vertices]
    #fbz_y = [v[2] for v in fbz_vertices]
    #ax.plot(fbz_x, fbz_y, color="black", linestyle="-", linewidth=1)

    # Plot reciprocal lattice vectors b1 and b2 as arrows
    #ax.arrow(0, 0, b1[1], b1[2], color="green", width=0.05, head_width=0.15, head_length=0.2, label="b₁")
    #ax.arrow(0, 0, b2[1], b2[2], color="darkgreen", width=0.05, head_width=0.15, head_length=0.2, label="b₂")

    # Set plot properties
    ax.set_xlabel(L"$k_x$")
    ax.set_ylabel(L"$k_y$")
    #ax.set_title("Fermi Surface: U = $(round(params.U, digits=2)), α = $(round(params.α, digits=2)), δm = $(round(δm, digits=4)), μ = $(round(μ, digits=2))")
    ax.legend()
    #ax.set_aspect("equal")
    ax.grid(alpha=0.3)

    # Save figure if requested
    if savepath !== nothing
        fig.savefig(savepath, dpi=300, bbox_inches="tight")
    end

    return fig
end



function plot_fermi_surface_imshow(params::ModelParams, δm::Float64, μ::Float64=-1.26111; 
                                   nk::Int=100, savepath=nothing, band_index::Int=1)
    # Determine system size based on lattice
    if params.lattice == HEXATRIANGULAR || params.lattice == ALPHA_T3
        matrix_size = 6  # 3 sublattices × 2 spins
        nbands = 3       # 3 bands per spin
    else
        matrix_size = 4  # 2 sublattices × 2 spins
        nbands = 2       # 2 bands per spin
    end

    # Validate band_index
    if band_index < 1 || band_index > nbands
        error("Band index $band_index is out of range (1 to $nbands)")
    end

    # Generate k-points using provided function
    kpoints = generate_kpoints(params.lattice, nk)
    # Reshape k-points into a 2D grid for plotting
    kx = zeros(nk, nk)
    ky = zeros(nk, nk)
    for idx in eachindex(kpoints)
        i = div(idx-1, nk) + 1
        j = mod(idx-1, nk) + 1
        kx[i,j] = kpoints[idx][1]
        ky[i,j] = kpoints[idx][2]
    end

    # Define FBZ boundary using reciprocal lattice vectors
    b1, b2 = get_reciprocal_vectors(params.lattice)
    if params.lattice == HEXATRIANGULAR || params.lattice == ALPHA_T3
        # Hexagonal FBZ: vertices at ±b1/2, ±b2/2, ±(b1-b2)/2, etc.
        fbz_vertices = [
            0.5*b1, 0.5*(b1+b2), 0.5*b2,
            -0.5*b1, -0.5*(b1+b2), -0.5*b2,
            0.5*b1  # Close the loop
        ]
    else
        # Square FBZ: vertices at (±π, ±π)
        fbz_vertices = [
            [-π, -π], [-π, π], [π, π], [π, -π], [-π, -π]
        ]
    end

    # Compute band energies over the k-grid
    ϵ_up = zeros(nk, nk, nbands)
    ϵ_dn = zeros(nk, nk, nbands)
    for idx in eachindex(kpoints)
        i = div(idx-1, nk) + 1
        j = mod(idx-1, nk) + 1
        k = (kx[i,j], ky[i,j])
        H = build_hamiltonian(k, params, δm)
        H_up = H[1:matrix_size÷2, 1:matrix_size÷2]
        H_dn = H[matrix_size÷2+1:end, matrix_size÷2+1:end]
        ϵ_up[i,j,:] = sort(real(eigvals(Hermitian(H_up)))[1:nbands])
        ϵ_dn[i,j,:] = sort(real(eigvals(Hermitian(H_dn)))[1:nbands])
    end

    # Diagnostic: Check band energy ranges and proximity to μ
    crosses_μ = false
    for b in 1:nbands
        min_up, max_up = minimum(ϵ_up[:,:,b]), maximum(ϵ_up[:,:,b])
        min_dn, max_dn = minimum(ϵ_dn[:,:,b]), maximum(ϵ_dn[:,:,b])
        println("Band $b (Spin ↑): min = $min_up, max = $max_up")
        println("Band $b (Spin ↓): min = $min_dn, max = $max_dn")
        if min_up <= μ <= max_up
            println("Band $b (Spin ↑): μ = $μ is within the band range")
            crosses_μ = true
        end
        if min_dn <= μ <= max_dn
            println("Band $b (Spin ↓): μ = $μ is within the band range")
            crosses_μ = true
        end
    end
    println("Chemical potential μ = $μ")
    if !crosses_μ
        println("Warning: μ = $μ does not cross any bands. This may indicate μ is in a band gap.")
        println("Try adjusting μ to a value within the band ranges above (e.g., -2.0, 0.0).")
    end

    # Prepare data for imshow: E - μ for spin-up, -(E - μ) for spin-down
    data_up = ϵ_up[:,:,band_index] .- μ
    data_dn = -(ϵ_dn[:,:,band_index] .- μ)  # Negate for opposite color
    # Combine data: positive for spin-up, negative for spin-down
    data = data_up + data_dn
    # Set color range symmetric around zero
    #vmax = max(maximum(abs.(data_up)), maximum(abs.(data_dn)))
    #vmin = -vmax
    vmin = -1.0  # Adjust as needed
    vmax = 1.0   # Adjust as needed

    # Create figure
    fig, ax = plt.subplots(figsize=(6, 6))

    # Plot color-coded energy map with imshow
    im = ax.imshow(data, origin="lower", cmap="bwr", vmin=vmin, vmax=vmax,
                   extent=(minimum(kx), maximum(kx), minimum(ky), maximum(ky)))
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=ax, label=L"$E_{\uparrow} - \mu - (E_{\downarrow} - \mu)$")
    cbar.ax.set_ylabel(L"$E - E_F$ (eV)", rotation=270, labelpad=15)

    # Plot FBZ boundary
    #fbz_x = [v[1] for v in fbz_vertices]
    #fbz_y = [v[2] for v in fbz_vertices]
    #ax.plot(fbz_x, fbz_y, color="black", linestyle="-", linewidth=1)

    # Plot reciprocal lattice vectors b1 and b2 as arrows
    #ax.arrow(0, 0, b1[1], b1[2], color="green", width=0.05, head_width=0.15, head_length=0.2)
    #ax.arrow(0, 0, b2[1], b2[2], color="darkgreen", width=0.05, head_width=0.15, head_length=0.2)

    # Add legend using proxy artists
    #proxy1 = plt.matplotlib.patches.Patch(color="red", label="Spin ↑ (E - μ)")
    #proxy2 = plt.matplotlib.patches.Patch(color="blue", label="Spin ↓ (-(E - μ))")
    #proxy3 = plt.matplotlib.lines.Line2D([0], [0], color="black", linestyle="-", linewidth=1, label="FBZ")
    #proxy4 = plt.matplotlib.lines.Line2D([0], [0], color="green", linestyle="-", linewidth=2, label="b₁")
    #proxy5 = plt.matplotlib.lines.Line2D([0], [0], color="darkgreen", linestyle="-", linewidth=2, label="b₂")
    #ax.legend(handles=[proxy1, proxy2, proxy3, proxy4, proxy5], loc="upper right")

    # Set plot properties
    ax.set_xlabel(L"$k_x$")
    ax.set_ylabel(L"$k_y$")
    #ax.set_title("Energy Map (Band $band_index): U = $(round(params.U, digits=2)), α = $(round(params.α, digits=2)), δm = $(round(δm, digits=4)), μ = $(round(μ, digits=2))")
    #ax.set_aspect("equal")
    #ax.grid(alpha=0.3)

    # Save figure if requested
    if savepath !== nothing
        fig.savefig(savepath, dpi=300, bbox_inches="tight")
    end

    return fig
end




"""
    plot_chi_0_fbz(qpoints::Vector{Tuple{Float64,Float64}}, 
                   chi_values::Vector{Float64}; 
                   a::Float64=1.0)

Plot χ₀(q) as a 2D scatter heatmap over the square-lattice first Brillouin zone.
"""
function plot_chi_0_fbz(qpoints, chi_values; a=1.0)
    # Extract components
    qx = [q[1] for q in qpoints]
    qy = [q[2] for q in qpoints]
    
    # Define square FBZ boundary using reciprocal vectors
    b1, b2 = get_reciprocal_vectors(SQUARE)

    # Define FBZ corners (square)
    fbz_vertices = [
        0.5b1 + 0.5b2,
        0.5b1 - 0.5b2,
       -0.5b1 - 0.5b2,
       -0.5b1 + 0.5b2,
    ]
    
    # Close boundary polygon
    hex_x = [v[1] for v in fbz_vertices]
    hex_y = [v[2] for v in fbz_vertices]
    push!(hex_x, hex_x[1])
    push!(hex_y, hex_y[1])
    
    # Plot heatmap
    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(qx, qy, c=chi_values, cmap="viridis", s=25)
    ax.plot(hex_x, hex_y, "k-", linewidth=1.0)  # FBZ boundary
    
    ax.set_xlabel(L"$q_x \, (2\pi/a)$")
    ax.set_ylabel(L"$q_y \, (2\pi/a)$")
    ax.set_title(L"$\chi_0(\mathbf{q}, \omega=0)$ for $\alpha = \pi/8$")
    plt.colorbar(sc, label=L"$\lambda_{\mathrm{max}}[\chi_0^{\mu\nu}(\mathbf{q}, 0)]$ (arb. units)")
    
    plt.tight_layout()
    plt.show()
end

end # module