module FermiSurface

using PyPlot
const plt = PyPlot  # Create an alias for PyPlot
using AltermagneticHubbardModel
using LinearAlgebra
export plot_fermi_surface_comparison

"""
    plot_fermi_surface_comparison(params::ModelParams, δm::Float64; nk::Int=200, μ_tol::Float64=0.05)

Plot Fermi surface with:
- Gray filled area: Non-magnetic (δm=0) Fermi surface
- Red/blue boundaries: Spin-up/down Fermi surfaces for δm≠0
"""
function plot_fermi_surface_comparison(params::ModelParams, δm::Float64; nk::Int=50, μ_tol::Float64=0.05)
    μ = find_chemical_potential(params, δm)
    kx = range(-π+0.1, π+0.1, length=nk)
    ky = range(-π+0.1, π+0.1, length=nk)
    
    # Initialize energy grids
    ϵ_nm = zeros(nk, nk)   # Non-magnetic
    ϵ_up = zeros(nk, nk)   # Spin up
    ϵ_dn = zeros(nk, nk)   # Spin down

    # Calculate energy surfaces
    for (i,kx_val) in enumerate(kx), (j,ky_val) in enumerate(ky)
        k = (kx_val, ky_val)
        H = build_hamiltonian(k, params, δm)
        
        # Explicitly extract spin blocks
        H_up = H[1:2, 1:2]   # ↑ block (A↑, B↑)
        H_dn = H[3:4, 3:4]   # ↓ block (A↓, B↓)
        
        # Non-magnetic reference (δm=0)
        eigvals_nm = sort(eigvals(Hermitian(H)))
        # Mark region where bands are occupied
        ϵ_nm[i,j] = sum(eigvals_nm .< μ) / 2.0
        #println("k = ($kx_val, $ky_val), μ = $μ, ϵ_nm = $(ϵ_nm[i,j])")
        #println("eigvals_nm = $eigvals_nm")


        # Spin-resolved
        ϵ_up[i,j] = minimum(eigvals(Hermitian(H_up))) - μ
        ϵ_dn[i,j] = minimum(eigvals(Hermitian(H_dn))) - μ
    end

    # Create plot
    fig, ax = plt.subplots(figsize=(8,6))
    
    # Plot non-magnetic Fermi surface as filled gray area
    ax.pcolormesh(kx, ky, ϵ_nm', cmap="gray", shading="auto", alpha=0.5)
    


    # 👉 Add Fermi surface as contour at ε = μ = 0
    cs_nm = ax.contour(kx, ky, ϵ_nm', levels=[0.0], colors="black", 
                      linewidths=2, linestyles="solid", label="Non-magnetic FS")


    cs_up = ax.contour(kx, ky, ϵ_up', levels=[0.0], 
                          colors=["red"], linewidths=2)
    cs_dn = ax.contour(kx, ky, ϵ_dn', levels=[0.0], 
                          colors=["blue"], linewidths=2)

    
        

    # Add BZ boundary
    #bz_points = [(π,π), (-π,π), (-π,-π), (π,-π), (π,π)]
    #ax.plot([p[1] for p in bz_points], [p[2] for p in bz_points], 
    #        "k--", linewidth=1, alpha=0.5)

  


    # Formatting
    #ax.set_title(δm ≠ 0 ? "Fermi Surface (δm = $(round(δm,digits=3))" : "Non-magnetic Fermi Surface")
    ax.set_xlabel("kₓ")
    ax.set_ylabel("kᵧ")
    #ax.set_xlim(-π, π)
    #ax.set_ylim(-π, π)
    ax.set_aspect("equal")
    ax.grid(alpha=0.2)

    return fig
end

end