module BandStructurePlotting

using PyPlot
const plt = PyPlot  # Create an alias for PyPlot

using AltermagneticHubbardModel
using LinearAlgebra
using Statistics
using CSV, DataFrames

export plot_band_structure, get_high_symmetry_path, plot_fermi_surface, compute_dos

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
        # Γ -> X -> M -> X2 -> Γ for square lattice
        Γ = [0.0, 0.0]
        X = [-π/2, π/2]
        M = [0, π]
        X2 = [π/2, π/2]

        # Define the k-path
        kpath = vcat(
            [(Γ[1] + t*(X[1]-Γ[1]), Γ[2] + t*(X[2]-Γ[2])) for t in range(0, 1, length=npoints)],
            [(X[1] + t*(M[1]-X[1]), X[2] + t*(M[2]-X[2])) for t in range(0, 1, length=npoints)],
            [(M[1] + t*(X2[1]-M[1]), M[2] + t*(X2[2]-M[2])) for t in range(0, 1, length=npoints)],
            [(X2[1] + t*(Γ[1]-X2[1]), X2[2] + t*(Γ[2]-X2[2])) for t in range(0, 1, length=npoints)]
        )
        
        labels = ["Γ", "X", "M", "X2", "Γ"]
        ticks = [1, npoints, 2npoints, 3npoints, 4npoints]
        
    elseif lattice == HEXATRIANGULAR || lattice == ALPHA_T3
        
        # δ1, δ2, δ3 = [0.0, -1.0], [√3/2, 0.5], [-√3/2, 0.5]
        # a1 = δ1 - δ2 = a (√3/2, 3/2)
        # a2 = δ2 - δ3 = a (-√3/2, 3/2)

        # b1 = (2π√3/3, 2π/3)
        # b2 = (-2π√3/3, 2π/3)

        # Γ = (0.0, 0.0)
        # M1 = b1/2 = (√3π/3, π/3)
        # M2 = b2/2 = (-√3π/3, π/3)
        # K = (1/3) b1 + (2/3) b2 = (-2π√3/9, 2π/3)
        # K' = (2/3) b1 + (1/3) b2 = (2π√3/9, 2π/3)

        # Γ -> M1 -> K -> M2 -> Γ for honeycomb

        Γ = [0.0, 0.0]
        K = [-2π√3/9, 2π/3]
        Kp = [2π√3/9, 2π/3]
        M1 = [√3π/3, π/3]
        M2 = [-√3π/3, π/3]
        


        kpath = vcat(
            [(Γ[1] + t*(M1[1]-Γ[1]), Γ[2] + t*(M1[2]-Γ[2])) for t in range(0, 1, length=npoints)],
            [(M1[1] + t*(K[1]-M1[1]), M1[2] + t*(K[2]-M1[2])) for t in range(0, 1, length=npoints)],
            [(K[1] + t*(M2[1]-K[1]), K[2] + t*(M2[2]-K[2])) for t in range(0, 1, length=npoints)],
            [(M2[1] + t*(Γ[1]-M2[1]), M2[2] + t*(Γ[2]-M2[2])) for t in range(0, 1, length=npoints)]
        )

        labels = ["Γ", "M1", "K", "M2", "Γ"]
        ticks = [1, npoints, 2npoints, 3npoints, 4npoints]

    else
        error("Unsupported lattice for k-path: $lattice")
    end
    
    return (kpath, labels, ticks)
end



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

function plot_band_structure(params::ModelParams, δm::Float64; 
                            npoints::Int=100, savepath=nothing, showlegend=true)
    kpath, labels, ticks = get_high_symmetry_path(params.lattice, npoints)


    # Determine system size based on lattice
    
    if params.lattice == HEXATRIANGULAR || params.lattice == ALPHA_T3
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
    fig, ax = plt.subplots(figsize=(5,4))
    for b in 1:nbands
        ax.plot(1:length(kpath), ϵ_up[:,b], "r-", label=b==1 ? "Spin ↑" : "")
        ax.plot(1:length(kpath), ϵ_dn[:,b], "b-", label=b==1 ? "Spin ↓" : "")
    end
    


    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.set_xlim(1, length(kpath))

    ax.axhline(0, color="black", linestyle=":", alpha=0.5)  # Fermi level
    ax.set_xlabel(L"$k$")
    ax.set_ylabel(L"$E(k)$")
    ax.set_title("U = $(round(params.U, digits=2)), α = $(round(params.α, digits=2)), δm = $(round(δm, digits=4))")
    showlegend && ax.legend()
    ax.grid(alpha=0.3)

    if savepath !== nothing
        fig.savefig(savepath, dpi=300, bbox_inches="tight")
    end

    export_band_data("band_data", kpath, ϵ_up, ϵ_dn)

    
    return fig
end
    

 

function plot_fermi_surface(params::ModelParams, δm::Float64; nk=100)
    b1, b2 = get_reciprocal_vectors(params.lattice)
    kx_vals = range(0, stop=2π, length=nk)
    ky_vals = range(0, stop=2π, length=nk)
    
    fs_up = zeros(nk, nk)
    fs_dn = zeros(nk, nk)
    
    matrix_size = size(build_hamiltonian((0.0,0.0), params, δm), 1)
    half = matrix_size ÷ 2  # Assuming spin blocks are half the matrix
    
    for (ix, kx) in enumerate(kx_vals), (iy, ky) in enumerate(ky_vals)
        k = (kx, ky)
        H = build_hamiltonian(k, params, δm)
        
        # Extract spin-up and spin-down blocks
        H_up = Hermitian(H[1:half, 1:half])
        H_dn = Hermitian(H[half+1:end, half+1:end])
        
        E_up = eigvals(H_up)
        E_dn = eigvals(H_dn)
        
        fs_up[iy, ix] = minimum(abs.(E_up))
        fs_dn[iy, ix] = minimum(abs.(E_dn))
    end

    figure(figsize=(4,3))
    contour(kx_vals, ky_vals, fs_up, levels=[0.09], colors="red", linewidths=2, label="Spin ↑")
    contour(kx_vals, ky_vals, fs_dn, levels=[0.09], colors="blue", linewidths=2, label="Spin ↓")

    xlabel(L"k_x")
    ylabel(L"k_y")
    title("U = $(round(params.U, digits=2)), α = $(round(params.α, digits=2)), δm = $(round(δm, digits=4))")

    axis("equal")
    
    # Add legend manually because contour doesn't handle it directly
    plot([], [], "r-", label="Spin ↑")
    plot([], [], "b-", label="Spin ↓")
    legend()
    
    show()
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
    
    hist, edges = fit(Histogram, E_all, nbins)
    centers = 0.5 .* (edges[1][1:end-1] .+ edges[1][2:end])
    
    figure()
    plot(centers, hist.weights ./ length(kpoints), label="DOS")
    xlabel("Energy (t)")
    ylabel("DOS")
    title("Density of States")
    axvline(0.0, linestyle="--", color="gray", label="Fermi Level")
    legend()
end


end # module