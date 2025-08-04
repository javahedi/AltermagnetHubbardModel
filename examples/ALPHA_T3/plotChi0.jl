using PyPlot
const plt = PyPlot
using BSON: @load
using LaTeXStrings
using LinearAlgebra
using AltermagneticHubbardModel  # For ModelParams
using Interpolations  # Added for native Julia interpolation




α = π/4

params = ModelParams(
        lattice = ALPHA_T3,
        t       = 1.0,
        t_prime = 0.0,
        δ       = 0.0,
        U       = 0.0,  # irrelevant at this stage
        λ       = 0.0,
        n       = 1.0,
        β       = 1000.0,
        α       = α,
        kpoints = 30,
        mixing  = 0.4,
        tol     = 1e-6
    )


# Plot χ_0 as a 2D heatmap over the hexagonal FBZ
function plot_chi_0_fbz(qpoints, chi_values, a=1.0)
    # Extract qx, qy
    qx = [q[1] for q in qpoints]
    qy = [q[2] for q in qpoints]
    
    # Define hexagonal FBZ boundary
    b1, b2 = get_reciprocal_vectors(ALPHA_T3)

    K1 = (2/3)*b1 + (1/3)*b2   # First Dirac point
    K2 = (1/3)*b1 + (2/3)*b2   # Second Dirac point (equivalent to K1)
    K3 = (-1/3)*b1 + (1/3)*b2  # Third Dirac point
    K4 = (-2/3)*b1 + (-1/3)*b2 # Fourth Dirac point (equivalent to K1)
    K5 = (-1/3)*b1 + (-2/3)*b2 # Fifth Dirac point
    K6 = (1/3)*b1 + (-1/3)*b2  # Sixth Dirac point (equivalent to K1)
        

      # High-symmetry points
    sym_points = Dict(
        L"K_1" => K1,           # Matches original K'
        L"K_2" => K2,              # (0, 4√3/9)
        L"K_3" => K3,          # Matches original K
        L"K_4" => K4,         # (-2√3/9, -2/3)
        L"K_5" => K5,             # (0, -4√3/9)
        L"K_6" => K6              # (2√3/9, -2/3)
    )

    fbz_vertices = [
      K1, K2, K3, K4, K5, K6
    ]

    
    # Close the hexagon by appending the first vertex
    hex_x = [v[1] for v in fbz_vertices]
    hex_y = [v[2] for v in fbz_vertices]
    push!(hex_x, hex_x[1])
    push!(hex_y, hex_y[1])

  
    
    # Create heatmap via scatter plot
    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(qx, qy, c=chi_values, cmap="viridis", s=25)
    ax.plot(hex_x, hex_y, "k-", linewidth=1)  # FBZ boundary

  
    
    for (label, point) in sym_points
        ax.scatter([point[1]], [point[2]], c="red", s=50, marker="o", edgecolors="black")
        # Offset labels to avoid overlap
        offset_x = label == L"K" ? 0.5 : (label == L"K'" ? -0.5 : 0.0)
        offset_y =  0.3
        ax.text(point[1] + offset_x, point[2] + offset_y, label, fontsize=12, color="black", ha="center")
    end

    ax.set_xlabel(L"$q_x \, (2\pi/a)$")
    ax.set_ylabel(L"$q_y \, (2\pi/a)$")
    ax.set_title(L"$\chi_0(\mathbf{q}, \omega=0)$ for $α=π/4$")
    plt.colorbar(sc, label=L"$\lambda_{\mathrm{max}}[\chi_0^{\mu\nu}(\mathbf{q}, 0)]$ (arb. units)")

    #ax.set_aspect("equal")
    plt.show()
end

# Compute and plot χ_0
δm = 0.0
μ  = find_chemical_potential(params, δm; μ_min=-3.0, μ_max=3.0)

qpoints, chi_values = compute_chi0_leading_eigenvalue_parallel(params, μ, δm; η=0.01)
plot_chi_0_fbz(qpoints, chi_values)