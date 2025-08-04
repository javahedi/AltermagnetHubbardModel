using PyPlot
const plt = PyPlot
using BSON: @load
using LaTeXStrings
using AltermagneticHubbardModel  # For ModelParams

# Load the n-U phase diagram data
@load "examples/phase_diagram_alpha_0.785_2025-07-24_144209.bson" save_data

# Parameters
U_target = 4.0  # Choose your U value
nk = 200        # k-point grid density
η = 0.01        # Berry curvature broadening
β = 1000.0       # Inverse temperature

# Extract data for the target U
n_vals = save_data[:n_vals]
U_vals = save_data[:U_vals]
δm_matrix = save_data[:δm_matrix]
μ_matrix = save_data[:μ_matrix]

# Find index of closest U value
U_idx = argmin(abs.(U_vals .- U_target))
println("Using U = $(U_vals[U_idx]) (closest to target $U_target)")

# Initialize AHE array
σ_xy      = zeros(length(n_vals))
σ_xy_kubo = zeros(length(n_vals))

# Compute AHE for each n
for (i, n) in enumerate(n_vals)
    μ = μ_matrix[i, U_idx]
    δm = δm_matrix[i, U_idx]

    println("Computing AHE for n = $n, U = $(U_vals[U_idx]), δm = $δm, μ = $μ")
    
    # Skip failed SCMF points
    if isnan(μ) || isnan(δm)
        σ_xy[i] = NaN
        continue
    end
    
    # Set up params (use original α and other parameters from data)
    params = ModelParams(
        lattice = ALPHA_T3,
        t = 1.0,
        t_prime = 0.0,
        δ       = 0.0,
        U = U_vals[U_idx],
        λ       = 0.0,
        n = n,
        β = β,
        α = save_data[:α],  # Preserve original α
        kpoints = nk,
        mixing = 0.4,
        tol = 1e-6
    )    

    σ_xy[i]      = compute_AHE(params, δm, μ; η=η)
    σ_xy_kubo[i] = compute_AHE_kubo(params, δm, μ; η=η) 

    # save csv  
    if i == 1
        open("AHE_results.csv", "w") do file
            write(file, "n,σ_xy,σ_xy_kubo\n")
        end
    end
    open("AHE_results.csv", "a") do file
        write(file, "$(n),$(σ_xy[i]),$(σ_xy_kubo[i])\n")
    end
end

# Plot σ_xy vs. n
figure(figsize=(6,4))
plot(n_vals, σ_xy, lw=2, color="C0")
plot(n_vals, σ_xy_kubo, lw=2, linestyle="--", color="C1", label="Kubo")
xlabel(L"Electron density $n$", fontsize=14)
ylabel(L"$\sigma_{xy}$ ($e^2/\hbar$)", fontsize=14)
title(L"AHE at $U = %$(round(U_vals[U_idx], digits=2))$", fontsize=14)
grid(alpha=0.3)
legend(fontsize=12)
tight_layout()
savefig("AHE_vs_n_U$(U_target).pdf", dpi=300, bbox_inches="tight")
plt.close()  # Prevent memory leaks