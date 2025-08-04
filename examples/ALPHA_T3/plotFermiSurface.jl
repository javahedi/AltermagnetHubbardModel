
using AltermagneticHubbardModel
using PyPlot
const plt = PyPlot
using BSON: @save, @load



@load "examples/ALPHA_T3/phaseDiagram_alpha_U_n1.0_2025-07-23_0002.bson" results α_vals U_vals fixed_n

α_grid = collect(α_vals)
U_grid = collect(U_vals)
δm_alpha = [get(results, (α, U), NaN) for α in α_grid, U in U_grid]
δm_matrix_alpha = permutedims(δm_alpha)


α = π/8       # Or α_grid[9], etc.
U = 3.8      # Or U_grid[10], etc.
δm = get(results, (α, U), NaN)
if isnan(δm)
    println("No data available for α = $α and U = $U")
    return
end

params = ModelParams(
            lattice = ALPHA_T3,
            t       = 1.0,
            t_prime = 0.0,
            δ       = 0.0,
            U       = U,
            λ       = 0.0,
            n       = fixed_n,
            β       = 10000.0,
            α       = α,
            kpoints = 100,
            mixing  = 0.4,
            tol     = 1e-6
        )


fig = plot_fermi_surface(params, δm; nk=100)
if fig
    plt.show()
else
    println("Fermi surface plot not generated.")
end