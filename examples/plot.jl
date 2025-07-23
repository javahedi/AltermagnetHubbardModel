"""
using AltermagneticHubbardModel
using PyPlot
const plt = PyPlot
using BSON: @save, @load

@load "examples/phaseDiagram_alpha_U_n1.0_2025-07-23_0002.bson" results α_vals U_vals fixed_n

#α_vals = 0.0:π/64:π/2

α_grid = collect(α_vals)
U_grid = collect(U_vals)
δm_alpha = [get(results, (α, U), NaN) for α in α_grid, U in U_grid]
δm_matrix_alpha = permutedims(δm_alpha)


figure()
imshow(δm_matrix_alpha;
    extent=(minimum(α_grid), maximum(α_grid), minimum(U_grid), maximum(U_grid)),
    aspect="auto", origin="lower", cmap="viridis", interpolation="hanning") # nearest, hanning
colorbar(label=L"\delta m")  # LaTeX in colorbar label
xlabel(L"\alpha")            # LaTeX in x-axis label
xticks([0, π/4, π/2], [L"0", L"\frac{\pi}{4}", L"\frac{\pi}{2}"])
ylabel(L"U")                 # LaTeX in y-axis label
title(L" n = 1.0")
savefig("heatmap_alpha_U.pdf", dpi=300)


@load "examples/phaseDiagram_n_U_alpha0.79_2025-07-23_1107.bson" results n_vals U_vals fixed_α

n_grid = collect(n_vals)
U_grid = collect(U_vals)

δm_n = [get(results, (n, U), NaN) for n in n_grid, U in U_grid]
δm_matrix_n = permutedims(δm_n)

figure()
imshow(δm_matrix_n;
    extent=(minimum(n_grid), maximum(n_grid), minimum(U_grid), maximum(U_grid)),
    aspect="auto", origin="lower", cmap="viridis", interpolation="hanning") # bilinear
colorbar(label=L"\delta m")  # LaTeX in colorbar label
xlabel(L"n")            # LaTeX in x-axis label
ylabel(L"U")
title(L" α = π/4")
savefig("heatmap_n_U.pdf", dpi=300)
"""