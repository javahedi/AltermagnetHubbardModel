
using PyPlot
const plt = PyPlot
using BSON: @load
using LaTeXStrings

# Load the α-U phase diagram data
@load "examples/ALPHA_T3/phase_diagram_n1.0_0.785_2025-07-23_212624.bson" save_data

# Extract data
α_grid = save_data[:α_vals]
U_grid = save_data[:U_vals]
δm_matrix = save_data[:δm_matrix]  
μ_matrix  = save_data[:μ_matrix]


# Create alpha-U heatmap
figure(figsize=(5,4))
imshow(δm_matrix',
       extent=(minimum(α_grid), maximum(α_grid), minimum(U_grid), maximum(U_grid)),
       aspect="auto",
       origin="lower",
       cmap="viridis",
       interpolation="hanning")

colorbar(label=L"\delta m")
xlabel(L"\alpha")
xticks([0, π/4, π/2], [L"0", L"\pi/4", L"\pi/2"])
ylabel(L"U")
title(L"Staggered Magnetization ($n=1.0$)")
tight_layout()
savefig("heatmap_alpha_U.pdf", dpi=300, bbox_inches="tight")

# Load the n-U phase diagram data
@load "examples/phase_diagram_alpha_0.785_2025-07-23_203620.bson" save_data

# Extract data
n_grid = save_data[:n_vals]
U_grid = save_data[:U_vals]
δm_matrix = save_data[:δm_matrix] 
μ_matrix  = save_data[:μ_matrix]


# Create n-U heatmap
figure(figsize=(5,4))
imshow(δm_matrix',
       extent=(minimum(n_grid), maximum(n_grid), minimum(U_grid), maximum(U_grid)),
       aspect="auto",
       origin="lower",
       cmap="viridis",
       interpolation="hanning")

colorbar(label=L"\delta m")
xlabel(L"Filling $n$")
ylabel(L"U")
title(L"Staggered Magnetization ($\alpha = \pi/4$)")
tight_layout()
savefig("heatmap_n_U.pdf", dpi=300, bbox_inches="tight")