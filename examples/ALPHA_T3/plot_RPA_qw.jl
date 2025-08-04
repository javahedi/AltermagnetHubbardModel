using AltermagneticHubbardModel
using BSON: @load
using PyPlot
const plt = PyPlot

# --------------------------
# Step 1: Load RPA data
# --------------------------

# Load your RPA data file (update filename accordingly)
data = Dict()
@load "examples/ALPHA_T3/rpa_susceptibility_U2.0_α0.785_time_20250801_1241.bson" data


# --------------------------
# Step 2: Extract ω, χ matrices
# --------------------------

ωlist = data[:ωlist]


χzz_matrix = data[:χzz]
χpm_matrix = data[:χpm]
χmp_matrix = data[:χmp]


# Optional: Load symmetry labels if saved
qpath, labels, ticks = get_high_symmetry_path(ALPHA_T3, 100)


nq = length(qpath)
nω = length(ωlist)
# --------------------------
# Step 3: Plotting with PyPlot
# --------------------------

fig, axs = plt.subplots(1, 3, figsize=(15, 4), sharex=true, sharey=true)

extent = (1, nq, ωlist[1], ωlist[end])  # (xmin, xmax, ymin, ymax)

# Plot χzz
im1 = axs[1].imshow(χzz_matrix, aspect="auto", origin="lower", extent=extent, cmap="inferno")
axs[1].set_title("Im(χᶻᶻ)")
axs[1].set_xlabel("q path")
axs[1].set_ylabel("ω")

# Plot χpm
im2 = axs[2].imshow(χpm_matrix, aspect="auto", origin="lower", extent=extent, cmap="inferno")
axs[2].set_title("Im(χ⁺⁻)")
axs[2].set_xlabel("q path")

# Plot χmp
im3 = axs[3].imshow(χmp_matrix, aspect="auto", origin="lower", extent=extent, cmap="inferno")
axs[3].set_title("Im(χ⁻⁺)")
axs[3].set_xlabel("q path")

# Optional symmetry labels on x-axis
if !isempty(ticks) && !isempty(labels)
    for ax in axs
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels)
    end
end

plt.tight_layout()
plt.show()
