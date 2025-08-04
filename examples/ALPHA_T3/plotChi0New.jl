using PyPlot
const plt = PyPlot
using BSON: @load
using LaTeXStrings
using LinearAlgebra
using AltermagneticHubbardModel  # For ModelParams
using Interpolations  # Added for native Julia interpolation
using Dierckx         # Added for 2D spline interpolation



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

# Helper function to check if a point is inside a convex polygon
function is_inside_convex_polygon(point, polygon_vertices)
    # point = (px, py)
    # polygon_vertices = [(x1, y1), (x2, y2), ...]
    num_vertices = length(polygon_vertices)
    if num_vertices < 3
        return false # Not a polygon
    end

    px, py = point
    
    # Calculate initial cross product sign using the first edge
    v1x, v1y = polygon_vertices[1]
    v2x, v2y = polygon_vertices[2]
    initial_cross_product = (v2x - v1x) * (py - v1y) - (v2y - v1y) * (px - v1x)
    
    for i in 2:num_vertices
        v1x, v1y = polygon_vertices[i]
        v2x, v2y = polygon_vertices[(i % num_vertices) + 1] # Wrap around to first vertex
        
        current_cross_product = (v2x - v1x) * (py - v1y) - (v2y - v1y) * (px - v1x)
        
        # If signs are different (and not zero), point is outside
        # Account for points exactly on the boundary (cross_product == 0)
        if initial_cross_product * current_cross_product < 0
            return false
        end
    end
    return true
end

# Plot χ_0 as a 2D heatmap over the hexagonal FBZ
function plot_chi_0_fbz(qpoints, chi_values, a=1.0)
    # Extract qx, qy
    qx = [q[1] for q in qpoints]
    qy = [q[2] for q in qpoints]
    
    # Define hexagonal FBZ boundary using reciprocal vectors
    b1, b2 = get_reciprocal_vectors(AltermagneticHubbardModel.ALPHA_T3)

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

    # Close the hexagon for plotting the boundary
    hex_x = [v[1] for v in fbz_vertices]
    hex_y = [v[2] for v in fbz_vertices]
    push!(hex_x, hex_x[1])
    push!(hex_y, hex_y[1])

    # 1. Create a denser grid for interpolation
    min_qx_fbz = minimum([v[1] for v in fbz_vertices])
    max_qx_fbz = maximum([v[1] for v in fbz_vertices])
    min_qy_fbz = minimum([v[2] for v in fbz_vertices])
    max_qy_fbz = maximum([v[2] for v in fbz_vertices])

    grid_res = 200 # Number of points in each dimension for the new grid
    buffer = 0.01 # Small buffer to ensure the grid covers the entire FBZ
    qx_dense = range(min_qx_fbz - buffer, max_qx_fbz + buffer, length=grid_res)
    qy_dense = range(min_qy_fbz - buffer, max_qy_fbz + buffer, length=grid_res)

    # 2. Create the 2D spline object for interpolation
    # s=0 for exact interpolation. A small positive s could be used for smoothing noisy data.
    spl = Spline2D(qx, qy, chi_values, s=3) 

    # 3. Evaluate the spline on the dense grid
    # Note: Dierckx.evaluate expects (x, y) order, and the resulting matrix
    # will have rows corresponding to y-values and columns to x-values,
    # which is the correct format for PyPlot's pcolormesh.
    chi_interp = [evaluate(spl, qxd, qyd) for qyd in qy_dense, qxd in qx_dense]

    # 4. Mask out points outside the hexagonal FBZ
    masked_chi_interp = copy(chi_interp)
    for i in 1:grid_res
        for j in 1:grid_res
            qxd = qx_dense[j]
            qyd = qy_dense[i]
            if !is_inside_convex_polygon((qxd, qyd), fbz_vertices)
                masked_chi_interp[i, j] = NaN # Set to NaN to mask in PyPlot
            end
        end
    end

    # 5. Plot with pcolormesh
    fig, ax = plt.subplots(figsize=(7, 6)) # Slightly larger for colorbar
    # Use pcolormesh for a continuous heatmap. shading="gouraud" smooths colors.
    sc = ax.pcolormesh(qx_dense, qy_dense, masked_chi_interp, cmap="viridis", shading="gouraud")

    # Add the FBZ boundary
    ax.plot(hex_x, hex_y, "k-", linewidth=1.5) # Thicker boundary

    # Add high-symmetry points
    for (label, point) in sym_points
        ax.scatter([point[1]], [point[2]], c="red", s=50, marker="o", edgecolors="black", zorder=5) # zorder to ensure points are on top
        # Offset labels to avoid overlap
        offset_x = 0.0
        offset_y = 0.0
        if label == L"K_1"
            offset_x, offset_y = 0.05, 0.05
        elseif label == L"K_2"
            offset_x, offset_y = 0.05, -0.05
        elseif label == L"K_3"
            offset_x, offset_y = -0.05, 0.05
        elseif label == L"K_4"
            offset_x, offset_y = -0.05, -0.05
        elseif label == L"K_5"
            offset_x, offset_y = 0.05, -0.05
        elseif label == L"K_6"
            offset_x, offset_y = -0.05, 0.05
        end
        ax.text(point[1] + offset_x, point[2] + offset_y, label, fontsize=12, color="black", ha="center", va="center")
    end

    ax.set_xlabel(L"$q_x \, (2\pi/a)$")
    ax.set_ylabel(L"$q_y \, (2\pi/a)$")
    ax.set_title(L"$\chi_0(\mathbf{q}, \omega=0)$ for $α=π/4$")
    plt.colorbar(sc, label=L"$\lambda_{\mathrm{max}}[\chi_0^{\mu\nu}(\mathbf{q}, 0)]$ (arb. units)")

    #ax.set_aspect("equal") # Keep aspect ratio
    plt.show()
end


δm = 0.0
μ  = find_chemical_potential(params, δm; μ_min=-3.0, μ_max=3.0)
qpoints, chi_values = compute_chi0_leading_eigenvalue_parallel(params, μ, δm; η=0.01)

plot_chi_0_fbz(qpoints, chi_values)
