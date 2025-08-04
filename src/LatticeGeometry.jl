module LatticeGeometry

using AltermagneticHubbardModel
using LinearAlgebra
export get_reciprocal_vectors, generate_kpoints, generate_fbz_grid
export get_high_symmetry_path

"""
    get_reciprocal_vectors(lattice::LatticeType) -> (b1, b2)

Returns the reciprocal lattice vectors for the given lattice type.
"""
function get_reciprocal_vectors(lattice::Symbol)
    if lattice == SQUARE
        a1 = [1.0, 0.0]
        a2 = [0.0, 1.0]
    elseif lattice == HONEYCOMB || lattice == ALPHA_T3 || lattice == KMmodel
        # Honeycomb (or alpha-T3) uses hexagonal real-space basis
        a1 = [0.0, 1.0]
        a2 = [1/2, √3/2]
    elseif lattice == TRIANGULAR
        a1 = [1.0, 0.0]
        a2 = [1/2, √3/2]
    elseif lattice == HEXATRIANGULAR
        # Hextriangular lattice reciprocal vectors
        # Primitive vectors in real space:
        a1 = [1.0, 0.0]
        a2 = [1/2, √3/2]
        
    else
        error("Unknown lattice type: $lattice")
    end

    # Reciprocal lattice formula
    volume = a1[1]*a2[2] - a1[2]*a2[1]
    b1 = 2π * [ a2[2], -a2[1]] / volume
    b2 = 2π * [-a1[2],  a1[1]] / volume
    
    return b1, b2

        
end

"""
    generate_kpoints(lattice::LatticeType, nk::Int) -> Matrix{Tuple{Float64,Float64}}

Generates a nk×nk grid of k-points in the Brillouin zone.
"""



function generate_kpoints(lattice::Symbol, nk::Int)
    # k=u*b1 + v*b2 , u,v ∈ [0,1)
    #   centering the Brillouin zone around the origin u, v ∈ [-0.5, 0.5)
    if nk <= 0
        error("Number of k-points must be positive")
    end
    b1, b2 = get_reciprocal_vectors(lattice)
    kpoints = []
    for i in 0:nk-1, j in 0:nk-1
        #u = i / nk
        #v = j / nk
        u = (i - nk/2) / nk
        v = (j - nk/2) / nk
        k = u .* b1 .+ v .* b2
        push!(kpoints, (k[1], k[2]))
    end
    return kpoints
end




# Check if a point is in the hexagonal FBZ
function in_hexagonal_fbz(q, b1, b2)
    # Reciprocal lattice vectors and their combinations
    G = [ [0, 0], b1, b2, b1 - b2, -b1, -b2, -(b1 - b2) ]
    for g in G[2:end]
        if norm(q) > norm(q - g)
            return false
        end
    end
    return true
end

# Generate uniform grid in the hexagonal FBZ
function generate_fbz_grid(nq::Int)

    b1, b2 = get_reciprocal_vectors(HONEYCOMB)

    # Grid points in parallelogram spanned by b1, b2
    qpoints = []
    n = ceil(Int, sqrt(nq))  # Approximate square grid for desired number of points
    for i in range(-1, 1, length=n)
        for j in range(-1, 1, length=n)
            q = i * b1 + j * b2
            if in_hexagonal_fbz(q, b1, b2)
                push!(qpoints, (q[1], q[2]))
            end
        end
    end
    return qpoints
end


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
            [(Γ[1]  + t*(X[1]-Γ[1]),  Γ[2]  + t*(X[2]-Γ[2]))  for t in range(0, 1, length=npoints)],
            [(X[1]  + t*(M[1]-X[1]),  X[2]  + t*(M[2]-X[2]))  for t in range(0, 1, length=npoints)],
            [(M[1]  + t*(X2[1]-M[1]), M[2]  + t*(X2[2]-M[2])) for t in range(0, 1, length=npoints)],
            [(X2[1] + t*(Γ[1]-X2[1]), X2[2] + t*(Γ[2]-X2[2])) for t in range(0, 1, length=npoints)]
        )
        
        labels = ["Γ", "X", "M", "X2", "Γ"]
        ticks = [1, npoints, 2npoints, 3npoints, 4npoints]
        
    elseif lattice == HEXATRIANGULAR || lattice == ALPHA_T3 || lattice == HONEYCOMB || lattice == KMmodel
        
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
            [(Γ[1]  + t*(M1[1]-Γ[1]), Γ[2]  + t*(M1[2]-Γ[2])) for t in range(0, 1, length=npoints)],
            [(M1[1] + t*(K[1]-M1[1]), M1[2] + t*(K[2]-M1[2])) for t in range(0, 1, length=npoints)],
            [(K[1]  + t*(M2[1]-K[1]), K[2]  + t*(M2[2]-K[2])) for t in range(0, 1, length=npoints)],
            [(M2[1] + t*(Γ[1]-M2[1]), M2[2] + t*(Γ[2]-M2[2])) for t in range(0, 1, length=npoints)]
        )

        labels = ["Γ", "M1", "K", "M2", "Γ"]
        ticks = [1, npoints, 2npoints, 3npoints, 4npoints]

    else
        error("Unsupported lattice for k-path: $lattice")
    end
    
    return (kpath, labels, ticks)
end


end # module