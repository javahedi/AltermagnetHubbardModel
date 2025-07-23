module LatticeGeometry

using AltermagneticHubbardModel
export get_reciprocal_vectors, generate_kpoints

"""
    get_reciprocal_vectors(lattice::LatticeType) -> (b1, b2)

Returns the reciprocal lattice vectors for the given lattice type.
"""
function get_reciprocal_vectors(lattice::Symbol)
    if lattice == SQUARE
        a1 = [1.0, 0.0]
        a2 = [0.0, 1.0]
    elseif lattice == HONEYCOMB || lattice == ALPHA_T3
        # Honeycomb (or alpha-T3) uses hexagonal real-space basis
        a1 = [0.0, 1.0]
        a2 = [1/2, √3/2]
    elseif lattice == TRIANGULAR
        a1 = [1.0, 0.0]
        a2 = [1/2, √3/2]
    elseif lattice == HEXATRIANGULAR
        # Hextriangular lattice reciprocal vectors
        # Primitive vectors in real space:
        a1 = [cos(π/3), sin(π/3)]  # (1/2, √3/2)
        a2 = [0.0, 1.0]
        
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

end # module