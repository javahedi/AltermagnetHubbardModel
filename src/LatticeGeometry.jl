module LatticeGeometry

using AltermagneticHubbardModel
export get_reciprocal_vectors, generate_kpoints

"""
    get_reciprocal_vectors(lattice::LatticeType) -> (b1, b2)

Returns the reciprocal lattice vectors for the given lattice type.
"""
function get_reciprocal_vectors(lattice::Symbol)
    if lattice == SQUARE
        return ([2π, 0.0], [0.0, 2π])
    elseif lattice == HONEYCOMB
        return ([2π, 2π/√3], [2π, -2π/√3])
    elseif lattice == TRIANGULAR
        return ([2π, -2π/√3], [0.0, 4π/√3])
    elseif lattice == HEXATRIANGULAR
        # Hextriangular lattice reciprocal vectors
        # Primitive vectors in real space:
        a1 = [cos(π/3), sin(π/3)]  # (1/2, √3/2)
        a2 = [0.0, 1.0]
        
        # Calculate reciprocal vectors using 2π (aᵢ·bⱼ) = 2πδᵢⱼ
        b1 = 2π * [1, -1/√3]
        b2 = 2π * [0, 2/√3]
        return (b1, b2)
    else
        error("Unknown lattice type: $lattice. Valid options are: $SQUARE, $HONEYCOMB, $TRIANGULAR")
    end
end

"""
    generate_kpoints(lattice::LatticeType, nk::Int) -> Matrix{Tuple{Float64,Float64}}

Generates a nk×nk grid of k-points in the Brillouin zone.
"""
function generate_kpoints(lattice::Symbol, nk::Int)
    b1, b2 = get_reciprocal_vectors(lattice)
    kx = range(-π, π, length=nk)
    ky = range(-π, π, length=nk)
    return [(k1, k2) for k1 in kx, k2 in ky]
end

end # module