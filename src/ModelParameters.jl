module ModelParameters

# Define lattice type constants
const SQUARE = :square
const HONEYCOMB = :honeycomb
const TRIANGULAR = :triangular

# Export lattice types and ModelParams
export SQUARE, HONEYCOMB, TRIANGULAR, ModelParams

"""
    ModelParams

Parameters for the altermagnetic Hubbard model.

# Fields
- `lattice::Symbol`: Lattice geometry (:square, :honeycomb, :triangular)
- `t::Float64`: Nearest-neighbor hopping
- `t_prime::Float64`: Diagonal hopping amplitude
- `δ::Float64`: Altermagnetic parameter
- `U::Float64`: On-site Hubbard interaction
- `n::Float64`: Electron filling (n=1 for half-filling)
- `β::Float64`: Inverse temperature (1/kT)
- `kpoints::Int`: Number of k-points per dimension
- `mixing::Float64`: Mixing parameter for SCF (0.0-1.0)
- `tol::Float64`: Convergence tolerance
"""
struct ModelParams
    lattice::Symbol
    t::Float64
    t_prime::Float64
    δ::Float64
    U::Float64
    n::Float64
    β::Float64
    kpoints::Int
    mixing::Float64
    tol::Float64
end
    
function ModelParams(;
    lattice::Symbol,
    t::Float64,
    t_prime::Float64,
    δ::Float64,
    λ_R::Float64, 
    U::Float64,
    n::Float64,
    β::Float64,
    kpoints::Int,
    mixing::Float64,
    tol::Float64
)
    if !(lattice in [SQUARE, HONEYCOMB, TRIANGULAR])
        error("Invalid lattice type: $lattice")
    end
    ModelParams(lattice, t, t_prime, δ, λ_R, U, n, β, kpoints, mixing, tol)
end


end