module ModelParameters

# Define lattice type constants
const SQUARE = :square
const HONEYCOMB = :honeycomb
const TRIANGULAR = :triangular
const HEXATRIANGULAR = :hexatriangular
const ALPHA_T3 = :alpha_t3

# Export lattice types and ModelParams
export SQUARE, HONEYCOMB, HEXATRIANGULAR, TRIANGULAR, ALPHA_T3, ModelParams

"""
    ModelParams

Parameters for the altermagnetic Hubbard model.

# Fields
- `lattice::Symbol`: Lattice geometry (:square, :honeycomb, :triangular)
- `t::Float64`: Nearest-neighbor hopping
- `t_prime::Float64`: Diagonal hopping amplitude
- `δ::Float64`: Altermagnetic parameter
- `U::Float64`: On-site Hubbard interaction
- `λ::Float64`: Rashba spin-orbit coupling strength
- `n::Float64`: Electron filling (n=1 for half-filling)
- `β::Float64`: Inverse temperature (1/kT)
- `α::Float64`: α parameter for T3 lattice
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
    λ::Float64
    n::Float64
    β::Float64
    α::Float64
    kpoints::Int
    mixing::Float64
    tol::Float64
end
    
function ModelParams(;
    lattice::Symbol,
    t::Float64,
    t_prime::Float64,
    δ::Float64, 
    U::Float64,
    λ::Float64,
    n::Float64,
    β::Float64,
    α::Float64,
    kpoints::Int,
    mixing::Float64,
    tol::Float64
)
    if !(lattice in [SQUARE, HONEYCOMB, HEXATRIANGULAR, TRIANGULAR, ALPHA_T3])
        error("Invalid lattice type: $lattice")
    end
    ModelParams(lattice, t, t_prime, δ, U, λ, n, β, α, kpoints, mixing, tol)
end


end