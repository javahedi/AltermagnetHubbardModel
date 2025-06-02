module AltermagneticHubbardModel

    using Reexport

    # Include and reexport all submodules
    include("ModelParameters.jl")
    @reexport using .ModelParameters


    include("LatticeGeometry.jl")
    @reexport using .LatticeGeometry

    include("ModelHamiltonian.jl")
    @reexport using .ModelHamiltonian

    include("SelfConsistentLoop.jl")
    @reexport using .SelfConsistentLoop

    include("BandStructurePlotting.jl")
    @reexport using .BandStructurePlotting

    include("FermiSurface.jl")
    @reexport using .FermiSurface


    include("ConductivityTensor.jl")
    @reexport using .ConductivityTensor

    include("SpectralFunction.jl")
    @reexport using .SpectralFunction


end # module