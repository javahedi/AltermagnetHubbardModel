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

  
    include("FermiSurface.jl")
    @reexport using .FermiSurface


    include("BandStructurePlotting.jl")
    @reexport using .BandStructurePlotting


    include("ConductivityTensor.jl")
    @reexport using .ConductivityTensor

    include("SpectralFunction.jl")
    @reexport using .SpectralFunction

    #include("SpectralFunctionFull.jl")
    #@reexport using .SpectralFunctionFull

    #include("BandStructurePlottingFull.jl")
    #@reexport using .BandStructurePlottingFull

    #include("ConductivityTensorFull.jl")
    #@reexport using .ConductivityTensorFull


end # module