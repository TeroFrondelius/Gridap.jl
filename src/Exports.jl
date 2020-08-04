
macro publish(mod,name)
  quote
    using Gridap.$mod: $name; export $name
  end
end

# Reexport from LinearAlgebra (just for convenience)
using LinearAlgebra:  det, inv, tr, cross, dot, norm, ×, ⋅
export det, inv, tr, cross, dot, norm, ×, ⋅

@publish Helpers operate
@publish Helpers GridapType

@publish Algebra solve
@publish Algebra solve!
@publish Algebra symbolic_setup
@publish Algebra numerical_setup
@publish Algebra numerical_setup!
@publish Algebra LUSolver
@publish Algebra BackslashSolver
@publish Algebra zero_initial_guess
@publish Algebra NLSolver
@publish Algebra get_matrix
@publish Algebra get_vector
@publish Algebra SparseMatrixCSR
@publish Algebra SymSparseMatrixCSR

@publish Arrays array_cache
@publish Arrays getindex!
@publish Arrays get_array
@publish Arrays apply
@publish Arrays reindex

@publish TensorValues VectorValue
@publish TensorValues TensorValue
@publish TensorValues inner
@publish TensorValues outer
@publish TensorValues diagonal_tensor
@publish TensorValues num_components
using Gridap.TensorValues: ⊙; export ⊙
using Gridap.TensorValues: ⊗; export ⊗

@publish Fields gradient
@publish Fields ∇
@publish Fields integrate
@publish Fields Point
@publish Fields evaluate
@publish Fields evaluate!
@publish Fields curl
@publish Fields laplacian
@publish Fields divergence
@publish Fields Δ
@publish Fields ε
@publish Fields symmetric_gradient

@publish Integration get_coordinates
@publish Integration get_weights

@publish ReferenceFEs is_simplex
@publish ReferenceFEs is_n_cube
@publish ReferenceFEs simplexify
@publish ReferenceFEs num_dims
@publish ReferenceFEs num_cell_dims
@publish ReferenceFEs num_point_dims
@publish ReferenceFEs VERTEX
@publish ReferenceFEs SEGMENT
@publish ReferenceFEs TRI
@publish ReferenceFEs QUAD
@publish ReferenceFEs TET
@publish ReferenceFEs HEX
@publish ReferenceFEs WEDGE
@publish ReferenceFEs PYRAMID
@publish ReferenceFEs LagrangianRefFE
@publish ReferenceFEs is_first_order
@publish ReferenceFEs is_affine
@publish ReferenceFEs is_Q
@publish ReferenceFEs is_P
@publish ReferenceFEs is_S
@publish ReferenceFEs VERTEX1
@publish ReferenceFEs SEG2
@publish ReferenceFEs TRI3
@publish ReferenceFEs QUAD4
@publish ReferenceFEs TET4
@publish ReferenceFEs HEX8
@publish ReferenceFEs SerendipityRefFE
@publish ReferenceFEs Polytope

@publish Geometry get_triangulation
@publish Geometry num_cells
@publish Geometry Triangulation
@publish Geometry get_normal_vector
@publish Geometry get_cell_coordinates
@publish Geometry restrict
@publish Geometry get_physical_coordinate
@publish Geometry get_cell_map
@publish Geometry CartesianGrid
@publish Geometry CartesianDiscreteModel
@publish Geometry DiscreteModel
@publish Geometry DiscreteModelFromFile
@publish Geometry num_tags
@publish Geometry num_entities
@publish Geometry get_grid
@publish Geometry get_face_labeling
@publish Geometry add_tag!
@publish Geometry add_tag_from_tags!
@publish Geometry BoundaryTriangulation
@publish Geometry jump
@publish Geometry mean
@publish Geometry SkeletonTriangulation
@publish Geometry CellQuadrature
@publish Geometry QPointCellField
@publish Geometry CellField
@publish Geometry RestrictedTriangulation
@publish Geometry InterfaceTriangulation

@publish FESpaces FESpace
@publish FESpaces TrialFESpace
@publish FESpaces TestFESpace
@publish FESpaces FETerm
@publish FESpaces FEEnergy
@publish FESpaces AffineFETerm
@publish FESpaces LinearFETerm
@publish FESpaces FESource
@publish FESpaces AffineFEOperator
@publish FESpaces LinearFESolver
@publish FESpaces get_free_values
@publish FESpaces get_dirichlet_values
@publish FESpaces num_dirichlet_dofs
@publish FESpaces num_free_dofs
@publish FESpaces num_dirichlet_tags
@publish FESpaces FEFunction
@publish FESpaces interpolate
@publish FESpaces interpolate_everywhere
@publish FESpaces interpolate_dirichlet
@publish FESpaces FEOperator
@publish FESpaces FESolver
@publish FESpaces apply_statelaw
@publish FESpaces update_state_variables!

using Gridap.FESpaces: @law; export @law
using Gridap.FESpaces: @statelaw; export @statelaw

@publish MultiField MultiFieldFESpace

@publish Visualization writevtk
@publish Visualization createvtk
@publish Visualization print_op_tree

