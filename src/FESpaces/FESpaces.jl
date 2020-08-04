"""

The exported names are
$(EXPORTS)
"""
module FESpaces

using DocStringExtensions
using Test
using FillArrays
using SparseArrays
using LinearAlgebra

using Gridap.Inference
using Gridap.Helpers
using Gridap.Arrays
using Gridap.ReferenceFEs
using Gridap.Geometry
using Gridap.Fields
using Gridap.Integration
using Gridap.Algebra
using Gridap.Polynomials
using Gridap.TensorValues

using Gridap.ReferenceFEs: evaluate_dof_array

using Gridap.Geometry: CellFieldLike
using Gridap.Geometry: UnimplementedField
using Gridap.Geometry: test_cell_field_like

using Gridap.Arrays: _split
using Gridap.Arrays: Reindexed
using Gridap.Arrays: IdentityVector

import Gridap.Arrays: get_array
import Gridap.Arrays: array_cache
import Gridap.Arrays: getindex!
import Gridap.Arrays: kernel_cache
import Gridap.Arrays: apply_kernel!
import Gridap.Arrays: kernel_return_type
import Gridap.Arrays: kernel_testitem!
import Gridap.Arrays: reindex
import Gridap.Arrays: apply
import Gridap.Geometry: get_cell_map
import Gridap.Geometry: get_cell_shapefuns
import Gridap.Geometry: get_reffes
import Gridap.Geometry: get_cell_type
import Gridap.Geometry: RefStyle
import Gridap.Geometry: change_ref_style
import Gridap.Helpers: operate
import Gridap.Geometry: similar_object
import Gridap.Geometry: jump
import Gridap.Geometry: mean
import Gridap.Geometry: restrict
import Gridap.Geometry: get_cell_id
import Gridap.Fields: integrate
import Gridap.Fields: evaluate
import Gridap.Fields: gradient
import Gridap.Fields: grad2curl
import Gridap.Fields: evaluate_field_array
import Gridap.ReferenceFEs: evaluate_dof_array
import Gridap.Fields: field_cache
import Gridap.Fields: evaluate_field!
import Gridap.Fields: field_gradient

import Gridap.Algebra: allocate_residual
import Gridap.Algebra: allocate_jacobian
import Gridap.Algebra: residual!
import Gridap.Algebra: jacobian!
import Gridap.Algebra: residual
import Gridap.Algebra: jacobian
import Gridap.Algebra: residual_and_jacobian!
import Gridap.Algebra: residual_and_jacobian
import Gridap.Algebra: zero_initial_guess
import Gridap.Algebra: get_matrix
import Gridap.Algebra: get_vector
import Gridap.Algebra: solve!
import Gridap.Algebra: solve
import Gridap.Algebra: allocate_vector
import Gridap.Algebra: allocate_matrix
import Gridap.Algebra: allocate_matrix_and_vector

export FEFunctionStyle
export is_a_fe_function
export get_free_values
export get_fe_space
export get_cell_values
export test_fe_function

export FESpace
export FEFunction
export num_free_dofs
export get_cell_basis
export zero_free_values
export constraint_style
export has_constraints
export get_cell_isconstrained
export get_cell_constraints
export get_constraint_kernel_matrix_cols
export get_constraint_kernel_matrix_rows
export get_constraint_kernel_vector
export apply_constraints_matrix_cols
export apply_constraints_matrix_rows
export apply_constraints_vector
export apply_constraints_matrix_and_vector_cols
export apply_constraints_matrix_and_vector_rows
export test_fe_space

export Assembler
export AssemblyStrategy
export row_map
export col_map
export row_mask
export col_mask
export DefaultAssemblyStrategy
export get_test
export get_trial
export assemble_matrix!
export assemble_matrix_add!
export assemble_matrix
export assemble_vector!
export assemble_vector_add!
export assemble_vector
export assemble_matrix_and_vector!
export assemble_matrix_and_vector_add!
export assemble_matrix_and_vector
export allocate_vector
export allocate_matrix
export allocate_matrix_and_vector
export test_assembler
export get_matrix_type
export get_vector_type
export count_matrix_nnz_coo
export count_matrix_and_vector_nnz_coo
export fill_matrix_coo_symbolic!
export fill_matrix_and_vector_coo_symbolic!
export fill_matrix_coo_numeric!
export fill_matrix_and_vector_coo_numeric!
export test_sparse_matrix_assembler

export SingleFieldFESpace
export num_dirichlet_dofs
export get_cell_dofs
export zero_dirichlet_values
export gather_free_and_dirichlet_values
export gather_free_and_dirichlet_values!
export scatter_free_and_dirichlet_values
export get_dirichlet_values
export gather_dirichlet_values
export gather_dirichlet_values!
export num_dirichlet_tags
export gather_free_values
export gather_free_values!
export get_dirichlet_dof_tag
export compute_free_and_dirichlet_values
export compute_dirichlet_values
export compute_free_values
export compute_dirichlet_values_for_tags
export compute_dirichlet_values_for_tags!
export test_single_field_fe_space
export interpolate
export interpolate_everywhere
export interpolate_dirichlet
export get_cell_dof_basis
export get_cell_shapefuns

export SingleFieldFEFunction

export UnconstrainedFESpace
export GradConformingFESpace
export DiscontinuousFESpace

export CellBasis
export test_cell_basis
export CellMatrixField
export test_cell_matrix_field
export GenericCellBasis
export GenericCellMatrixField
export TrialStyle
export is_trial
export is_test
export attach_dirichlet_bcs

export CellDofBasis
export GenericCellDofBasis
export test_cell_dof_basis

export FECellBasisStyle
export is_a_fe_cell_basis

export TrialFESpace
export TrialFESpace!
export HomogeneousTrialFESpace
export HomogeneousTrialFESpace!
export TestFESpace
export compute_conforming_cell_dofs
export SparseMatrixAssembler

export FEOperator
export test_fe_operator
export AffineFEOperator
export FEOperatorFromTerms
export get_algebraic_operator

export FESolver
export LinearFESolver
export NonlinearFESolver
export test_fe_solver

export FETerm
export AffineFETerm
export LinearFETerm
export FESource
export AffineFETermFromCellMatVec
export FETermFromCellJacRes
export get_cell_matrix
export get_cell_vector
export get_cell_jacobian
export get_cell_jacobian_and_residual
export get_cell_residual
export collect_cell_matrix
export collect_cell_vector
export collect_cell_matrix_and_vector
export collect_cell_jacobian
export collect_cell_jacobian_and_residual
export collect_cell_residual

export FESpaceWithLastDofRemoved
export ZeroMeanFESpace
export CLagrangianFESpace
export ConformingFESpace
export DirichletFESpace
export FESpaceWithLinearConstraints
export ExtendedFESpace

export @law
export operate
export GridapType

export apply_cellmatvec
export apply_cellmatrix
export apply_cellvector

export @statelaw
export apply_statelaw
export CellField
export update_state_variables!

export autodiff_cell_residual_from_energy
export autodiff_cell_jacobian_from_energy
export autodiff_cell_jacobian_from_residual

export FEEnergy

include("CellBases.jl")

include("CellDofBases.jl")

include("Law.jl")

include("FEFunctions.jl")

include("FESpacesInterfaces.jl")

include("Assemblers.jl")

include("FEOperators.jl")

include("SingleFieldFESpaces.jl")

include("SingleFieldFEFunctions.jl")

include("TrialFESpaces.jl")

include("SparseMatrixAssemblers.jl")

include("UnconstrainedFESpaces.jl")

include("ConformingFESpaces.jl")

include("DiscontinuousFESpaces.jl")

include("FETerms.jl")

include("CellKernels.jl")

include("AffineFEOperators.jl")

include("FEOperatorsFromTerms.jl")

include("FESolvers.jl")

include("FESpacesWithLastDofRemoved.jl")

include("ZeroMeanFESpaces.jl")

include("CLagrangianFESpaces.jl")

include("DirichletFESpaces.jl")

include("ExtendedFESpaces.jl")

include("FESpacesWithLinearConstraints.jl")

include("FESpaceFactories.jl")

include("StateLaws.jl")

include("FEAutodiff.jl")

include("FETermsWithAutodiff.jl")

end # module
