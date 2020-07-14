
"""
"""
struct MultiFieldArray{T,N,A<:AbstractArray{T,N}} <: AbstractArray{T,N}
  blocks::Vector{A}
  coordinates::Vector{NTuple{N,Int}}
  ptrs::Array{Int,N}
  block_size::NTuple{N,Int}
  scalar_size::Union{NTuple{N,Int},Nothing}
  cumulative_block_sizes::Union{Vector{Vector{Int}},Nothing}

  function MultiFieldArray(
    blocks::Vector{A},
    coordinates::Vector{NTuple{N,Int}}) where {T,N,A<:AbstractArray{T,N}}

    @assert length(blocks) == length(coordinates)
    msg = "Trying to build a MultiFieldArray with repeated blocks"
    @assert _no_has_repeaded_blocks(coordinates) msg
    ptrs = _prepare_ptrs(coordinates)
    block_size = _get_block_size(coordinates)
    scalar_size = _get_scalar_size(N,blocks,coordinates,block_size)
    cumulative_block_sizes = _get_cumulative_block_sizes(N,
                                                         block_size,
                                                         blocks,
                                                         coordinates)
    new{T,N,A}(blocks,
               coordinates,
               ptrs,
               block_size,
               scalar_size,
               cumulative_block_sizes)
  end
end

function _no_has_repeaded_blocks(coordinates::Vector{NTuple{N,Int}}) where N
  maxblocks = _get_block_size(coordinates)
  touched = zeros(Int,maxblocks)
  for c in coordinates
    touched[c...] += 1
  end
  all( touched .<= 1 )
end

function _prepare_ptrs(coordinates)
  s = _get_block_size(coordinates)
  ptrs = zeros(Int,s)
  for (i,c) in enumerate(coordinates)
    ptrs[c...] = i
  end
  ptrs
end

function _get_block_size(coordinates::Vector{NTuple{N,Int}}) where N
  m = zeros(Int,N)
  for c in coordinates
    for n in 1:N
      m[n] = max(m[n],c[n])
    end
  end
  NTuple{N,Int}(m)
end

function Base.copy(a::MultiFieldArray)
  MultiFieldArray([copy(b) for b in a.blocks],a.coordinates)
end

"""
"""
function get_block_size(a::MultiFieldArray)
  a.block_size
end

"""
"""
function num_blocks(a::MultiFieldArray)
  s = get_block_size(a)
  prod(s)
end

"""
"""
function num_stored_blocks(a::MultiFieldArray)
  length(a.coordinates)
end

"""
"""
function has_all_blocks(a::MultiFieldArray{T,N}) where {T,N}
  num_blocks(a) == num_stored_blocks(a)
end

function Base.:*(a::MultiFieldArray,b::Number)
  blocks = [ block*b for block in a.blocks ]
  coordinates = a.coordinates
  MultiFieldArray(blocks,coordinates)
end

function Base.:*(b::Number,a::MultiFieldArray)
  blocks = [ b*block for block in a.blocks ]
  coordinates = a.coordinates
  MultiFieldArray(blocks,coordinates)
end

function Base.show(io::IO,a::MultiFieldArray)
  print(io,"MultiFieldArray($(a.blocks),$(a.coordinates))")
end

function Base.show(io::IO,::MIME"text/plain",a::MultiFieldArray)
  println(io,"MultiFieldArray object:")
  cis = CartesianIndices(a.ptrs)
  for ci in cis
    p = a.ptrs[ci]
    if p == 0
      println(io,"Block $(Tuple(ci)) -> Empty")
    else
      println(io,"Block $(Tuple(ci)) -> $(a.blocks[p])")
    end
  end
end

function add_to_array!(a::MultiFieldArray{Ta,N},b::MultiFieldArray{Tb,N},combine=+) where {Ta,Tb,N}
  for coords in b.coordinates
    ak = a.blocks[a.ptrs[coords...]]
    bk = b.blocks[b.ptrs[coords...]]
    add_to_array!(ak,bk,combine)
  end
  a
end

function add_to_array!(a::MultiFieldArray,b::Number,combine=+)
  for k in 1:length(a.blocks)
    ak = a.blocks[k]
    add_to_array!(ak,b,combine)
  end
end

function Base.:*(a::MultiFieldArray{Ta,2},b::MultiFieldArray{Tb,1}) where {Ta,Tb}
  @assert num_stored_blocks(a) != 0
  @notimplementedif ! has_all_blocks(b)

  function fun(i,a,b)
    ai = a.blocks[i]
    ci, cj = a.coordinates[i]
    p = b.ptrs[cj]
    bi = b.blocks[p]
    ai*bi
  end

  blocks = [ fun(i,a,b) for i in 1:length(a.blocks)]
  coordinates = [ (c[1],)  for c in a.coordinates]

  data = _merge_repeated_blocks(blocks,coordinates)
  MultiFieldArray(data...)
end

function Base.fill!(a::MultiFieldArray,b)
  for k in 1:length(a.blocks)
    ak = a.blocks[k]
    fill!(ak,b)
  end
  a
end

Base.eltype(::Type{<:MultiFieldArray{T}}) where T = T

Base.eltype(a::MultiFieldArray{T}) where T = T

function _merge_repeated_blocks(blocks,coordinates::Vector{NTuple{N,Int}}) where N
  @assert length(blocks) == length(coordinates)
  s = _get_block_size(coordinates)
  ptrs = zeros(Int,s)
  A = eltype(blocks)
  _blocks = A[]
  _coords = NTuple{N,Int}[]
  q = 1
  for b in 1:length(blocks)
    c = coordinates[b]
    block = blocks[b]
    p = ptrs[c...]
    if p == 0
      push!(_blocks,block)
      push!(_coords,c)
      ptrs[c...] = q
      q += 1
    else
      add_to_array!(_blocks[p],block)
    end
  end
  (_blocks,_coords)
end

function mul!(c::MultiFieldArray{Tc,1},a::MultiFieldArray{Ta,2},b::MultiFieldArray{Tb,1}) where {Tc,Ta,Tb}
  for ci in c.blocks
    fill!(ci,zero(Tc))
  end
  for k in 1:length(a.blocks)
    ak = a.blocks[k]
    ci, cj = a.coordinates[k]
    p = b.ptrs[cj]
    bk = b.blocks[p]
    q = c.ptrs[ci]
    ck = c.blocks[q]
    muladd!(ck,ak,bk)
  end
end

function CachedMultiFieldArray(a::MultiFieldArray)
  blocks = [ CachedArray(b) for b in a.blocks ]
  coordinates = a.coordinates
  MultiFieldArray(blocks,coordinates)
end

function _resize_for_mul!(
  c::MultiFieldArray{Tc,1},a::MultiFieldArray{Ta,2},b::MultiFieldArray{Tb,1}) where {Tc,Ta,Tb}
  for k in 1:length(a.blocks)
    ak = a.blocks[k]
    ci, cj = a.coordinates[k]
    q = c.ptrs[ci]
    ck = c.blocks[q]
    setsize!(ck,(size(ak,1),))
  end
end

function _move_cached_arrays!(r::MultiFieldArray,c::MultiFieldArray)
  for  k in 1:length(c.blocks)
    ck = c.blocks[k]
    r.blocks[k] = ck.array
  end
end

function _get_scalar_size(N,blocks,coordinates,block_size)
  try 
     (length(blocks)>0) && (blocks[1])  
  catch  
     return nothing 
  end 
  s=zeros(Int,N)
  visited_coordinates = Vector{Vector{Bool}}(undef,N)
  for i=1:N
     visited_coordinates[i] = fill(false,block_size[i])
  end 
  for (icoord,coord) in enumerate(coordinates)
    for (i,j) in enumerate(coord)
      if (!visited_coordinates[i][j])
        visited_coordinates[i][j]=true
        s[i] += size(blocks[icoord])[i]
      end 
    end 
  end
  Tuple(s) 
end 

function Base.size(a::MultiFieldArray{T,N}) where {T,N}
  if ( a.scalar_size isa Nothing )
    _get_scalar_size(N,a.blocks,a.coordinates,a.block_size)
  else
    a.scalar_size
  end
end

function Base.length(a::MultiFieldArray)
  result=0
  for i=1:length(a.blocks)
    result=result+length(a.blocks[i])
  end 
  result
end

function _get_cumulative_block_sizes(N, 
                                     block_size,
                                     blocks,
                                     coordinates)
  try 
     (length(blocks)>0) && (blocks[1])  
  catch  
     return nothing 
  end 
  result  = Vector{Vector{Int}}(undef,N)
  visited = Vector{Vector{Bool}}(undef,N)
  for i=1:N
    result[i] = fill(0,block_size[i])
    visited[i] = fill(false,block_size[i])
  end 
  for (icoord,coord) in enumerate(coordinates)
    for (i,j) in enumerate(coord)
      if (!visited[i][j])
         visited[i][j]=true
         result[i][j]=size(blocks[icoord])[i]
      end
    end 
  end
  for i=1:length(result)
   for j=2:length(result[i])
     result[i][j] += result[i][j-1]
   end
  end
  result
end

function _find_block_and_local_indices(a::MultiFieldArray{T,N,A},I)  where {T,N,A}
  bc=Vector{Int}(undef,N)
  bi=Vector{Int}(undef,N)
  if (a.cumulative_block_sizes isa Nothing)
    _get_cumulative_block_sizes(N,
                                a.block_size,
                                a.blocks,
                                a.coordinates)
  else
    sizes=a.cumulative_block_sizes
  end 

  for i=1:N
    b=1
    j=1
    _I = I[i]
    if ( _I > sizes[i][j] )
      while ( _I > sizes[i][j] && _I <= sizes[i][j+1] )
        b+=1
        j+=1
      end 
      _I = _I - sizes[i][j-1]
    end
    bc[i] = b
    bi[i] = _I
  end
  (Tuple(bc),Tuple(bi))
end 

function Base.getindex(
  a::MultiFieldArray{T,N},
  I::Vararg{Int,N}) where {T,N}
  @assert length(I) == N  
  (bc,bi) = _find_block_and_local_indices(a,I)
  p=a.ptrs[bc...]
  @assert p > 0 "You are attempting to access a block that is not stored"
  a.blocks[a.ptrs[bc...]][bi...]
end

function Base.getindex(a::MultiFieldArray{T,N},
                     i::Integer) where {T,N}
  cis=CartesianIndices(a.block_size)
  start=1
  for is in cis
    p=a.ptrs[is]
    println(p)
    if (p>0)
      final=start + length(a.blocks[p]) - 1
      if (i>=start && i<=final)
        return a.blocks[p][i-start+1]
      end
      start=final+1
    end
  end
  @assert false "Index out of range"
end

function Base.iterate(a::MultiFieldArray)
  cis=CartesianIndices(a.block_size)
  start=1
  for (i,is) in enumerate(cis)
    p=a.ptrs[is]
    if (p>0)
      final=start+length(a.blocks[p])-1
      entry=a.blocks[p][1]
      next=2
      return (entry,(cis,i,start,final,next))
    end
  end
end

function Base.iterate(a::MultiFieldArray,state)
  cis,i,start,final,current=state
  is=cis[i]
  p=a.ptrs[is]
  if (current >= start && current <= final)
    entry=a.blocks[p][current-start+1]
    return (entry,(cis,i,start,final,current+1))
  else 
    #Search for the next block 
    for j=i+1:length(cis)
      is=cis[j]
      p=a.ptrs[is]
      if (p>0)
        start=final+1
        final=start+length(a.blocks[p])-1
        entry=a.blocks[p][1]
        return (entry,(cis,j,start,final,start+1))
      end
    end 
  end
end
