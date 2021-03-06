module CompressedArraysTests

using Test
using Gridap.Arrays

values = [10,20,31]
ptrs = [1,2,3,3,2,2]
a = CompressedArray(values,ptrs)
r = values[ptrs]
test_array(a,r)

b = apply(-,a)
test_array(b,-r)
@test isa(b,CompressedArray)

c = apply(*,a,b)
test_array(c,r.*(-r))
@test isa(c,CompressedArray)

k = CompressedArray([zero,+,-],copy(ptrs))
r = collect(CompressedArray([0,20,-31],ptrs))
c = apply(k,a)
test_array(c,r)
@test isa(c,CompressedArray)

k = CompressedArray([zero,+,-],ptrs)
r = collect(CompressedArray([0,20,-31],ptrs))
c = apply(k,a)
test_array(c,r)
@test isa(c,CompressedArray)

end # module
