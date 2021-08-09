import numpy
import sparse

a_idx = numpy.array([[0, 0], [1, 1]], dtype = numpy.int32)
a_val = numpy.array([0.6, 0.8], dtype = numpy.float16)

sp = sparse.COO(coords = a_idx.T, data = a_val, shape=(5, 2))


print(len(sp))

print(sp.todense())

print(sp[0].todense())
