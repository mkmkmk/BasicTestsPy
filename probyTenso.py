"""

https://docs.scipy.org/doc/numpy/reference/generated/numpy.tensordot.html#numpy.tensordot

"""

import numpy as np

#--------------------------------
A = np.arange(4 * 9).reshape(9, 4) 
B = np.arange(2 * 4).reshape(4, 2)

A @ B
np.tensordot(A, B, [[1], [0]])

#--------------------------------
A = np.arange(4 * 5).reshape(5, 4) 
B = np.arange(4 * 5).reshape(4, 5)

A@B
np.dot(A,B)
R = np.tensordot(A, B, [[1], [0]])
A.shape
B.shape
R.shape

np.tensordot(A, B, [[1,0], [0,1]])

R = np.zeros((5,5))
for x in range(5):
    for y in range(5):
        for i in range(4):
            R[x,y] += A[x,i] * B[i,y]

R = 0
for j in range(5):
    r0 = 0
    for i in range(4):
        r0 += A[j,i] * B[i,j]
    R+=r0
    print(r0)
R
sum(np.diag(np.dot(A, B)))
np.sum(A * np.transpose(B))

#--------------------------------

A3 = np.array([A,A,A])
B2 = np.array([B,B])
A3.shape
B2.shape

A.shape
B2.shape
R = np.tensordot(A, B2, [[1], [1]])
R.shape

R = np.tensordot(A3, B2, [[2], [1]])
R.shape

R[0][0]


#------

#--------------------------------
_A = np.arange(4 * 9).reshape(9, 4) 
flip = [slice(None)] * 2
flip[0] = slice(None, None, -1)
_A
_A[tuple(flip)]


#--------------------------------
A.shape
B.shape


np.all(B[2, 0, 12:17, 12:17] == B_view[2, 0, :, :, 12, 12])

ref.shape

A_view.shape
B_view.shape

cmp2 = np.tensordot(A_view, B_view, all_axes)

cmp2.shape

ref[0, 0, 12, 12]
cmp2[0, 0, 12, 12]

np.max(ref - cmp2)
np.max(np.abs(ref))
np.min(np.abs(ref))

ref[0, 0, 12, 12]
cmp2[0, 0, 12, 12]

A_sel = A_view[0,0]
B_sel = B_view[0,0,:,:,12,12]
np.tensordot(A_sel, B_sel, [[0,1],[0,1]])
sum(np.diag(np.dot(A_sel, np.transpose(B_sel)))) 
np.sum(A_sel * B_sel)


ref[0, 0, 10, 10]
cmp2[0, 0, 10, 10]
A_sel = A_view[0,0]
B_sel = B_view[0,0,:,:,10,10]
np.tensordot(A_sel, B_sel, [[0,1],[0,1]])
sum(np.diag(np.dot(A_sel, np.transpose(B_sel)))) 
np.sum(A_sel * B_sel)


#-------------------------------- 1D
A.shape
B.shape

B[0, 0, 0:20].shape
B_view[0, 0, 0, 0:20].shape
(B[0, 0, 0:40] - B_view[0, 0, 0, 0:20])


