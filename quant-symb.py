from sympy import *
from sympy import Matrix
from sympy.physics.quantum import TensorProduct

c1, c2 = symbols('c1 c2')
C = Matrix([c1,c2])
C

p1, p2 = symbols('p1 p2')
P = Matrix([p1,p2])
P

TensorProduct(C, P)

cnot = Matrix([[1,0,0,0], [0,1,0,0], [0,0,0,1] , [0,0,1,0]])
cnot
cnot * TensorProduct(C, P)

cnot * Matrix(symbols('b00 b01 b10 b11'))

dt = Symbol('delta')

q = Matrix(symbols('q0, q1'))
q0 = Matrix([1, 0])
q1 = Matrix([0, 1])

R = Matrix([[cos(dt), -sin(dt)], [sin(dt), cos(dt)]])
R * q0

TensorProduct(R * q0, q0)
cnot * TensorProduct(R * q0, q0)
