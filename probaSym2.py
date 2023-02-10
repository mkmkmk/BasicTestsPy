# https://github.com/sympy/sympy/wiki/Quick-examples
from sympy import *
x = Symbol('x')
y, z, t = symbols('y z t')
Rational(3,2)*pi + exp(I*x) / (x**2 + y)
((x+y)**2 * (x+1))
((x+y)**2 * (x+1)).expand()
solve([Eq(x + 5*y, 2), Eq(-3*x + 6*y, 15)], [x, y])
tez_metoda(x**2)**2 / (1+x)
diff(tez_metoda(x**2)**2 / (1+x), x)
x**2 * tez_metoda(x)
integrate(x**2 * tez_metoda(x), x)

f = Function('f')
Eq(Derivative(f(x),x,x) + f(x), 0)
dsolve(Eq(Derivative(f(x),x,x) + f(x), 0), f(x))
f = Function('f')
t, H = symbols('t H')
Eq(1j*Derivative(f(t),t), H*f(t))
dsolve(Eq(1j*Derivative(f(t),t), H*f(t)), f(t))

r = Function('r')
k, t = symbols('k t')
Eq(Derivative(r(t),t), k*r(t))
dsolve(Eq(Derivative(r(t),t), k*r(t)), r(t))
dsolve(Eq(1j*Derivative(r(t),t), k*r(t)), r(t))


Integral(x, x)
integrate(x, x)
2+2

# -----------------------
(Rational(21,92) * Integral(4*x**2-3*x+4, (x,1,3)))**2*Rational(191,2)*(80+3**5*7**2)*Integral(x/ (1+3*x)**.5, (x,0,5))
(Rational(21,92) * Integral(4*x**2-3*x+4, (x,1,3)))**2*Rational(191,2)*(80+3**5*7**2)*Integral(x/ (1+3*x)**Rational('0.5'), (x,0,5))
(Rational(21,92) * integrate(4*x**2-3*x+4, (x,1,3)))**2 * Rational(191,2)*(80+3**5*7**2) * integrate(x / (1+3*x)**Rational('0.5'), (x,0,5))

Integral(x / (1+3*x)**Rational('0.5'), (x,0,5))
integrate(x / (1+3*x)**Rational('0.5'),  (x,0,5))

Integral(4*x**2-3*x+4, (x,1,3))
integrate(4*x**2-3*x+4, (x,1,3))

# -----------------------
from sympy import *
from sympy import Matrix
from sympy.physics.quantum import TensorProduct

a11, a12, a21, a22 = symbols('a11 a12 a21 a22')
b11, b12, b21, b22 = symbols('b11 b12 b21 b22')
x, a, b, c, d = symbols('x, a b c d')

A = Matrix([[a11,a12],[a21,a22]])
A
B = Matrix([[b11,b12],[b21,b22]])
B

TensorProduct(A, B)

A**x
A**0

A = Matrix([[a,0],[0,0]])
A = Matrix([[a,0],[0,b]])
A = Matrix([[a,0, 0],[0,b,0],[0,0,c]])
A**x

#---------------------------------
from sympy import *

h11, h12, h13, h21, h22, h23, h31, h32, h33 = symbols('h11, h12, h13, h21, h22, h23, h31, h32, h33')
h11, h12, h13, h14, h21, h22, h23, h24, h31, h32, h33, h34 = symbols('h11, h12, h13, h14, h21, h22, h23, h24, h31, h32, h33, h34')
H = Matrix([[h11, h12, h13, h14], [h21, h22, h23, h24], [h31, h32, h33, h34]])
H

y1, y2, y3, y4 = symbols('y1, y2, y3, y4')
y = Matrix([y1, y2, y3, y4])
y

inv_quick(H * transpose(H)) * H * y
s1, s2, s3, s4 = symbols('sigma1, sigma2, sigma3, sigma4')

W = diag(1/sqrt(s1),1/sqrt(s2),1/sqrt(s3), 1/sqrt(s4))
W

inv_quick(H * W * transpose(H)) * H * W * y
simplify(inv_quick(H * W * transpose(H)) * H * W * y)


sig = Matrix([1/sqrt(s1),1/sqrt(s2),1/sqrt(s3), 1/sqrt(s4)])
sig
tsig = transpose(sig)
y2 = y.multiply_elementwise(sig)
y2

H2 = H.multiply_elementwise(Matrix([tsig, tsig, tsig]))
H2

inv_quick(H2 * transpose(H2)) * H2 * y2
simplify(inv_quick(H2 * transpose(H2)) * H2 * y2)

(inv_quick(H2 * transpose(H2)) * H2 * y2) - (inv_quick(H * W * transpose(H)) * H * W * y)
simplify((inv_quick(H2 * transpose(H2)) * H2 * y2) - (inv_quick(H * W * transpose(H)) * H * W * y))


#
a, b, c, d = symbols('a, b, c, d')
simplify( (a+b)**2 - (a**2 + 2*a*b + b**2) ) == 0

simplify( (a+b)**2 + 2*(a+b)*c + c**2 - (a+b+c)**2 ) == 0












#
