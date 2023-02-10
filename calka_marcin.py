from sympy import *
x = Symbol('x')

(Rational(21,92) * Integral(4*x**2-3*x+4, (x,1,3)))**2*Rational(191,2)*(80+3**5*7**2)*Integral(x/ (1+3*x)**Rational('0.5'), (x,0,5))
(Rational(21,92) * integrate(4*x**2-3*x+4, (x,1,3)))**2 * Rational(191,2)*(80+3**5*7**2) * integrate(x / (1+3*x)**Rational('0.5'), (x,0,5))

Integral(x / (1+3*x)**Rational('0.5'), (x,0,5))
integrate(x / (1+3*x)**Rational('0.5'),  (x,0,5))

Integral(4*x**2-3*x+4, (x,1,3))
integrate(4*x**2-3*x+4, (x,1,3))
