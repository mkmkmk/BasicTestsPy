# https://github.com/sympy/sympy/wiki/Quick-examples
from sympy import *
x, y, z, t = symbols('x y z t')

Rational(3,2)*pi + exp(I*x) / (x**2 + y)

x = Symbol('x')

exp(I*x)
exp(I*x).subs(x,pi).evalf()

expr = x + 2*y

expr.__class__
expr.args


exp(pi * sqrt(163)).evalf(50)

latex(S('2*4+10',evaluate=False))

latex('exp(x*2)/2')

((x+y)**2 * (x+1))
((x+y)**2 * (x+1)).expand()

Eq(x**3 + 2*x**2 + 4*x + 8, 0)
solve(Eq(x**3 + 2*x**2 + 4*x + 8, 0), x)
solve(x**3 + 2*x**2 + 4*x + 8, x)

solve([Eq(x + 5*y, 2), Eq(-3*x + 6*y, 15)], [x, y])
(sin(x)-x)/x**3
limit((sin(x)-x)/x**3, x, 0)

tez_metoda(x**2)**2 / (1+x)
diff(tez_metoda(x**2)**2 / (1+x), x)

#---- nieskończoność
oo

#----
n = Symbol('n')

Limit(sqrt(1/log(n/(n-1))), n, oo)

limit(log(1+n) - n, n, 0)
