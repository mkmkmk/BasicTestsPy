# https://github.com/sympy/sympy/wiki/Quick-examples

#%%
from sympy import *
from IPython.display import display, Math
init_printing()

x, y, z, t = symbols('x y z t')

Rational(3,2)*pi + exp(I*x) / (x**2 + y)

x = Symbol('x')

exp(I*x)
exp(I*x).subs(x,pi).evalf()

expr = x + 2*y
display(expr)
expr.__class__
expr.args


exp(pi * sqrt(163)).evalf(50)

display(Math(latex(S('2*4+10',evaluate=False))))
display(Math(latex('exp(x*2)/2')))


((x+y)**2 * (x+1))
((x+y)**2 * (x+1)).expand()

Eq(x**3 + 2*x**2 + 4*x + 8, 0)
solve(Eq(x**3 + 2*x**2 + 4*x + 8, 0), x)
solve(x**3 + 2*x**2 + 4*x + 8, x)

display(solve([Eq(x + 5*y, 2), Eq(-3*x + 6*y, 15)], [x, y]))
display((sin(x)-x)/x**3)
display(limit((sin(x)-x)/x**3, x, 0))

#%%

cos(x**2)**2 / (1+x)
diff(cos(x**2)**2 / (1+x), x)

#---- nieskończoność
display(oo)

#----
n = Symbol('n')

display(Limit(sqrt(1/log(n/(n-1))), n, oo))
display(solve(Limit(sqrt(1/log(n/(n-1))), n, oo)),n)

limit(log(1+n) - n, n, 0)

# %%
import sympy as sp

# Symbole
t = sp.Symbol('t')
y = sp.Function('y')

# Proste równanie: dy/dt = -k*y (rozpad radioaktywny)
k = sp.Symbol('k', positive=True)
eq = sp.Eq(y(t).diff(t), -k*y(t))

display("Równanie:")
display(eq)

# Rozwiązanie
solution = sp.dsolve(eq, y(t))
display("Rozwiązanie:")
display(solution)

# Z warunkiem początkowym
C1 = sp.Symbol('C1')
y0 = sp.Symbol('y0')
particular = solution.subs(C1, y0)
display("Z warunkiem początkowym y(0)=y_0:")
display(particular)
# %%
