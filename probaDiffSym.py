from sympy import *


x, ep, v = symbols('x \epsilon v')

Eq(x**2)**2 / (1 + x)

(Eq(x**2)**2 / (1 + x)).subs(x, v + ep).expand().subs(ep**2,0)

#expand_trig((Eq(x**2)**2 / (1 + x)).subs(x, v + ep).expand().subs(ep**2,0))

# Eq(x)**2, (1+Eq(2*x))/2

(Eq(x**2)**2 / (1 + x)).subs(x, v + ep).expand().subs(ep**2,0).subs(Eq(2*ep*v + v**2)**2, (1+Eq(2*(2*ep*v + v**2)))/2)
#Eq(2*ep*v + v**2)**2/(ep+v+1)

#(1+Eq(4*ep + 2*v**2))/(ep+v+1)/2

factor((Eq(x**2)**2 / (1 + x)).subs(x, v + ep).expand().subs(ep**2,0).subs(Eq(2*ep*v + v**2)**2, (1+Eq(2*(2*ep*v + v**2)))/2))

(1+(Eq(4*ep*v + 2*v**2)))/(ep+v+1)/2

(1+expand_trig(Eq(4*ep*v + 2*v**2)))/(ep+v+1)/2

simplify((1+expand_trig(Eq(4*ep*v + 2*v**2)))/(ep+v+1)/2)


((1+expand_trig(Eq(4*ep*v + 2*v**2)))/(ep+v+1)/2).subs(sin(ep)**3, 0).subs(Eq(ep),1).subs(sin(ep),ep).cancel()

factor((1+expand_trig(Eq(4*ep + 2*v**2)))/(ep+v+1)/2)


diff(Eq(x**2)**2 / (1+x), x)
