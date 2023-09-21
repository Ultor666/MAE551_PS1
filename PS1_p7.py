import numpy as np
import sympy as sp
from sympy import *
x=sp.symbols('x')
f1 = sp.ln(x**2-sp.sin(x))
f2 = sp.cos(2*x - 3*sp.exp(x))
der_f1=sp.diff(f1,x)
der_f2=sp.diff(f2,x)
print("Derivative of ln(x^2-sin(x):")
print(der_f1)
print("Derivative of cos(2*x - 3*e^x):")
print(der_f2)