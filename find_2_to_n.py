"""

Is there a power of 2 whose decimal digits start with the decimal digits of k?
https://scottaaronson.blog/?p=9732

"""

# %%
import numpy as np
import time

# import sys
# sys.set_int_max_str_digits(1000000)

def frac(x): return x - np.floor(x)

k = 3141
k = 41943
k = 134078
k = 1980
k = 160580
k = 16051980
k = 2**40
k = 314159

a = frac(np.log10(k))
b = frac(np.log10(k + 1))

log10_2 = np.log10(2)
r_max  = 2 **27
max_print_dig = 100

next_print =  time.time() + 10

for n in range(1, r_max):

    tm = time.time()
    if tm >= next_print:
        print(f"{(n / r_max) * 100:.1f}%", end='\r')
        next_print = tm + 10

    frac_part = frac(n * log10_2)

    if a <= frac_part <= b:
        print()
        print(f"k = {k}")
        print(f"n = {n}")
        rd = int(n * log10_2) + 1
        if rd <= max_print_dig:
            power = 2**n
            print(f"2^{n} = {power}")
        else:
            first_digits = int(10**(frac_part) * 10**(max_print_dig-1))
            print(f"2^{n} = {first_digits} ... ({rd} digits)")
        break

# %%
