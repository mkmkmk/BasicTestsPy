# %%
import numpy as np
import matplotlib.pyplot as plt

# Punkty kontrolne krzywej Béziera
P0 = np.array([63, 3])
P2 = np.array([3, 13])
P1 = np.array([6, 10])
P3 = np.array([65, 30])

# Lista do przechowywania punktów krzywej
curve_points = []

# Pętla generująca punkty krzywej
for t in np.linspace(0, 1, 10):
    # Wzór na krzywą Béziera stopnia 3
    x = (1-t)**3 * P0[0] + 3*(1-t)**2*t * P1[0] + 3*(1-t)*t**2 * P2[0] + t**3 * P3[0]
    y = (1-t)**3 * P0[1] + 3*(1-t)**2*t * P1[1] + 3*(1-t)*t**2 * P2[1] + t**3 * P3[1]
    print(x,y)
    curve_points.append([x, y])

# Konwersja do numpy array dla łatwiejszego plotowania
curve_points = np.array(curve_points)

# Wizualizacja
plt.figure(figsize=(10, 6))
plt.plot(curve_points[:, 0], curve_points[:, 1], 'b-', linewidth=2, label='Krzywa Béziera')
plt.plot([P0[0], P1[0], P2[0], P3[0]], [P0[1], P1[1], P2[1], P3[1]], 
         'ro--', linewidth=1, markersize=8, label='Punkty kontrolne')
plt.grid(True, alpha=0.3)
plt.legend()
plt.title('Krzywa Béziera stopnia 3')
plt.xlabel('x')
plt.ylabel('y')
plt.axis('equal')
plt.show()

# Wyświetlenie kilku przykładowych punktów
print("Przykładowe punkty krzywej (t, x, y):")
for i, t in enumerate(np.linspace(0, 1, 5)):
    print(f"t={t:.2f}: x={curve_points[int(i*24.75)][0]:.3f}, y={curve_points[int(i*24.75)][1]:.3f}")
# %%
