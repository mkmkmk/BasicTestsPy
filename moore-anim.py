import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Polygon
import matplotlib.animation as animation
from IPython.display import HTML

# Ustawienie wykresu
fig, ax = plt.subplots(figsize=(10, 8))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.set_aspect('equal')
ax.set_title('Animacja maszyny Christophera Moore\'a', fontsize=14)

# Definiowanie przeszkód (wielokąty)
obstacles = [
    Polygon([[2, 2], [3, 2], [2.5, 3]], closed=True, color='darkblue'),
    Polygon([[7, 3], [8, 3], [8, 4], [7, 4]], closed=True, color='darkblue'),
    Polygon([[4, 6], [5, 6], [5, 7], [4, 7]], closed=True, color='darkblue'),
    Polygon([[1, 8], [2, 7], [3, 8], [2, 9]], closed=True, color='darkblue'),
    Polygon([[8, 8], [9, 7], [9, 9]], closed=True, color='darkblue')
]

# Dodanie ścian (granic)
walls = [
    Rectangle((0, 0), 10, 0.2, color='gray'),  # dolna ściana
    Rectangle((0, 9.8), 10, 0.2, color='gray'),  # górna ściana
    Rectangle((0, 0), 0.2, 10, color='gray'),  # lewa ściana
    Rectangle((9.8, 0), 0.2, 10, color='gray')  # prawa ściana
]

# Dodanie przeszkód i ścian do wykresu
for obstacle in obstacles:
    ax.add_patch(obstacle)
for wall in walls:
    ax.add_patch(wall)

# Początkowa pozycja i prędkość kuli
ball_pos = np.array([1.0, 5.0])
ball_velocity = np.array([0.2, 0.15])
ball_radius = 0.2

# Tworzenie kuli
ball = Circle(ball_pos, ball_radius, color='red')
ax.add_patch(ball)

# Ślad trajektorii
trajectory_x = [ball_pos[0]]
trajectory_y = [ball_pos[1]]
trajectory_line, = ax.plot(trajectory_x, trajectory_y, 'r-', alpha=0.5, linewidth=1)

# Funkcja do sprawdzania kolizji
def check_collision(pos, vel, obstacles, walls, radius):
    new_vel = vel.copy()
    
    # Kolizja z poziomymi ścianami
    if pos[1] - radius < 0.2:
        new_vel[1] = abs(new_vel[1])
    elif pos[1] + radius > 9.8:
        new_vel[1] = -abs(new_vel[1])
    
    # Kolizja z pionowymi ścianami
    if pos[0] - radius < 0.2:
        new_vel[0] = abs(new_vel[0])
    elif pos[0] + radius > 9.8:
        new_vel[0] = -abs(new_vel[0])
    
    # Uproszczone sprawdzanie kolizji z przeszkodami
    for obstacle in obstacles:
        path = obstacle.get_path()
        vertices = path.vertices
        center = np.mean(vertices, axis=0)
        dist = np.linalg.norm(pos - center)
        
        if dist < 1.0:  # Przybliżona odległość kolizji
            # Odbicie - uproszczone
            dir_to_center = pos - center
            dir_to_center = dir_to_center / np.linalg.norm(dir_to_center)
            new_vel = vel - 2 * np.dot(vel, dir_to_center) * dir_to_center
            break
            
    return new_vel

# Funkcja animacji
def animate(i):
    global ball_pos, ball_velocity, trajectory_x, trajectory_y
    
    # Sprawdzenie kolizji
    new_velocity = check_collision(ball_pos, ball_velocity, obstacles, walls, ball_radius)
    
    # Jeśli nastąpiła kolizja, zaznacz punkt
    if not np.array_equal(ball_velocity, new_velocity):
        ax.plot(ball_pos[0], ball_pos[1], 'yo', markersize=8)
    
    ball_velocity = new_velocity
    
    # Aktualizacja pozycji
    ball_pos = ball_pos + ball_velocity
    ball.center = ball_pos
    
    # Aktualizacja trajektorii
    trajectory_x.append(ball_pos[0])
    trajectory_y.append(ball_pos[1])
    trajectory_line.set_data(trajectory_x, trajectory_y)
    
    return ball, trajectory_line

# Tworzenie animacji
ani = animation.FuncAnimation(fig, animate, frames=100, interval=100, blit=True)

# Dodanie legendy
ax.text(0.5, 0.05, 'Przeszkody (odbijające ściany)', transform=ax.transAxes, fontsize=12)
ax.text(0.5, 0.02, 'Czerwona linia: trajektoria kuli', transform=ax.transAxes, fontsize=12)
ax.text(0.5, -0.01, 'Żółte punkty: miejsca odbicia (kroki obliczeniowe)', transform=ax.transAxes, fontsize=12)

plt.tight_layout()
#plt.close()  # Zapobiega wyświetleniu statycznego obrazu
#print(plt)
# Wyświetlenie animacji
#HTML(ani.to_jshtml())
plt.show()

