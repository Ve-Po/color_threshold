import json
import numpy as np
from math import sqrt, atan2, degrees, radians, sin, cos, exp
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from skimage.color import lab2rgb


# Функции преобразования цветов
def xyz2lab(colors_xyz, white_n_XYZ):
    c = np.array(colors_xyz)
    if len(c.shape) == 1:
        c = c[np.newaxis, :]
    c[:, 0] /= white_n_XYZ[0]
    c[:, 1] /= white_n_XYZ[1]
    c[:, 2] /= white_n_XYZ[2]
    x, y, z = _cie_lab_f(c[:, 0]), _cie_lab_f(c[:, 1]), _cie_lab_f(c[:, 2])
    L = 116. * y - 16
    a = 500. * (x - y)
    b = 200. * (y - z)
    return np.array([L, a, b]).T

def _cie_lab_f(t):
    res = np.zeros(t.shape)
    mask = t > (6 / 29) ** 3
    res[mask] = t[mask] ** (1 / 3)
    res[~mask] = 1 / 3 * (29 / 6) ** 2 * t[~mask] + 4 / 29
    return res



def show_color_difference(lab1, lab2, delta_e):
    rgb1 = lab2rgb(np.array([[lab1]]))[0, 0]
    rgb2 = lab2rgb(np.array([[lab2]]))[0, 0]
    
    fig, ax = plt.subplots(1, 2, figsize=(4, 2))
    ax[0].add_patch(Rectangle((0, 0), 1, 1, color=rgb1))
    ax[1].add_patch(Rectangle((0, 0), 1, 1, color=rgb2))
    ax[0].set_title("Color 1")
    ax[1].set_title("Color 2")
    for a in ax:
        a.axis('off')
    plt.suptitle(f"ΔE₀₀ = {delta_e:.4f}")
    plt.show()

def delta_e_00(lab1, lab2, kL=1, kC=1, kH=1):
    """
    Вычисление цветового различия ΔE₀₀ (CIEDE2000) между двумя цветами в Lab
    Реализация стандарта CIE Delta E 2000
    """
    L1, a1, b1 = lab1
    L2, a2, b2 = lab2
    
    # Шаг 1: Преобразование a* → a' (компенсация нейтральных цветов)
    C1_ab = sqrt(a1**2 + b1**2)
    C2_ab = sqrt(a2**2 + b2**2)
    C_ab_avg = (C1_ab + C2_ab) / 2
    G = 0.5 * (1 - sqrt((C_ab_avg**7) / (C_ab_avg**7 + 25**7)))
    
    a1_prime = (1 + G) * a1
    a2_prime = (1 + G) * a2
    
    # Шаг 2: Расчёт C' (новая насыщенность)
    C1_prime = sqrt(a1_prime**2 + b1**2)
    C2_prime = sqrt(a2_prime**2 + b2**2)
    
    # Шаг 3: Расчёт угла тона h'
    def compute_h_prime(a, b):
        if a == 0 and b == 0:
            return 0
        h_rad = atan2(b, a)
        h_deg = degrees(h_rad)
        return h_deg if h_deg >= 0 else h_deg + 360
    
    h1_prime = compute_h_prime(a1_prime, b1)
    h2_prime = compute_h_prime(a2_prime, b2)
    
    # Шаг 4: Разница по светлоте (ΔL')
    delta_L_prime = L2 - L1
    
    # Шаг 5: Разница по насыщенности (ΔC')
    delta_C_prime = C2_prime - C1_prime
    
    # Шаг 6: Разница по тону (ΔH')
    delta_h_prime = 0
    if C1_prime * C2_prime != 0:
        delta_h_prime = h2_prime - h1_prime
        if abs(delta_h_prime) > 180:
            if delta_h_prime > 0:
                delta_h_prime -= 360
            else:
                delta_h_prime += 360
    
    delta_H_prime = 2 * sqrt(C1_prime * C2_prime) * sin(radians(delta_h_prime / 2))
    
    # Шаг 7: Средние значения (L̄, C̄', H̄')
    L_avg = (L1 + L2) / 2
    C_prime_avg = (C1_prime + C2_prime) / 2
    
    def compute_H_prime_avg(h1, h2, C1, C2):
        if C1 * C2 == 0:
            return h1 + h2
        if abs(h1 - h2) <= 180:
            return (h1 + h2) / 2
        elif (h1 + h2) < 360:
            return (h1 + h2 + 360) / 2
        else:
            return (h1 + h2 - 360) / 2
    
    H_prime_avg = compute_H_prime_avg(h1_prime, h2_prime, C1_prime, C2_prime)
    
    # Шаг 8: Весовые коэффициенты (S_L, S_C, S_H)
    T = (1 - 0.17 * cos(radians(H_prime_avg - 30))
         + 0.24 * cos(radians(2 * H_prime_avg))
         + 0.32 * cos(radians(3 * H_prime_avg + 6))
         - 0.20 * cos(radians(4 * H_prime_avg - 63)))
    
    S_L = 1 + (0.015 * (L_avg - 50)**2) / sqrt(20 + (L_avg - 50)**2)
    S_C = 1 + 0.045 * C_prime_avg
    S_H = 1 + 0.015 * C_prime_avg * T
    
    # Шаг 9: Поправочный член R_T (для синей области)
    delta_theta = 30 * exp(-((H_prime_avg - 275) / 25)**2)
    R_C = 2 * sqrt((C_prime_avg**7) / (C_prime_avg**7 + 25**7))
    R_T = -R_C * sin(radians(2 * delta_theta))
    
    # Итоговый расчёт ΔE₀₀
    delta_E = sqrt(
        (delta_L_prime / (kL * S_L))**2 +
        (delta_C_prime / (kC * S_C))**2 +
        (delta_H_prime / (kH * S_H))**2 +
        R_T * (delta_C_prime / (kC * S_C)) * (delta_H_prime / (kH * S_H))
    )
    
    return delta_E

# Загрузка данных из JSON
with open('input_bfd-m.json') as f:
    data = json.load(f)

# Преобразование всех XYZ цветов в Lab
reference_white = np.array(data['reference_white'])
xyz_colors = np.array(data['xyz'])
print(reference_white )
print(reference_white / reference_white[1])
lab_colors = xyz2lab(xyz_colors, reference_white )

# Создание результата с расстояниями между парами
result = {
    "reference_white": data['reference_white'],
    "pairs": []
}
print(data['pairs'])
for pair in data['pairs']:
    idx1, idx2 = pair
    print(idx1, idx2)
    color1 = lab_colors[idx1]
    print(color1)
    color2 = lab_colors[idx2]
    print(color2)
    distance_00 = delta_e_00(color1.tolist(), color2.tolist())

   # show_color_difference(color1, color2, distance_00)
    result["pairs"].append({
            "pair": pair,
            "lab1": color1.tolist(),
            "lab2": color2.tolist(),
            "delta_e_00": distance_00
        })

# Сохранение результата в JSON
with open('output_bfd-m.json', 'w') as f:
    json.dump(result, f, indent=2)

print("Преобразование завершено. Результат сохранен в output_bfd.json")

