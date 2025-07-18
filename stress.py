import numpy as np
import json
import matplotlib.pyplot as plt

cielab2000 = 'output_witt.json' # в формате cielab
real = 'input_witt.json'  # в формате  эксп данных


with open(cielab2000, 'r') as f:
     data_cielab2000 = json.load(f)

with open(real, 'r') as f:
     data_real = json.load(f)
     
delta_cielab= np.array([pair['delta_e_00'] for pair in data_cielab2000['pairs']])
delta_real = np.array(data_real['dv'])
y = delta_real
x = delta_cielab


print(len(x), len(y))


def stress(x, y):

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    
    k = np.dot(x, y) / np.dot(x, x)
    num1 = np.sqrt(np.sum((k * x - y) ** 2))
    num2 = np.sqrt(np.sum(y ** 2))
    
    stress_value = (num1 / num2)
    return stress_value, k
t, k = stress(x, y)

print(t)
