import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dias = np.array(["Segunda", "Terça", "Quarta", "Quinta", "Sexta", "Sábado", "Domingo"])
temperaturas = np.random.randint(20, 36, size=7)  

print('Dias: ', dias)
print('Temperaturas:', temperaturas)
print()

df = pd.DataFrame({"Dia": dias, "Temp": temperaturas})
print(df)
print("Média da semana:", df["Temp"].mean())
print()

plt.plot(dias, temperaturas, color='purple',  label='Temperatura')
plt.xlabel('Dias')
plt.ylabel('Temperatura (°C)')
plt.title('Variação de temperatura durante a semana')
plt.legend()
plt.show()



