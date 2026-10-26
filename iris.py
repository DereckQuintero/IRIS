from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 1. Cargar el dataset de Iris incluido en scikit-learn
iris = load_iris()
X = iris.data  # Las 4 características (largo/ancho de sépalo y pétalo)
y = iris.target # Las 3 clases a predecir (0: Setosa, 1: Versicolor, 2: Virginica)

# 2. Dividir los datos: 80% para entrenar la IA, 20% para evaluarla
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Crear el modelo (Clasificador de 3 vecinos más cercanos)
modelo = KNeighborsClassifier(n_neighbors=3)

# 4. Entrenar la Inteligencia Artificial
modelo.fit(X_train, y_train)

# 5. Hacer predicciones con los datos de prueba
predicciones = modelo.predict(X_test)

# 6. Evaluar la precisión del sistema
precision = accuracy_score(y_test, predicciones)
print(f"Precisión del modelo: {precision * 100:.2f}%\n")

# --- PRUEBA EN VIVO ---
# Si encontramos una flor nueva y medimos sus características en centímetros:
# [Largo Sépalo, Ancho Sépalo, Largo Pétalo, Ancho Pétalo]
flor_desconocida = [[5.1, 3.5, 1.4, 0.2]] 
prediccion_nueva = modelo.predict(flor_desconocida)

nombre_especie = iris.target_names[prediccion_nueva[0]]
print(f"La nueva flor ha sido clasificada como: {nombre_especie.capitalize()}")