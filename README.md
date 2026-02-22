Enlace de video: 

https://drive.google.com/file/d/1iFkG4cmdHaiDmvvtgYo-3078bwMi9XId/view?usp=sharing


# Preprocesamiento de datos y preguntas técnicas  
**Materia:** Minería de Datos y Ciencia de Datos  
**Profesor:** M.C. Eduardo Francisco Sánchez Ocampo  
**Integrantes del equipo:**  
- Pablo García Espejo  
- Diego Arath Franco Herrera  
- Miguel Ángel Guerrero Alvarez  
- Manuel de la Torre  
- Giulian Thibaut Elías-Libera  

**Fecha:** 19 de febrero de 2026  


En el caso de que no se puedan apreciar en los documentos igualmente se puede acceder a todo el contenido por los siguientes enlaces: 

Presentación pptx: 

https://docs.google.com/presentation/d/1IgAQxFV7NbsIPbVQBFDuvol2egAkaS7s/edit?usp=sharing&ouid=111078524243763443739&rtpof=true&sd=true

MiniProyecto1 colab: 

https://colab.research.google.com/drive/16xR8O35mLycySiIdOvg6cDYGibQUgeRI?usp=sharing

Video presentación: 

https://drive.google.com/file/d/1iFkG4cmdHaiDmvvtgYo-3078bwMi9XId/view?usp=sharing


---

## Introducción

En minería de datos, el preprocesamiento define la calidad del modelo final. Un dataset con valores faltantes o variables categóricas sin codificar puede provocar errores de entrenamiento o resultados engañosos. En este documento se resuelven ejercicios prácticos de imputación, codificación de variables, separación de datos en entrenamiento y prueba, y escalamiento. Además, se contestan preguntas técnicas relacionadas con normalización, estandarización, selección de encoders, prevención de data leakage, cálculo de MSE e implementación de una función para imputación por media.

---

## 1) Dataset de salud: imputación de NaNs (Sobrepeso)

**Objetivo:** Rellenar valores faltantes con una estrategia adecuada para dejar el dataset listo para ML.

**Estrategia aplicada:**
- Variables numéricas (`Edad`, `Altura`, `Peso`) con imputación por **media**.
- Variable categórica (`Actividad`) con imputación por **moda** (valor más frecuente).

> Nota: en el enunciado original la lista de `Actividad` puede aparecer incompleta. En este script se deja como `np.nan` para imputar correctamente.

```python
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

data = {
    "Edad": [25, 30, np.nan, 40, 35, 50, np.nan, 28],
    "Altura": [175, 180, 165, np.nan, 160, 170, 168, 172],
    "Peso": [70, np.nan, 55, 85, 60, np.nan, 72, 68],
    "Actividad": ["Moderado", "Sedentario", "Activo", "Moderado", "Sedentario", "Activo", np.nan, "Moderado"],
    "Sobrepeso": ["No", "Sí", "No", "Sí", "No", "Sí", "Sí", "No"]
}
df = pd.DataFrame(data)

print("Dataset original:")
print(df)

# Imputación numérica por media
imp_num = SimpleImputer(missing_values=np.nan, strategy="mean")
df[["Edad", "Altura", "Peso"]] = imp_num.fit_transform(df[["Edad", "Altura", "Peso"]])

# Imputación categórica por moda
imp_cat = SimpleImputer(missing_values=np.nan, strategy="most_frequent")
df[["Actividad"]] = imp_cat.fit_transform(df[["Actividad"]])

print("\nDataset con NaNs imputados:")
print(df)
```

---

## 2) One-Hot Encoding y dataset final totalmente numérico

**Objetivo:** Transformar la columna categórica `Actividad` a formato numérico usando One-Hot Encoding y codificar `Sobrepeso` (Sí/No) a números.

**Resultados esperados:**
- `X` totalmente numérico con One-Hot en `Actividad`.
- `y` numérico (0/1) para el objetivo.

```python
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

data = {
    "Edad": [25, 30, np.nan, 40, 35, 50, np.nan, 28],
    "Altura": [175, 180, 165, np.nan, 160, 170, 168, 172],
    "Peso": [70, np.nan, 55, 85, 60, np.nan, 72, 68],
    "Actividad": ["Moderado", "Sedentario", "Activo", "Moderado", "Sedentario", "Activo", np.nan, "Moderado"],
    "Sobrepeso": ["No", "Sí", "No", "Sí", "No", "Sí", "Sí", "No"]
}
df = pd.DataFrame(data)

# Imputación
imp_num = SimpleImputer(missing_values=np.nan, strategy="mean")
df[["Edad", "Altura", "Peso"]] = imp_num.fit_transform(df[["Edad", "Altura", "Peso"]])

imp_cat = SimpleImputer(missing_values=np.nan, strategy="most_frequent")
df[["Actividad"]] = imp_cat.fit_transform(df[["Actividad"]])

# Separación X e y
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# One-Hot en "Actividad" (columna índice 3 de X)
ct = ColumnTransformer(
    transformers=[("encoder", OneHotEncoder(), [3])],
    remainder="passthrough"
)
X = np.array(ct.fit_transform(X))

# Label encoding del target
le = LabelEncoder()
y = le.fit_transform(y)

print("X final (numérico):")
print(X)
print("\ny final (numérico):")
print(y)
```

---

## 3) Label Encoding a las columnas y dataset final numérico

**Objetivo:** Codificar variables categóricas con Label Encoding para obtener un dataset completamente numérico.

**Nota técnica:** Label Encoding en una variable categórica nominal puede introducir un orden artificial. Se aplica aquí porque el ejercicio lo solicita.

```python
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

data = {
    "Edad": [25, 30, np.nan, 40, 35, 50, np.nan, 28],
    "Altura": [175, 180, 165, np.nan, 160, 170, 168, 172],
    "Peso": [70, np.nan, 55, 85, 60, np.nan, 72, 68],
    "Actividad": ["Moderado", "Sedentario", "Activo", "Moderado", "Sedentario", "Activo", np.nan, "Moderado"],
    "Sobrepeso": ["No", "Sí", "No", "Sí", "No", "Sí", "Sí", "No"]
}
df = pd.DataFrame(data)

# Imputación
imp_num = SimpleImputer(missing_values=np.nan, strategy="mean")
df[["Edad", "Altura", "Peso"]] = imp_num.fit_transform(df[["Edad", "Altura", "Peso"]])

imp_cat = SimpleImputer(missing_values=np.nan, strategy="most_frequent")
df[["Actividad"]] = imp_cat.fit_transform(df[["Actividad"]])

# Label Encoding para columnas categóricas
le_act = LabelEncoder()
df["Actividad"] = le_act.fit_transform(df["Actividad"])

le_y = LabelEncoder()
df["Sobrepeso"] = le_y.fit_transform(df["Sobrepeso"])

print("Dataset final (todo numérico):")
print(df)

print("\nMapa Actividad:", dict(zip(le_act.classes_, le_act.transform(le_act.classes_))))
print("Mapa Sobrepeso:", dict(zip(le_y.classes_, le_y.transform(le_y.classes_))))
```

---

## 4) Train-Test Split 75/25 y mostrar X_train, X_test, y_train, y_test

**Objetivo:** Dividir un dataset de clientes en entrenamiento y prueba (75% / 25%).

```python
import pandas as pd
from sklearn.model_selection import train_test_split

clientes = pd.DataFrame({
    "Edad":   [25, 30, 35, 40, 45, 50, 28, 33],
    "Salario":[50000, 60000, 58000, 62000, 55000, 70000, 48000, 61000],
    "Compró": ["Sí", "No", "Sí", "No", "Sí", "No", "Sí", "No"]
})

X = clientes[["Edad", "Salario"]].values
y = clientes["Compró"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

print("X_train:")
print(X_train)
print("\nX_test:")
print(X_test)
print("\ny_train:")
print(y_train)
print("\ny_test:")
print(y_test)
```

---

## 5) StandardScaler con Split 80/20 y mostrar X escalado

**Objetivo:** Dividir el dataset (80% / 20%) y aplicar estandarización.

**Regla aplicada:** Ajustar (`fit`) el escalador solo con entrenamiento para evitar data leakage.

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.DataFrame({
    "Edad":   [25, 30, 35, 40, 45],
    "Salario":[50000, 60000, 58000, 62000, 55000]
})

X = df.values

X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

sc = StandardScaler()
X_train_scaled = sc.fit_transform(X_train)
X_test_scaled = sc.transform(X_test)

print("X_train (escalado):")
print(X_train_scaled)
print("\nX_test (escalado):")
print(X_test_scaled)
```

---

# Preguntas técnicas

## 1) ¿Cuál es la diferencia entre normalización y estandarización?

**Estandarización (Standardization):**
- Centra en media 0 y escala a desviación estándar 1.
- Fórmula:
  \[
  x' = \frac{x - \mu}{\sigma}
  \]
- Se implementa con `StandardScaler`.

**Normalización (Min-Max Scaling):**
- Escala los datos a un rango, normalmente [0, 1].
- Fórmula:
  \[
  x' = \frac{x - x_{\min}}{x_{\max} - x_{\min}}
  \]
- Se implementa con `MinMaxScaler`.

**Uso práctico:**
- Estandarización: útil cuando el modelo es sensible a la escala (distancias, gradiente).
- Normalización: útil cuando se requiere un rango fijo, por ejemplo, para ciertos modelos o interpretaciones.

---

## 2) ¿En qué casos se prefiere OneHotEncoder sobre LabelEncoder?

**Se prefiere OneHotEncoder cuando:**
- La variable categórica es **nominal** (sin orden real).
- Se quiere evitar que el modelo interprete un “orden” artificial.

**Se usa LabelEncoder cuando:**
- Se necesita codificar el objetivo `y` (por ejemplo “Sí/No”) a números.
- La variable categórica es **ordinal** y existe un orden real (en ese caso se recomienda mapear con criterio).

---

## 3) ¿Por qué no se deben escalar los datos antes de hacer el train-test split?

Porque el escalamiento usa estadísticas globales (media, desviación estándar, mínimos, máximos). Si se calcula con todo el dataset antes del split, se introduce información del conjunto de prueba en el entrenamiento. Eso genera **data leakage**, y las métricas se vuelven más altas de lo que deberían. La forma correcta es:
1) separar train/test,
2) ajustar el escalador con `X_train`,
3) transformar `X_test` con ese mismo escalador.

---

## 4) Calcule manualmente el MSE

**Definición:**
\[
MSE = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2
\]

**Ejemplo:**
- \( y = [3, 5, 2] \)
- \( \hat{y} = [2, 5, 4] \)

1) Errores: \([1, 0, -2]\)  
2) Cuadrados: \([1, 0, 4]\)  
3) Promedio:
\[
MSE = \frac{1+0+4}{3} = \frac{5}{3} \approx 1.6667
\]



## 5) Implemente una función que encuentre la media de una columna y sustituya ese valor en todos los NaNs

### Opción A: con Pandas (DataFrame)

```python
import pandas as pd
import numpy as np

def fill_nan_with_mean(df: pd.DataFrame, col: str) -> float:
    mean_value = df[col].mean(skipna=True)
    df[col] = df[col].fillna(mean_value)
    return mean_value

# Ejemplo
df = pd.DataFrame({"Edad":[25, np.nan, 35, 40]})
m = fill_nan_with_mean(df, "Edad")
print("Media usada:", m)
print(df)
```

### Opción SimpleImputer

Como se alcanza a apreciar en el formulario con el que trabajamos.
