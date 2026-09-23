# Modelo de Riesgo de Crédito — Detección de malos pagadores (Morosos)

Modelo de *credit scoring* para predecir la probabilidad de impago de solicitantes de tarjeta de crédito, optimizado para **minimizar el costo de negocio real** (no solo la precisión estadística).

> **Resultado clave:** el modelo final, con el umbral de decisión calibrado al costo de negocio (0.41), **reduce el costo esperado de la cartera en ~29%** frente a usar el umbral por defecto (0.50), detectando al **94.3% de los morosos reales** en el set de prueba.


![umbral_decision](./images/umbral_decision.png)
- Umbral óptimo según coste de negocio: 0.41
- Coste estimado con umbral óptimo: 84,800$
- Coste estimado con umbral por defecto (0.50): 120,000$
- Ahorro potencial: 35,200$ solo por elegir bien el umbral


## 📌 El problema de negocio

Un banco pierde dinero de dos formas al evaluar solicitudes de crédito:

| Error                     | Descripción                              | Costo asumido |
|---------------------------|------------------------------------------|---------------|
| **Falso Negativo (FN)**   |  Aprobar a un cliente que resulta moroso | **8,000 $**   |
| **Falso Positivo (FP)**   | Rechazar a un cliente que hubiera pagado | **400 $**     |    

Un FN cuesta **20 veces más** que un FP. Por eso el objetivo no fue maximizar Accuracy, sino **maximizar Recall** (capturar la mayor cantidad posible de morosos reales) y luego **calibrar el umbral de decisión al costo real**, no al 0.5 por defecto que usa la mayoría de tutoriales.


## 📊 Resultados

### Benchmark de modelos (5 algoritmos, con diagnóstico de overfitting)

| Modelo                | ROC-AUC   | Recall | Gap Overfitting (ROC-AUC)|
|-----------------------|-----------|--------|--------------------------|
| **Random Forest** ✅  | **75.2%** | 68.6%  | 10.1 pts                 |
| Logística (L2)        | 64.8%     | 68.6%  | 0.6 pts                  |
| LightGBM              | 80.7%     | 60.0%  | 15.1 pts                 |
| XGBoost               | 72.7%     | 54.3%  | 18.6 pts                 |
| Árbol de Decisión     | 65.1%     | 48.6%  | 3.2 pts                  |

**Random Forest** fue seleccionado como modelo final: mismo Recall que Logística, pero 10 puntos más de capacidad de discriminación (ROC-AUC), con un nivel de overfitting controlado y muy por debajo del de los modelos de *boosting* (que sobreajustaron más pese a regularización — un hallazgo interesante dado el tamaño reducido del dataset).

### Impacto económico del umbral óptimo

|                               | Umbral 0.50 (default) | Umbral 0.41 (óptimo) |
|-------------------------------|-----------------------|----------------------|
| Falsos Negativos              | 11                    | **2**                |
| Falsos Positivos              | 80                    | 172                  |
| **Costo estimado**            | 120,000 $             | **84,800 $**         |
| Recall (detección de morosos) | 68.6%                 | **94.3%**            |


## 🔍 Metodología

1. **Limpieza de datos** — detección y tratamiento de un valor sentinela crítico (`Employed_days = 365243`), que de no identificarse habría distorsionado todo el modelo.
2. **Prevención de fuga de información (*data leakage*)** — todos los estadísticos de imputación, outliers y escalado se calculan exclusivamente sobre el set de entrenamiento y se aplican después al test.
3. **Imputación** — mediana agrupada por variable categórica relevante (no mediana global).
4. **Tratamiento de outliers** — capping por rango intercuartílico (IQR / Winsorización), evaluado variable por variable (con criterio de negocio, no aplicado ciegamente).
5. **Multicolinealidad (VIF)** — análisis sobre variables numéricas antes del *one-hot encoding*.
6. **5 modelos comparados**: Regresión Logística, Árbol de Decisión, Random Forest, XGBoost, LightGBM — todos con `class_weight`/`scale_pos_weight` balanceado y tuneados con `GridSearchCV`.
7. **Diagnóstico de overfitting en cada etapa** (train vs. test), no solo al final — con corrección vía regularización e hiperparámetros de complejidad.
8. **Interpretabilidad**: coeficientes + odds ratios para el modelo lineal, `feature_importances_` + SHAP para el modelo de árbol — incluyendo una revisión crítica de variables con sesgo potencial (ej. género) y su implicación regulatoria en *credit scoring*.
9. **Optimización de umbral de decisión** según una función de costo de negocio explícita, no el 0.5 por defecto.
10. **Simulación de producción** — decisiones automáticas de Aprobado/Rechazado sobre solicitudes nuevas.


## 🛠️ Stack técnico

- Python
- Pandas
- Scikit-learn
- XGBoost
- LightGBM
- SHAP
- Statsmodels (VIF)
- Matplotlib
- Seaborn

## 📁 Estructura del repositorio

```text
.
├── data
│   ├── interim
│   ├── processed
│   └── raw
│       ├── Credit_card.csv
│       └── Credit_card_label.csv
├── environment.yml
├── images
│   └── umbral_decision.png
├── models
├── notebooks
│   └── riesgo_credito.ipynb
├── README.md
├── reports
│   └── figures
└── src

11 directories, 6 files
```
## ▶️ Cómo reproducirlo


Clona el repositorio:

```bash
git clone https://github.com/antonio-xrp/riesgo_credito.git
```

Ingresa al directorio del proyecto:

```bash
cd riesgo_credito
```

Crea el entorno de Conda a partir del archivo `environment.yml`:

```bash
conda env create -f environment.yml
```

Activa el entorno:

```bash
conda activate riesgo_credito
```
## 🧠 Aprendizajes destacados

- Un modelo con **mayor Accuracy no siempre es el mejor modelo de negocio** — el modelo ganador tiene la Accuracy más baja de los cinco (43.9% a umbral óptimo) mientras es el que menos costo genera.
- La **regularización afecta de forma muy distinta a bagging vs. boosting** en datasets pequeños: Random Forest generalizó mejor que XGBoost/LightGBM incluso después de aplicar controles de complejidad equivalentes.
- Detectar y corregir **fuga de información** (orden correcto de imputación/split) puede cambiar sustancialmente los resultados reportados — un error común incluso en proyectos publicados.


## 👨‍💻 Autor

**Antonio Palacios**

**Ingenierio Industrial | Analista de datos | Científico de datos**

- GitHub: [Antonio Palacios](https://github.com/antonio-xrp)
- LinkedIn: [Antonio Palacios](https://www.linkedin.com/in/antonio-palacios-orihuela-xrp/)
- correo: palaciosorihuelaantonio@gmail.com