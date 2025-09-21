# smv

Implementación sencilla de un SVM lineal multiclase entrenado con SGD.

## Parámetros principales

- `lr`: tasa de aprendizaje del optimizador.
- `n_iter`: número de épocas completas sobre los datos.
- `C`: peso del término hinge de la pérdida.
- `avg`: activa el promedio de pesos (Averaged SGD).
- `l2`: factor de regularización L2 aplicado como weight decay (`1.0` por defecto). Valores menores reducen la contracción de los pesos, lo que puede ser útil si se necesitan coeficientes más grandes.

## Uso rápido

```python
from ml_svm import SVM

model = SVM(lr=0.001, n_iter=200, C=1.0, avg=True, l2=0.5)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

Ajusta `l2` para equilibrar la regularización: un valor más bajo mantiene pesos de mayor magnitud, mientras que valores altos fomentan coeficientes pequeños.
