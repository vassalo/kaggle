## Modelos
- Modelo simples

Arquitetura: `Flatten(28x28x1) -> Dense(512, relu) -> Dropout(0.2) -> Dense(10, softmax)`

- Modelo complexo 1

Arquitetura: `[[Conv2D(relu)]*2 -> MaxPool2D(2x2) -> Dropout(0.25)]*2 -> Flatten() -> Dense(256, relu) -> Dropout(0.5) -> Dense(10, softmax)`

- Modelo complexo 2

Arquitetura: `[[Conv2D(relu)]*2 -> MaxPool2D(2x2) -> Dropout(0.25)]*2 -> Flatten() -> Dense(512, relu) -> Dropout(0.2) -> Dense(10, softmax)`


## Resultados
### #1 Modelo simples
- **Optimizer:** adam
- **Loss:** sparse_categorical_crossentropy
- **Acurácia:** 0.97457

### #2 Gerador de imagens + modelo simples
- **Acurácia:** 0.97257

### #3 Gerador de imagens + modelo complexo 1
- **Acurácia:** 0.98971

### #4 Gerador de imagens + modelo complexo 2
- **Acurácia:** 0.99271
- **Perda (local)**: 0.0526
