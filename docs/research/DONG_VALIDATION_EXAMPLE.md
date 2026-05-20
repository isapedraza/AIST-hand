# Dong Validation Example

Referencia local usada:
- `/home/yareeez/Downloads/applsci-15-08921.xml`

Paper:
- Dong, Y.; Payandeh, S. "Hand Kinematic Model Construction Based on Tracking Landmarks". *Applied Sciences* 2025, 15, 8921.

## Objetivo

Usar el ejemplo numérico del paper como prueba de validación para cualquier implementación del modelo cinemático de `Dong`.

## Landmarks 3D del ejemplo

Valores de `Table 2`, en metros:

- `d0 = (-0.21, -0.11, 0.54)` wrist
- `d5 = (-0.13, -0.16, 0.56)` index MCP
- `d6 = (-0.10, -0.16, 0.56)` index PIP
- `d7 = (-0.08, -0.16, 0.55)` index DIP
- `d8 = (-0.06, -0.16, 0.55)` index tip
- `d9 = (-0.13, -0.14, 0.57)` middle MCP
- `d13 = (-0.13, -0.12, 0.57)` ring MCP
- `d17 = (-0.14, -0.10, 0.57)` little MCP

## Frame de palma `{0}`

Según las Ecs. (5)-(7):

- `X0 = (0.9171, -0.1299, 0.3768)`
- `Y0 = (0.0427, 0.9720, 0.2310)`
- `Z0 = (-0.3963, -0.1957, 0.8970)`

## Punto del índice en frame de palma

Según la Ec. (16):

- `d5^0 = (0.0892, -0.0327, 0)`

## Frame local del índice `{5}`

Según las Ecs. (20)-(22):

- `X5^0 = (0.9006, -0.1439, -0.4102)`
- `Y5^0 = (0.1577, 0.9875, 0)`
- `Z5^0 = (0.4050, -0.0647, 0.9120)`

## Ángulos esperados

Según las Ecs. (24), (38) y (39):

- `β5 = arccos(0.9120) = 24.22°`
- `γ5 = arccos(0.9875) = 9.07°`
- `β6 = 22.22°`
- `β7 = 8.77°`

## Datos intermedios para `β6` y `β7`

Expresados en el frame `{5}`:

- `d6^5 = (0.0333, 0, 0)`
- `d7^5 = (0.0520, 0.0016, -0.0077)`
- `d8^5 = (0.0679, 0.0037, -0.0114)`

Vectores usados:

- `d7^5 - d6^5 = (0.0187, 0, -0.0077)`
- `d8^5 - d7^5 = (0.0158, 0, -0.0038)`

## Criterio de validación

Una implementación fiel a `Dong` debería reproducir aproximadamente:

- `β5 = 24.22°`
- `γ5 = 9.07°`
- `β6 = 22.22°`
- `β7 = 8.77°`

Si no salen estos valores con los landmarks del ejemplo:

- hay un error en la construcción del frame de palma,
- o en la transformación al frame local,
- o en la extracción de `β` / `γ`,
- o en la formulación de `β6` / `β7`.
