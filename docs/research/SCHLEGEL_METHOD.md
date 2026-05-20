# Schlegel Method

Basado unicamente en `Schlegel et al. (2024)`, secciones `3.1` y `3.2`.

Este documento resume solo lo que el paper dice sobre:

- como define un `Joint Coordinate System` local (`JCS`)
- como calcula `flexion`, `abduction` y `axial rotation`

No adapta el metodo a la mano. No introduce proxies. No traduce nada a `MediaPipe`.

## 1. Objetivo del metodo

Schlegel coloca un `JCS` local en cada articulacion para describir la posicion del segmento movil de forma:

- independiente del punto de vista global
- independiente de la orientacion global del cuerpo
- independiente de la pose en otras articulaciones

El paper dice que, para poder determinar la posicion del hueso movil en el espacio, la orientacion del `JCS` debe depender del segmento fijo y potencialmente del movimiento aguas arriba, pero debe ser independiente del segmento movil.

## 2. Ejes del JCS

Schlegel indica que, en general:

- `x` apunta hacia adelante
- `y` apunta hacia arriba
- `z` apunta lateralmente, alejandose de la linea media del cuerpo

Para joints localizados sobre la linea media, el convenio ISB hace que `z` apunte hacia el lado derecho.

## 3. Construccion general del JCS

El procedimiento general descrito por Schlegel es:

1. Convertir el conjunto de keypoints en una estructura osea.
2. En cada articulacion identificar el unico hueso que articula el movimiento del joint.
3. Definir el `JCS` usando dos huesos cercanos al joint que no participen en ese movimiento.

El paper establece estas reglas:

- Uno de los ejes del `JCS` siempre viene dado por uno de los huesos estaticos.
- Un segundo eje se define como el producto cruz entre ese primer eje y el segundo hueso estatico.
- El tercer eje se define como el producto cruz entre los dos primeros ejes.

Schlegel tambien aclara que el orden de asignacion a `x`, `y` y `z` cambia segun el joint.

### Implementacion del JCS

Bajado a pasos operativos, la construccion del `JCS` segun Schlegel es:

1. Convertir los keypoints en una estructura osea.
2. Elegir una articulacion concreta.
3. Identificar el unico hueso que articula el movimiento de esa articulacion.
4. Elegir dos huesos cercanos a la articulacion que no participen en ese movimiento.
5. Tomar uno de esos huesos estaticos como eje base del `JCS`.
6. Calcular un segundo eje como el producto cruz entre:
   - el eje base
   - el segundo hueso estatico
7. Calcular el tercer eje como el producto cruz entre los dos primeros ejes.
8. Asignar esos tres ejes a `x`, `y`, `z` segun la articulacion concreta.

Forma abstracta:

```text
axis_1 = one_static_bone
axis_2 = cross(axis_1, second_static_bone)
axis_3 = cross(axis_1, axis_2)
```

Schlegel no da una unica asignacion universal de `axis_1`, `axis_2`, `axis_3` a `x`, `y`, `z`; esa asignacion depende del joint.

## 4. Excepcion: codo y rodilla

Schlegel dice que codo y rodilla son la excepcion a la regla anterior, porque ahi no hay tres huesos asociados que permitan definir un hueso movil y dos huesos estaticos.

En esos joints:

- si usa el hueso movil para definir el `JCS`
- especificamente usa el segmento movil para definir el eje `z`

La justificacion que da es que esto no rompe la independencia del `JCS` respecto al movimiento, porque codo y rodilla solo se mueven en el plano sagital (`xy`).

## 5. Definiciones concretas de joints en Schlegel

En las secciones leidas, Schlegel explicita:

- `hip`: `z` es el vector desde la cadera opuesta hacia la cadera del joint; `x` es el producto cruz entre ese `z` y la parte inferior de la columna
- `shoulder`: definido de forma similar, usando la cintura escapular y la parte superior de la columna
- `elbow`: `y` es el hueso del brazo superior apuntando del codo al hombro; `z` es el producto cruz entre brazo y antebrazo
- `knee`: analogo al codo usando muslo y pierna
- `wrist`: `y` es el antebrazo; `x` es el producto cruz del antebrazo con el pulgar
- `ankle`: `y` es la pierna inferior; `z` es el producto cruz con el vector desde la base del pie hacia los dedos
- `neck`: `y` es la parte superior de la columna; `x` es el producto cruz del vector de una oreja a la otra

Schlegel ademas dice explicitamente que en la muneca usar el pulgar para definir el eje es una aproximacion imprecisa, pero la mejor posible con keypoint sets comunes.

## 6. Que angulos describe

Schlegel describe tres tipos de movimiento:

- `flexion`
- `abduction`
- `axial rotation`

El paper dice que no todos los joints articulan movimiento en los tres planos, pero que en cada caso el conjunto de hasta tres angulos define exactamente el estado del joint.

## 7. Joints de 1 DOF

Para joints que no pueden moverse en el plano frontal, Schlegel dice:

- la `flexion` se determina simplemente como el angulo entre el hueso movil y el hueso proximal del joint
- ese hueso proximal coincide con el eje `y` del `JCS`

Ejemplos que da el paper:

- `elbows`
- `knees`

## 8. Joints de 2 DOF: hombro y cadera

Para hombro y cadera, Schlegel define `flexion` y `abduction` de forma similar a coordenadas esfericas:

- el eje `z` actua como direccion cenit
- `abduction` es analoga al angulo polar `phi`
- `flexion` es analoga al angulo acimutal `theta`

Pero el paper especifica dos definiciones concretas:

- `abduction` no es el angulo entre el hueso y el eje `z`
- `abduction` es el angulo entre el hueso y su proyeccion ortogonal sobre el plano sagital (`xy`)
- `flexion` es el angulo entre esa proyeccion ortogonal y el eje `y`

El paper tambien afirma:

- con esta definicion, la flexion pura es rotacion alrededor de `z`
- la abduccion pura es rotacion alrededor de `x`

Y para movimientos mezclados:

- el orden del movimiento importa
- la definicion adoptada considera a la flexion como el movimiento dominante
- con ello se obtiene una representacion continua casi en todas partes

Schlegel senala una singularidad en:

- `90` grados de abduccion
- `0` grados de flexion

cuando el hueso movil queda alineado con el eje `z` del `JCS`.

## 9. Rotacion axial

Schlegel dice que la `axial rotation` se manifiesta aguas abajo en la cadena cinematica porque es una rotacion alrededor del eje longitudinal del segmento movil.

Para medirla:

1. se aplican la `flexion` y `abduction` al `JCS` proximal
2. se alinea su eje `y` con el `JCS` aguas abajo
3. la rotacion restante entre ambos sistemas es la `axial rotation`

El paper lo relaciona con el uso de dos `JCS` por joint en los estandares ISB.

Tambien precisa que, por esta razon:

- la rotacion axial en codo y rodilla solo puede determinarse si existen keypoints mas alla de muneca y tobillo que permitan orientar el `JCS` distal correspondiente

## 10. Muneca y cuello

Schlegel distingue a `wrist` y `neck` de hombro y cadera.

Para estos joints:

- el sistema esferico se formula usando `x` como direccion cenit
- `abduction` o `lateral flexion` es el angulo entre el hueso y su proyeccion ortogonal sobre el plano frontal (`yz`)
- `flexion` es el angulo entre esa proyeccion y el eje `y`

## 11. Respuesta puntual

No. En el paper, Schlegel no calcula solo `flexion`.

Segun el tipo de articulacion, su metodo calcula:

- `flexion` en joints de 1 DOF
- `flexion + abduction` en joints que articulan plano sagital y frontal
- `axial rotation` cuando existe informacion suficiente aguas abajo para comparar sistemas locales

Si solo se quiere `flexion`, el metodo de Schlegel permite calcular unicamente esa componente: en joints de 1 DOF se usa directamente el angulo entre hueso movil y hueso proximal, y en joints de 2 DOF se toma solo la definicion de flexion basada en la proyeccion del hueso movil sobre el plano de referencia del `JCS`.
