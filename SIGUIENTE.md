# Dónde voy — leer esto primero

(Nota corta a mí mismo. Si estoy abrumado, esto es lo único que tengo que leer.)

## Qué funciona (de verdad, no es que "nada sirve")
- El clasificador GCN: 71% test acc. Funciona.
- Run 20 (retargeter): mimica cross-embodiment, fluido, robot-agnóstico, sin pairs. Funciona.

## Qué NO cierra del todo
- El nudillo (MCP) del robot no cierra a puño completo. Esa es LA pieza, no "todo".

## Lo que ya sé (medido, no opinión)
- El decoder está sano: si recibe el código de "robot cerrado", cierra perfecto.
- Las nubes latentes humano/robot no se tocan (gap).
- En el MCP: puño humano ~40°, puño robot ~135° (frame muñeca). Pero relativo-al-padre el gap baja a 74 vs 46. PISTA, no confirmada.

## EL ÚNICO siguiente paso (no abrir nada más)
Re-correr LIMPIO el test del MCP relativo-al-padre, con las poses de puño robot BUENAS (npz sintético, vía FK para sacar la cadena), no las filtradas.
- Pregunta única: ¿el MCP relativo-al-padre converge humano≈robot (como ya convergió el PIP)?
- El test anterior salió sucio: el control PIP falló (robot 24° en vez de ~90°) porque usé poses robot que no eran un puño real.

## Y AQUÍ TERMINA (las dos salidas, no hay tercera)
- Si converge → el fix es el FRAME del MCP. UNA cosa.
- Si NO converge → se cierra esa puerta. Se deja de tocar el modelo. Se escribe la tesis con lo que hay: clasificador + Run 20 + el diagnóstico de por qué el cierre no sale (eso ya es contribución).

## NO hacer
No volver a Run 20 vs 27. No mover similitud ni decoder ni pesos. Eso ya se intentó, los dos linajes, decenas de veces. No es ahí.

Para hablar con Claude mañana basta una frase: "corre el test del MCP limpio".
