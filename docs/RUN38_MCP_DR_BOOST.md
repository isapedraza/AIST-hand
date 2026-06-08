# Run 38 -- Boost del componente MCP de D_R sobre Run 20 limpio

Plan para continuar. NO corrido todavia. Branch: `run38-mcp-dr-boost` (creado desde `run20-original`).

## La idea en una frase
Subir SOLO el peso del MCP dentro del termino D_R (rotacion) del oracle de Run 20,
sin tocar la estructura per-finger (la que da la fluidez), para ver si el MCP cierra.

## Por que (mecanismo, verificado en los pesos reales de Run 20)
El oracle de Run 20 tiene dos terminos y NINGUNO ve bien la flexion del MCP:
- **D_joints** (DOMINANTE, w=1.2): mide POSICIONES de joints. El MCP es la base del
  dedo: su posicion casi no se mueve al flexionar el nudillo (lo que se mueve es la
  punta). Pesa alto (0.45-0.57) un punto casi fijo -> CIEGO a la flexion del MCP.
- **D_R** (w=1.0): mide ROTACION de segmentos -> SI ve la flexion del MCP. Pero sus
  pesos MCP son BAJOS: middle=0.188, pinky=0.197, ring=0.238, index=0.329, thumb=0.258.

Resultado: el unico termino que puede ver la flexion del MCP la sub-pesa; el dominante
es ciego a ella. Por eso Run 20 "quiere cerrar pero no llega" (el PIP cierra, el MCP no).
Subir el peso MCP de D_R le da ojos al MCP, manteniendo per-finger (fluidez).

## Por que esta SIN probar (celda vacia del cuadro)
Verificado contra los branches reales:
- `run21-paper-sk` = whole-hand (holistico) + Xin funcional = el caso "21sk"
  (pulgar perfecto, dedos desastre). NO es boost de MCP-D_R sobre Run 20.
  (MEMORY.md decia "Run 21 = MCP D_R weight 0.5 sobre abl11" -- MAL ATRIBUIDO.)
- Run 28 = `lam_dr=16.5` UNIFORME (MCP diluido 1/15) + Xin. No es MCP-especifico.
- Run 34 = oracle global, per-finger.
Ninguno hizo: **per-finger oracle Run 20 (D_R+D_joints+D_ahg, sin Xin) + boost MCP
especifico en D_R.** Esa celda esta vacia.

## El cambio exacto (tuneable, 1 flag, default = Run 20 byte-identico)
Archivo: `grasp-model/src/cross_emb/train_cross_emb.py` (branch run38).
1. Agregar flag en `_parse_args`:
   `--mcp_dr_scale` (float, default 1.0). 1.0 = identico a Run 20.
2. Donde se construye `sk_weights_dr` (dict `_sk_w`, ~linea 275), multiplicar la
   PRIMERA columna (MCP) de cada dedo por `args.mcp_dr_scale` ANTES de hacer el
   tensor. El loop ya renormaliza por-par (`_w_dr / _w_dr.sum()`), asi que el scale
   sube la cuota relativa del MCP.

`_sk_w` actual (orden [MCP, PIP, DIP, TIP]):
```
thumb:  [0.258, 0.544, 0.199, 0.0]
index:  [0.329, 0.325, 0.346, 0.0]
middle: [0.188, 0.362, 0.451, 0.0]   <- MCP bajisimo
ring:   [0.238, 0.357, 0.405, 0.0]
pinky:  [0.197, 0.405, 0.398, 0.0]
```

## Config del run (notebook, branch run20-original style)
- BRANCH = run38-mcp-dr-boost
- Oracle Run 20: w_r=1.0, w_joints=1.2, w_ahg=0.07 (NO Xin, NO anchor)
- z_dim=16, shared_dim=1024, margin=0.1, seed=21266, B=50k, n_steps=15k
- NUEVO: MCP_DR_SCALE = 3.0 (empezar; tunear 2/3/5)
- Flag en cmd: `--mcp_dr_scale 3.0`
- Subir nada extra a Drive (usa lo de Run 20; no necesita synthetic npz).

## Salidas (binario, sirve cualquiera)
- Cierra + sigue fluido -> ROMPE la teoria "per-finger no cierra es estructural".
  Ganaste cierre + fluidez. Tunear scale al minimo que cierre.
- No cierra -> CONFIRMA fuerte que es estructural (per-finger no puede). Cierra esa
  puerta con evidencia y se va a hibrido o a escribir la tesis con el diagnostico.

## Caveats honestos
- La memoria two_families dice "per-finger NUNCA cierra, es estructural" (20, 28-34).
  PERO ninguno de esos boosteo MCP-D_R especifico -> la conclusion se saco SIN esta
  perilla. Por eso vale probarla: es la unica celda no probada.
- Riesgo de over-boost (visto en variantes): subir mucho el MCP puede ROMPER la
  extension (dedos quedan doblados, no estiran). Por eso es tuneable: probar scale
  chico primero (2), subir si no cierra.

## Estado de lo anterior (para no repetir)
- anchor (Run 37) = FALLO. Alineo las nubes (dist 4.26->0.04) y el MCP NO cerro;
  ademas DEGRADO el decoder (robot-AE cerrado 1.5->0.6). Desmiente que "alinear nubes
  arregla el MCP". No volver a anchor. Ver memoria project_run37_anchor_run20base.
- Run 35/36 (hibrido/residual) = malos segun el usuario.
- Run 20 sigue siendo el mejor en fluidez; no cierra MCP.

## Para retomar
Checkout `run38-mcp-dr-boost`, aplicar el cambio del flag (seccion "El cambio exacto"),
configurar notebook, correr en Colab T4, evaluar con live_retarget + diag_run37_human.py
(mirar si MCP de salida sube de ~0.6).
