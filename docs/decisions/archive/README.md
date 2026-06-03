# DECISIONS.md -- archivo inmutable de variantes

DECISIONS.md divergio entre branches (git versiona por branch, no hay archivo
compartido). Este directorio preserva TODAS las variantes distintas como
snapshots inmutables, para no perder ninguna entry. NO se ha fusionado ni
elegido una canonica todavia.

Cada archivo es una copia byte-perfect del blob original (verificado con
`git hash-object`). El nombre incluye el hash corto del blob y las branches que
lo comparten al momento de archivar (2026-06-02).

| archivo | blob | branches | ubicacion original |
|---------|------|----------|--------------------|
| DECISIONS_90488204_run20-run24_root.md | 90488204 | run20-original, run24_weightless | `DECISIONS.md` (raiz) |
| DECISIONS_cb77df05_run21_root.md       | cb77df05 | run21-paper-sk | `DECISIONS.md` (raiz) |
| DECISIONS_ec22687f_main-run30.md       | ec22687f | main, run30-pinch-switching | `docs/decisions/DECISIONS.md` |
| DECISIONS_bbf40517_run27b.md           | bbf40517 | run27b-dr-yan | `docs/decisions/DECISIONS.md` |
| DECISIONS_eb1cd5c1_run28.md            | eb1cd5c1 | run28-per-finger-dr-yan | `docs/decisions/DECISIONS.md` |
| DECISIONS_91a14530_run29-31-33-34.md   | 91a14530 | run29-thumb-pos, run31-fingerpose-dense, run33-local-global-contrastive, run34-global-oracle | `docs/decisions/DECISIONS.md` |

## Pendiente (a futuro)
1. Designar la DECISIONS "actual" (la mas nueva / linea de runs vigente).
2. Reconciliar / seleccionar las entries correctas en una sola DECISIONS canonica.
3. Mover esa canonica a `main` como single source; branches de codigo dejan de
   editar docs (sincronizar con `git checkout main -- docs/`).

Por ahora: solo preservar. No tocar el contenido de estos archivos.
