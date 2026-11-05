# Respaldo en la nube — mapas de lluvia y viento (Guerrero a Chiapas)

Copia de `actualizar_mapas.py` que corre en GitHub Actions cuando la PC de la
oficina no puede publicar (apagada o sin internet).

- Cada 30 min revisa si la PC publicó la página en los últimos 70 min.
- Si la PC está activa, termina sin hacer nada.
- Si no, revisa NHC y modelo, genera los mapas cuando hay cambios, los manda por
  ntfy y publica la página `monitor-mapas-sc` (commits marcados con `[nube]`).
- Cuando la PC vuelve a publicar, la nube regresa sola a espera.

Datos privados (token y tema de ntfy) van en **Secrets**, nunca en el código.
Solo usa datos públicos: NHC (NOAA) y Open-Meteo.
