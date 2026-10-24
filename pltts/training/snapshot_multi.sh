#!/bin/bash
# Zachowuje co 5. epoke treningu wielogłosowego.
#
# Lightning z --checkpoint-epochs 1 trzyma tylko najswiezszy plik. Przy obu
# poprzednich glosach najlepszy na ucho nie byl ostatni, wiec bez kopiowania na
# boku nie byloby czego porownywac.

CK=/workspace/piper-multi-out/lightning_logs/version_0/checkpoints
KEPT=/workspace/piper-multi-out/kept
LOG=/workspace/voices/eliza/snapshot_multi.log
EVERY=5

mkdir -p "$KEPT"
echo "[$(date -u +%H:%M)] snapshoter wystartowal, co $EVERY epok -> $KEPT" >>"$LOG"

while true; do
    for f in "$CK"/epoch=*.ckpt; do
        [ -e "$f" ] || continue
        base=$(basename "$f")
        ep=$(echo "$base" | sed -n 's/epoch=\([0-9]*\)-.*/\1/p')
        [ -z "$ep" ] && continue
        [ $((ep % EVERY)) -ne 0 ] && continue
        [ -e "$KEPT/$base" ] && continue
        # Lightning zapisuje strumieniowo; dwa odczyty rozne o 3 s musza byc rowne.
        a=$(stat -c %s "$f"); sleep 3; b=$(stat -c %s "$f")
        [ "$a" != "$b" ] && continue
        cp "$f" "$KEPT/$base" && echo "[$(date -u +%H:%M)] zachowano $base" >>"$LOG"
    done
    sleep 60
done
