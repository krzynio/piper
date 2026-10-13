# pl_multi — jeden model, trzy polskie głosy

Piper VITS wielogłosowy, epoka 15. Jeden plik ONNX obsługuje trzy głosy; wybiera
się je czwartym wejściem `sid`.

| sid | głos | charakter | f0 |
|---|---|---|---|
| 0 | `glos3` | kobiecy | 209 Hz |
| 1 | `design` | głęboki męski | 115 Hz |
| 2 | `lektor` | jasny męski | 146 Hz |

```sh
pltts "Dzień dobry, z tej strony biuro obsługi klienta." -v all
# zapisuje probe_glos3.wav, probe_design.wav, probe_lektor.wav
```

## Co tu jest

| plik | rola |
|---|---|
| `src/pltts/models/pl_multi.onnx.json` | konfiguracja modelu, `num_speakers: 3`, `speaker_id_map` |
| `src/pltts/frontend.py` | espeak-pl → poprawka trz/drz → akronimy, liczby → angielskie słowa |
| `src/pltts/pl_trz_fix.py`, `pl_en_fix.py` | poprawki fonemowe; kopie plików z `src/python/piper_train/`, muszą zostać identyczne |
| `src/pltts/lid/` | detektor języka per słowo, 3,6 MB |
| `training/build_multi.py` | scala trzy korpusy jednogłosowe w `ścieżka\|mówca\|tekst` |
| `training/snapshot_multi.sh` | kopiuje co 5. epokę na bok, bo Lightning trzyma tylko ostatnią |
| `training/pl_multi-hparams.yaml` | hiperparametry treningu, zrzut z `lightning_logs` |

## Czego nie ma w repo

Za duże na GitHuba (limit 100 MB na plik), leżą na dysku:

| plik | gdzie |
|---|---|
| `pl_multi.onnx` (128 MB) | `/workspace/multi-release/` i `/workspace/pltts/src/pltts/models/` |
| `pl_multi-e15.ckpt` (1,08 GB) | `/workspace/multi-release/` |
| snapshoty e0, e5, e10, e15 | `/workspace/piper-multi-out/kept/` |
| wheel `pltts-0.1.1` z modelem w środku | `/workspace/pltts/dist/` |
| nagrania wzorcowe trzech głosów | `/workspace/multi-release/refs/` |

Instalacja i budowanie wheela z repo: patrz [README](README.md#instalacja).

## Odtworzenie treningu

Poprawka trz/drz działa w `piper_train.preprocess` dla każdego `--language pl*`,
więc wystarczy zwykły preprocessing z tego brancha.

```sh
python pltts/training/build_multi.py        # -> /workspace/piper-multi/metadata.csv

python -m piper_train.preprocess \
  --language pl --input-dir /workspace/piper-multi --output-dir /workspace/piper-multi-out \
  --dataset-format ljspeech --sample-rate 24000 --max-workers 16

python -m piper_train \
  --dataset-dir /workspace/piper-multi-out \
  --accelerator gpu --devices 1 --precision 32 --quality high \
  --batch-size 24 --validation-split 0.01 --num-test-examples 6 \
  --max_epochs 16 --checkpoint-epochs 1 \
  --resume_from_single_speaker_checkpoint /workspace/piper-design-out/kept/epoch=15-step=107744.ckpt
```

Komenda preprocessingu jest dokładnie tą, którą odpalono. Komenda treningu jest
odtworzona z `training/pl_multi-hparams.yaml`.

## Dlaczego jeden model zamiast trzech

Wektor mówcy (`gin_channels=512`) wchodzi tylko do modułów akustycznych —
`enc_q`, `flow`, `dp`, `dec`. **Enkoder tekstu jest wspólny**, więc fonetyka, w tym
poprawka `trz/drz`, uczy się na sumie trzech korpusów i obsługuje wszystkie głosy.

Koszt wielogłosowości jest jednorazowy: +6,7 mln parametrów na warstwy `cond`.
**Każdy kolejny głos to 512 parametrów, czyli 2 kB** — dziesięć głosów zmieściłoby
się w tym samym pliku.

```
trzy osobne modele:  342 MB
ten model:           128 MB      -63%
```

## Jak powstał

Korpusy scalone do 244 537 wypowiedzi (`glos3` 83 610, `design` 81 620,
`lektor` 79 307), format `ścieżka|mówca|tekst`. Ciepły start z `design-e15` przez
`--resume_from_single_speaker_checkpoint`. Trening 04:51–14:41, 9 h 50 min,
16 epok (zatrzymał się sam), batch 24, precision 32, quality high.

Preprocessing zajął **90 sekund zamiast 3,3 godziny**: cache Pipera jest kluczowany
`sha256(bezwzględna ścieżka audio)`, więc scalone metadane wskazują oryginalne
ścieżki z trzech poprzednich przebiegów, a trzy cache'e zostały połączone twardymi
dowiązaniami — 489 074 plików, 230 GB logicznie, zero skopiowanych bajtów.

## Znane ograniczenie: lektor ma zaniżone f0

Mediana z siedmiu zdań:

| głos | epoka 10 | epoka 15 | wzorzec |
|---|---|---|---|
| glos3 | 202,6 | **209,3** | 206,4 |
| design | **118,5** | 114,5 | 121,6 |
| lektor | **152,2** | 146,4 | **172,5** |

`lektor` schodzi 26 Hz poniżej wzorca i pogłębia się to z treningiem. Hipoteza:
jego f0 leży pomiędzy dwoma pozostałymi (172 przy 206 i 122), więc jest najbardziej
narażony na ściąganie ku średniej — głosy skrajne mają się od czego odróżnić,
środkowy nie. Nie zweryfikowane.

Każdy głos ma inne optimum: `glos3` na e15, `design` i `lektor` na e10. Przy jednym
wspólnym modelu trzeba wybrać jeden snapshot; wybrano 15 po odsłuchu.
Snapshoty 0, 5, 10, 15 leżą w `/workspace/piper-multi-out/kept/`.

Jeśli barwa `lektora` okaże się problemem w produkcji, jest gotowy model
jednogłosowy `/workspace/lektor-release` (e15), gdzie f0 się zgadza.

## Czego brakuje

- WER nie mierzony na tym modelu — nie wiadomo, czy dzielenie enkodera poprawiło
  zrozumiałość względem modeli jednogłosowych, czy tylko jej nie zepsuło
- port detektora języka do JS dla ONNX WebAssembly (~150 linii)
- sprawdzenie, czy build espeak-ng w WASM ma dane `en-us`, nie tylko `pl`
