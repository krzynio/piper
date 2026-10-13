# pltts — polski TTS z poprawionym frontendem fonemowym

Piper (VITS) wielogłosowy plus warstwa, która naprawia to, czego eSpeak-pl nie
robi dobrze. Jeden `pip install`, model w środku, bez pobierania czegokolwiek
przy pierwszym uruchomieniu.

## Instalacja

Model `pl_multi.onnx` (128 MB) jest za duży na gita, więc przed instalacją trzeba
go wgrać do paczki. Bez niego `pip install` kończy się błędem. To celowe, żeby nie
powstała paczka, która nie umie mówić.

```sh
git clone git@github.com:krzynio/piper.git
cd piper/pltts
cp /workspace/multi-release/pl_multi.onnx src/pltts/models/   # na innej maszynie: scp z serwera
python -m venv .venv && . .venv/bin/activate
pip install .
```

Wymaga Pythona ≥ 3.9. espeak-ng nie trzeba instalować osobno, bo jego dane
przychodzą razem z `piper-phonemize`.

Żeby przenieść paczkę na inną maszynę jednym plikiem, zbuduj wheel. Ma w środku
model i działa bez sieci:

```sh
pip wheel . -w dist --no-deps          # -> dist/pltts-0.1.1-py3-none-any.whl (122 MB)
pip install dist/pltts-0.1.1-py3-none-any.whl
```

## Użycie z CLI

```sh
pltts --list-voices                                   # 0 glos3, 1 design, 2 lektor
pltts "Trzy drzewa przy ulicy Długiej." -v lektor -o out.wav
pltts "Dzień dobry." -v all                           # probe_glos3.wav, probe_design.wav, probe_lektor.wav
echo "Tekst ze standardowego wejścia." | pltts -v design -o out.wav
pltts "Trochę wolniej." -s 1.3 -o wolno.wav          # -s > 1.0 zwalnia
pltts --phonemes "Trzy drzewa."                       # sama transkrypcja, bez syntezy
```

Bez `-v` mówi pierwszy głos (`glos3`), a bez `-o` zapisuje do `probe_<głos>.wav`.
`-o -` wypisuje na stdout surowe próbki float32 mono 24 kHz, nie WAV.

## Użycie z Pythona

```python
import pltts

pltts.voices()                                  # {'glos3': 0, 'design': 1, 'lektor': 2}

audio, rate = pltts.synth("Pociąg PKP odjeżdża z peronu 3.", "glos3")
pltts.save("out.wav", audio, rate)              # WAV 16 bit, 24 kHz

# głos można podać też jako id; length_scale > 1.0 zwalnia
audio, rate = pltts.synth("Trzydzieści trzy złote.", 2, length_scale=1.2)

pltts.phonemize("Trzy drzewa.")                 # ['t', 'ʂ', 'ˈ', 'ɨ', ' ', 'd', 'ʐ', ...]
```

`synth` zwraca `numpy.float32` w zakresie [-1, 1] i częstotliwość próbkowania.
Model wczytuje się przy pierwszym wywołaniu i zostaje w pamięci, więc kolejne
zdania idą szybko. Pełna lista funkcji jest niżej, w sekcji [API](#api).

## Głosy

| sid | nazwa | charakter | f0 |
|---|---|---|---|
| 0 | `glos3` | kobiecy | 209 Hz |
| 1 | `design` | głęboki męski | 115 Hz |
| 2 | `lektor` | jasny męski | 146 Hz |

`pltts --list-voices`, a `-v all` renderuje wszystkie trzy.

## Co ten frontend naprawia

Wszystko poniżej dzieje się **przed** syntezą, więc dotyczy każdego głosu naraz —
wektor mówcy warunkuje tylko moduły akustyczne i nigdy nie dochodzi do enkodera
tekstu.

| problem | eSpeak sam | z pltts |
|---|---|---|
| `trzy drzewa` | `tʃˈɨ dʒˈɛva` — to samo co „czy dżewa” | `tʂˈɨ dʐˈɛva` |
| `AGD` | `ˌaɡjˌɛdˈɛ` — jedna zbitka, zero granic wyrazów | `a ɡʲˈɛ dˈɛ` |
| `ZUS`, `BLIK` | `zˌɛtˌuˈɛs` — literowane, choć to słowa | `zˈus`, `blˈik` |
| `XIX wieku` | `rˌɔwmɛnnˈajntin` — angielskie „roman nineteen” | `dʑˌɛvʲɛtnastˈɛɡɔ` |
| `W XX w.` | `w ˌiksˈiks w` | `w dvudʑˈɛstɨm vʲˈɛku` |
| `Trump`, `weekend` | `trˈump`, `vˈɛɛkɛnt` | `trˈamp`, `wˈikɛnt` |
| `dział wsparcia` | `fsparsˈija` — błąd eSpeaka | `fspˈartɕa` |
| `Naciśnij 3` | `tʃˈɨ` — „czy” | `tʂˈɨ` |
| `603` | `ʃˈɛɕsɛt` — gubi `tɕ` | `ʃˈɛɕtɕsɛt` |

### trz / drz

eSpeak zwija `trz` do `cz`, a `drz` do `dż`, więc „trzy drzewa” i „czy dżewa”
dostają identyczne fonemy i model nie ma jak się ich nauczyć rozróżniać. Para
retrofleksów `ʂ` / `ʐ` została przejęta na `trz` / `drz` i **model był trenowany
na tej konwencji** — dlatego frontend i checkpoint muszą chodzić razem.

### Akronimy

Rozwijane na nazwy liter, bo eSpeak zwraca je jako jedną zbitkę bez ani jednego
fonemu spacji i model nie ma czego rozdzielić — słychać wtedy zlane „agiede”
zamiast „a gie de”. W niektórych zdaniach eSpeak czyta `AGD` wręcz jako „akt”.

Które akronimy czyta się jak słowa (`ZUS`, `BLIK`, `NATO`, `PESEL`), a które
literuje (`PKP`, `AGD`, `USB`), rozstrzyga **reguła fonotaktyczna**, nie lista:
akronim czyta się jak słowo dokładnie wtedy, gdy jego litery składają się w
wymawialne polskie słowo. Sprawdzane sylabifikacją — dopuszczalna nagłosowa
zbitka, dopuszczalny wygłos, brak zbitek trzech spółgłosek w środku. Na 52
prawdziwych polskich akronimach reguła zgadza się z wymową w 52 przypadkach.
Dwa wyjątki są wpisane wprost: `USA` sylabifikuje się („u-sa”), a jednak się je
literuje, i `AIDS`, które ma niedozwolony wygłos, a czyta się jak słowo.

### Liczby i błąd eSpeaka na `-rcia`

Liczby całkowite są rozwijane na słowa **w mianowniku**, bo własna ścieżka cyfrowa
eSpeaka jest gorsza od jego ścieżki literowej dla tych samych słów: skleja wyrazy
bez granic (`183` → `/stˈɔɔɕɛmdʑˈɛɕɔnttʃˈɨ/`), gubi fonemy (`603` → `/ʃˈɛɕsɛt/`,
bez `tɕ` w „sześćset”), wybiera niestandardowe warianty (`/ɕcʲi/` w „trzydzieści”)
i zwija `trzy` do `czy` tam, gdzie poprawka trz nie ma czego zobaczyć, bo w tekście
stoi cyfra, a nie litery. Przypadek się nie zmienia — eSpeak też czyta cyfry
w mianowniku — więc zgodność z resztą zdania nadal należy do normalizatora tekstu.
Liczebnik sprawdzony na 16 001 wartościach względem `num2words`, zero rozbieżności.

Ułamki i daty zapisane cyframi (`249,99`, `12.03.2024`) są **zostawione**
nietknięte, bo wymagają zgodności przypadków.

Osobno: eSpeak-pl myli każde słowo kończące się na `-rcia` — „wsparcia” wraca jako
`/fsparsˈija/`, jakby zadziałała reguła od łacińskiego „-tia”. Klasa jest częsta
(wsparcia, otwarcia, starcia, poparcia), a `-rciach` i `-rciu` są poprawne, więc
naprawa jest wąska: respelling samego wygłosu na `rća`, co daje fonemy, które
eSpeak wszędzie indziej produkuje prawidłowo.

### Liczby rzymskie

Zostawione samym sobie idą do eSpeaka i wracają **po angielsku**. Zamieniane na
polski liczebnik porządkowy, w przypadku narzuconym przez sąsiedztwo:

```
XIX wiek       -> dziewiętnasty wiek
z XIX wieku    -> z dziewiętnastego wieku
w XIX wieku    -> w dziewiętnastym wieku
W XX w.        -> w dwudziestym wieku
Rozdział IV    -> Rozdział czwarty
płyta CD       -> płyta ce de
```

`CD` też jest poprawną liczbą rzymską (400), więc sam kanoniczny zapis nie
wystarcza — czytanie liczbowe wymaga kontekstu, który o nie prosi („wiek”,
„rozdział”, „część”, imię władcy). Bez kontekstu token jest literowany, nigdy
oddawany eSpeakowi.

### Angielskie słowa

Wykrywane per słowo, bez listy wyrazów: filtry Blooma na formy polskie i
angielskie plus marginesy n-gramów znakowych, razem 3,6 MB. Wymowa jest potem
składana **wyłącznie z 48 fonemów, które model widział na treningu** — jeśli
fonem angielski nie ma polskiego odpowiednika, słowo zostaje przy polskiej
wymowie, zamiast trafić do modelu jako coś, czego nigdy nie słyszał. Polska
końcówka fleksyjna jest doklejana z powrotem, więc `Netflixie` czy `Trumpem`
odmieniają się normalnie.

## Wyjątki: nazwy własne i akronimy mówione jednym ciągiem

Trzy rzeczy, których żadna reguła nie zgadnie, więc są jawnymi wpisami.

**Nazwy własne.** Detektor angielskiego nie ma jak wiedzieć, że „Romet” to polska
nazwa handlowa — nie jest polskim słowem i wygląda obco, więc czyta ją `/rˈɔmit/`,
„romit”. Sam eSpeak-pl radzi sobie z nią poprawnie, trzeba go tylko dopuścić:

```python
pltts.keep_polish("Romet", "Budimex")
```

**Akronimy mówione jednym ciągiem.** `SMS` to nie „es · em · es” z pauzami, a
„esemes”. Literowanie jest właściwe dla `AGD`, ale w tych wstawia słyszalne przerwy:

```python
pltts.pronounce_as(SMS="esemes", MMS="ememes")   # domyślnie już wpisane
```

**Kiedy ortografia nie wystarcza.** Polski akcent pada na przedostatnią sylabę, więc
„ceerem” przeczyta się „ce-**E**-rem”, a literowany akronim ma akcent na ostatniej
nazwie litery — „ce-er-**em**”. Tego nie da się zapisać literami, więc fonemy
przypina się wprost; są wstawiane po eSpeaku, tak samo jak angielskie słowa, i mają
priorytet nad detektorem:

```python
pltts.pronounce_phonemes(CRM="tsɛɛrˈɛm")         # domyślnie już wpisane
```

Respellingi rejestrują się automatycznie jako polskie — inaczej detektor zabrałby
je sobie i „ceerem” wróciłoby jako `/sˈirɛm/`.

## API

```python
pltts.synth(text, voice=0, *, length_scale=1.0, noise_scale=0.667,
            noise_w=0.8, english=True, acronyms=True, threads=0)
pltts.save(path, audio, rate)
pltts.voices()                     # {'glos3': 0, 'design': 1, 'lektor': 2}
pltts.phonemize(text)              # lista fonemów
pltts.phoneme_ids(text)            # (id-ki wejściowe, string fonemów)
pltts.expand_acronyms(text)        # sam krok tekstowy
pltts.spell_numbers(text)          # tylko rozwinięcie liczb
pltts.reads_as_word("ZUS")         # True
pltts.keep_polish("Romet")
pltts.pronounce_as(SMS="esemes")
pltts.pronounce_phonemes(CRM="tsɛɛrˈɛm")
```

`length_scale` powyżej 1,0 zwalnia mowę. `noise_scale` i `noise_w` sterują tym,
jak bardzo prozodia zmienia się między wywołaniami — dwa wywołania na tym samym
tekście nie są identyczne.

CLI: `--phonemes` pokazuje transkrypcję bez syntezy, `--no-acronyms` i
`--no-english` wyłączają pojedyncze poprawki, `-s` to tempo, `--threads`
ogranicza wątki ONNX. Wyjątki bez pisania kodu:

```sh
pltts --keep-polish Romet,Budimex "Nowy rower Romet."
pltts --pronounce SMS=esemes --pronounce-phonemes 'CRM=tsɛ ɛrˈɛm' "Wyślemy SMS z CRM."
```

## Wydajność

CPU, jedno zdanie, model 128 MB: **RTF 0,05–0,21**, czyli 5–20× szybciej niż
czas rzeczywisty. Pierwsze wywołanie jest wolniejsze o wczytanie modelu.

## Zależności i instalacja

`numpy`, `onnxruntime`, `piper-phonemize`. Uwaga na to ostatnie: oryginalny
`piper-phonemize` 1.1.0 ma wheele Linuxowe tylko do cp310, więc na aktualnym
interpreterze **nie da się go zainstalować**. Paczka bierze więc `-fix` dla
Pythona ≥ 3.10 i `-cross` poniżej; oba wystawiają ten sam moduł
`piper_phonemize`. Sprawdziłem, że zwracają **bajtowo identyczne fonemy** co
1.1.0 na 814 fonemizacjach — to nie kosmetyka, bo inne fonemy nie pokryłyby się
z id-kami, na których model był trenowany.

Wheel 0.1.1 waży 122 MB, z czego 128 MB to sam model (`pl_multi.onnx`) — jest w środku
świadomie, żeby instalacja była jednym krokiem i działała bez sieci.

## Ograniczenia

- Liczebniki rzymskie tylko 1–30 i tylko rodzaj męski: `w klasie II` wychodzi
  „w klasie drugi”, nie „drugiej”. Pełna fleksja jest w osobnym normalizatorze
  tekstu, nie tutaj — ta warstwa jest ostatnią linią obrony, nie normalizatorem.
- Frontend i checkpoint są sprzężone przez konwencję `ʂ`/`ʐ`. Podmiana modelu na
  taki, który nie był na niej trenowany, da bełkot w miejscu `trz`/`drz`.
- `lektor` w modelu wielogłosowym ma f0 o 26 Hz niższe od wzorca; jeśli barwa
  przeszkadza, jest osobny model jednogłosowy tego głosu.
- Reguła sylabiczna nie zna wyjątków poza dwoma wpisanymi. Nowy akronim, który
  wymawia się inaczej, niż wygląda, wymaga dopisania go do `_FORCE_SPELL` albo
  `_FORCE_WORD`.
