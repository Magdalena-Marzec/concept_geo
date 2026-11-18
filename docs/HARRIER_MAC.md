# Harrier-OSS-v1-27B INT4 — MAG / Mac Studio (etap przygotowawczy)

**Cel:** dedykowane embeddingi EN SCORE (203 x 9 = 1827) i EN Tatum (597),
`float32`, 5376D, L2, z nienaruszonymi starymi pipeline'ami.

## Dlaczego osobny adapter?

Harrier jest *embedding-tuned Gemma3 decoder-only*: **causal attention,
last non-padding token pooling, L2 normalization**. Nie jest generatywnym
Qwenem testowanym w `mac/embed_layers.py`.

Uwaga: w zbadanym kodzie `mlx_embeddings.models.gemma3_text.Model` standardowy
forward używa **bidirectional attention + mean pooling + Dense** (ścieżka
EmbeddingGemma). Harrier nie ma Dense i wymaga *innej* uwagi i poolingu.
Usunięcie samej Dense bez poprawienia pozostałych dwóch elementów jest
merytorycznie niepoprawne, nawet jeśli zwrócony wektor ma wymiar 5376.

Upstream: https://huggingface.co/microsoft/harrier-oss-v1-27b
Konwersja MLX INT4: https://huggingface.co/majentik/harrier-oss-v1-27b-MLX-4bit

## Co jest już gotowe

- `mac/harrier/config.json`: rodzina, repo, przypięta rewizja, parametry INT4,
  strategia osadzania, liczności zbiorów, SHA-256 SCORE.
- `mac/harrier/preflight.py`: audyt wejść, używa istniejącego `mac.data`;
  nie instaluje MLX, nie pobiera wag i nie wykonuje inferencji.
- `mac/tests/test_harrier_preflight.py`: niezależne testy kontraktu danych i
  zabezpieczenie przed `mean_pooling`/`bidirectional attention`.

## Komendy teraz — na MacBooku albo Linuxie

```bash
python3 -m mac.harrier.preflight
python3 -m unittest mac.tests.test_harrier_preflight -v
```

Jeśli Tatum jest już na danym komputerze:

```bash
python3 -m mac.harrier.preflight --require-tatum
```

## Do ukończenia przed uruchomieniem joba Mac Studio

1. Dodać osobny, przypięty runtime MLX dla Harriera, bez edycji środowiska
   `.venv-mac` i bez uruchamiania generatywnego Qwena.
2. Użyć **causal Gemma3 backbone + last-token pooling + L2**; bez Dense;
   zweryfikować na krótkich tekstach z niezależnym referencyjnym BF16.
3. Przeprowadzić smoke test, kontrolę wymiarów / norm / deterministyczności /
   użycia RAM i offline.
4. Dopiero po sukcesie włączyć batch_size=1, `memmap` i `resume` dla
   1827 i 597 wierszy, z manifestami i identyfikatorami bez cytatów Tatuma.

**Status: NIE URUCHAMIAĆ PEŁNEJ EKSTRAKCJI.** Brak sprawdzonego adaptera
Harrier/MLX i walidacji referencyjnej. Pomyślne testy tego pakietu sprawdzają
wyłącznie dane/kontrakt, nie dowodzą poprawności konwersji.
