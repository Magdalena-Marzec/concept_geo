# MAG: kolejna seria embeddingów na Rysach (V100 32 GB)

Pakiet jest **dodatkiem** do przesłanego `code-kopia.zip`, nie zastępuje dotychczasowego pipeline'u Qwen3 ani istniejących wyników. Przeanalizowano lokalny kod i schematy CSV. **Nie uruchomiono prawdziwego inference na V100 i nie pobrano checkpointów** w środowisku, w którym przygotowano pakiet; dlatego obowiązkowy jest smoke test przed nocnym zadaniem.

## 0. Bezpieczeństwo Git i materiałów Tatuma

W przesłanym archiwum `.git` zawiera już *śledzone* oryginalne teksty książki (`tatum/source/` i `tatum/data/`). Stan taki może ujawnić teksty przy pushu do publicznego repo. Nie rób `git add .` ani `git push`, dopóki tego nie rozwiążesz. Sam `.gitignore` nie usuwa plików już śledzonych i nie usuwa ich z historii. Ten pakiet nie zawiera materiałów Tatuma. Do przeniesienia użyj prywatnego scp/rsync lub bezpiecznego repo bez danych źródłowych; skrypty na klastrze oczekują plików na dysku.

## 1. Dodanie do repo

W katalogu, gdzie jest `embed_three_qwen.py` i `score_concepts_completed.csv`, skopiuj katalog `v100_next_models/` z tego ZIP-a. Do istniejącego `.gitignore` **dopisz**, nie nadpisuj, reguły z `v100_next_models/.gitignore.snippet`.

Wykorzystuje te same parametry Rysy co obecny kod: `--partition=gpu`, `--account=g104-2594`, `--constraint=volta32G` i `--gres=gpu:1`. Sprawdź bieżące limity partycji i konto przed zgłoszeniem zadania.

Dotychczasowy `.venv` ma `torch==2.6.0` i `transformers==4.57.6`, ale nie ma pakietu Sentence Transformers. Aby zminimalizować zmiany, przygotuj najlepiej oddzielne środowisko do nowych modeli albo zachowaj `pip freeze` i doinstaluj:

```bash
module use /apps/modulefiles/common/common
module load anaconda/23.3.1
source .venv/bin/activate
python -m pip freeze > pip_freeze_before_v100_next.txt
python -m pip install -r v100_next_models/requirements-v100-extra.txt
python -m pip check
```

UWAGA: `pip_freeze_before_v100_next.txt` nie zawiera prywatnych tekstów. Jeśli na klastrze użyjesz innego środowiska niż `.venv`, zmień `source .venv/bin/activate` w trzech skryptach `.sbatch`. NVIDIA NV-Embed-v2 ma niestandardowy kod i może wymagać osobnej, starszej wersji `transformers`; **nie należy bez testu downgrade'ować wspólnego środowiska Qwen**.

## 2. Walidacja wejścia bez GPU

```bash
python -m v100_next_models.encode --dry-run
python -m unittest discover -s v100_next_models/tests -v
```

Skrypt sprawdza 203 SCORE, 1827 tekstów, 597 Tatum (252 fragmenty, 197 akapitów, 148 podpisów), brak pustych pól i kolejność `row_id`. Test dodatkowo porównuje generowaną tabelę SCORE z funkcją `embed_three_qwen.build_score_metadata` w aktualnym repo.

## 3. Pobranie modeli — etap niezależny od inference

Kolejno pobierane są wyłącznie pliki wymagane do ładowania checkpointów PyTorch (bez duplikatów `.bin` czy TF/Flax). Revision SHA jest zapisywany do manifestu modelu.

```bash
JOB=$(sbatch --parsable v100_next_models/download_v100.sbatch)
echo "$JOB"; squeue -j "$JOB"
```

Można ograniczyć pobieranie na przykład do lekkich modeli:

```bash
MODELS='medembed pubmedbert sapbert' sbatch v100_next_models/download_v100.sbatch
MODELS='nvembed nemotron_medical' sbatch v100_next_models/download_v100.sbatch
```

NV-Embed-v2 może wymagać zalogowania na Hugging Face i zaakceptowania warunków dostępu. Nie wkładaj tokenu do skryptów ani repo. Jeżeli compute nodes nie mają internetu, pobierz modele w dozwolonym miejscu z siecią i przenieś **całe** `models/next_v100/` prywatnie. `models/` jest ignorowany przez Git.

Modele mogą zająć łącznie kilkadziesiąt GB dysku. Przed startem: `df -h .` i sprawdzenie limitu quota.

## 4. Szybkie smoke testy (tylko wybrane zdania; bez pełnego korpusu)

```bash
JOB=$(sbatch --parsable v100_next_models/smoke_v100.sbatch)
echo "$JOB"; squeue -j "$JOB"
```

Każdy model otrzymuje tylko sześć tekstów SCORE i sześć tekstów Tatuma, plus kontrolę powtarzalności identycznych tekstów. Model jest ładowany w osobnym procesie, aby pamięć po poprzednim była zwolniona. Dane wyjściowe trafiają pod `embedding_results/next_v100/smoke/<model>/`.

Można też testować pojedynczo:

```bash
MODELS='medembed pubmedbert sapbert' sbatch v100_next_models/smoke_v100.sbatch
MODELS='nvembed' sbatch v100_next_models/smoke_v100.sbatch
MODELS='nemotron_medical' sbatch v100_next_models/smoke_v100.sbatch
```

`SMOKE VALIDATION OK` dla każdego modelu oznacza pomyślny test kształtu/normalizacji i działania na danych, **nie** pełną kontrolę poprawności semantycznej. Jeśli NV lub Nemotron nie ładują się w wersjach bibliotek z `.venv`, rozwiązuj zależności w osobnym środowisku zamiast wyłączać niestandardowy pooling.

## 5. Zadanie nocne dla modeli, które zaliczyły smoke

```bash
JOB=$(sbatch --parsable v100_next_models/night_v100.sbatch)
echo "$JOB"; squeue -j "$JOB"
```

Dla każdego modelu skrypt wykonuje osobno SCORE (1827) i Tatum (597), zapisuje dwa `*.npz` i manifest, a następnie zwalnia GPU. Jeżeli jeden model się nie powiedzie, następny nadal startuje. Nocny skrypt wymaga zgodnego manifestu smoke i nie nadpisuje ukończonych wyników bez `--force`.

Opcjonalnie jedna noc tylko dla małych modeli:

```bash
MODELS='medembed pubmedbert sapbert' sbatch v100_next_models/night_v100.sbatch
```

**Wyniki:** `embedding_results/next_v100/models/<name>/score_<name>.npz`, `tatum_<name>.npz`, `manifest.json`. Każdy NPZ zawiera `embeddings_native` float32 po L2; rozmiary natywne różnią się między modelami. **Nie ma `embeddings_1024`**: w przeciwieństwie do Qwen3 te modele nie mają jednolicie potwierdzonego MRL i obcięcie 1024 współrzędnych nie gwarantuje sensownego benchmarku.

Wspólne metadane są w `embedding_results/next_v100/metadata/score_embeddings_metadata.csv` i `tatum_embeddings_metadata.csv`. Drugi z nich zawiera tylko identyfikatory i typy, **bez tekstów Tatuma**. Kolejność jest 1:1 zgodna z wejściowymi tabelami i dotychczasowym pipeline'em Qwen.

## 6. MedEmb 2B/8B — eksperymentalnie, osobna ścieżka

To modele **Qwen3-VL-Embedding**, dostrajane do danych medycznych; nie należy traktować ich jako niezależnej architektury od Qwen. Do ich inference potrzebny jest kod `Qwen3VLEmbedder` z oficjalnego repo [Qwen3-VL-Embedding](https://github.com/QwenLM/Qwen3-VL-Embedding). Nie kopiuj przez Git chronionych danych Tatuma. Po zapoznaniu się z licencją i kodem zainstaluj go w katalogu ignorowanym przez Git:

```bash
mkdir -p third_party
git clone --depth 1 https://github.com/QwenLM/Qwen3-VL-Embedding.git third_party/Qwen3-VL-Embedding
# Sprawdź commit i zależności zgodnie z dokumentacją autora.
MODELS='medemb2b' sbatch v100_next_models/download_v100.sbatch
MODEL='medemb2b' sbatch v100_next_models/experimental_medemb_v100.sbatch
```

Dla modelu 8B analogicznie `medemb8b`; 8B może przekroczyć dostępny VRAM mimo batch=1. W kodzie ustawiono FP16, eager attention (V100 bez BF16/FA2) i wywołanie `Qwen3VLEmbedder.process([{'text': ...}])` zgodne ze sposobem użycia rodziny bazowej. **Nie potwierdziłem, że waga/adapter konkretnego modelu MedEmb da się wczytać tym wrapperem.** Jeśli smoke nie przejdzie, nie włączać go do nocnego joba. Po udanym smoke i ocenie czy model rzeczywiście używa medycznego adaptera:

```bash
MODELS='medemb2b' sbatch v100_next_models/night_v100.sbatch
```

W tym przypadku trzeba dopisać `--allow-experimental` w pętli nocnego skryptu lub uruchomić bezpośrednio:

```bash
python -u -m v100_next_models.encode --model medemb2b --full --require-smoke --allow-experimental
```

## 7. Zasady metodologiczne

- Wersja bazowa testu to **symetryczne osadzanie bez instrukcji** jak w istniejącym Qwen3. Każdy model używa jednak własnego, poprawnego poolingu: ST, SapBERT CLS, NV latent attention, Qwen3VL official process.
- Dla NV-Embed i modeli instrukcyjnych warto później wykonać **drugi, jawnie oddzielony wariant instruction/retrieval**. Nie mieszać jego wyników z baseline symetrycznym.
- Limit tokenów w pierwszej rundzie to 512 dla małych BERT, 768 dla NV/Nemotron, 512/384 dla MedEmb. Model zapisuje `token_lengths_<dataset>.csv` (bez oryginalnych tekstów) i szacunek przycięć w manifeście, jeśli tokenizer jest dostępny; dla modeli z niestandardowymi szablonami liczby są przybliżone. Modele mogą obcinać długie akapity; rozszerz limit dopiero po analizie tych plików i kosztów. Nie porównuj długości kontekstu bez kontroli przycięć.
- Po zapisie przebiegu można liczyć ten sam `cloud`, AUC, Top-1/Top-5/MRR, centrowanie i marginesy, korzystając z dotychczasowych notebooków, ale wskazując nowe ścieżki i natywne D.
- Licencje/dostęp: NV-Embed-v2 CC-BY-NC-4.0; SapBERT i PubMedBERT Apache-2.0; MedEmbed Apache-2.0. MedEmb i pochodne Nemotron wymagają zweryfikowania dokładnych licencji każdego checkpointu przed dalszym udostępnianiem.

## 8. Status weryfikacji

W środowisku przygotowania uruchomiono **tylko** walidację CSV i testy kodu bez CUDA. Nie wolno uznawać pobierania checkpointów ani testów GPU za wykonane. Skrypty zapisują błędy ładowania modeli. Aby ponowić nieukończony model, uruchom jego proces z `--force` (przeliczy również SCORE, jeśli plik już istnieje); ukończone modele pozostają nietknięte.
