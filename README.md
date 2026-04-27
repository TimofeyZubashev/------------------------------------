# Трансформер для саммаризации текстов

Репозиторий для пошаговой реализации нейросетевого суммаризатора текста на базе Transformer encoder-decoder архитектуры.

На текущем этапе реализованы:

- токенные и позиционные эмбеддинги;
- encoder-decoder Transformer;
- causal mask для авторегрессионного декодера;
- loss для teacher forcing;
- greedy generation для быстрой проверки инференса;
- training loop, готовый принять будущие DataLoader-ы;
- логирование метрик и построение графиков обучения.

В модульном коде `src/` и `scripts/train.py` DataLoader-ы пока оставлены точкой подключения для следующего шага. В Kaggle notebook уже добавлена учебная реализация:

- Train / Test Datasets;
- Train / Test DataLoaders.

## Структура

```text
scripts/
  train.py
  plot_training_metrics.py
notebooks/
  kaggle_text_summarizer_pipeline.ipynb
src/text_summarizer/
  __init__.py
  encoder.py
  model.py
  transformer.py
tests/
  test_transformer.py
```

## Основные файлы

- `scripts/train.py` - скрипт тренировки. Сейчас содержит training loop, optimizer, evaluation, checkpoint saving и явную точку подключения будущих DataLoader-ов.
- `src/text_summarizer/model.py` - seq2seq Transformer-суммаризатор: decoder, loss, checkpoint API и inference через `generate` / `summarize_token_ids`.
- `scripts/plot_training_metrics.py` - проверка процесса обучения по `metrics.jsonl`: строит графики loss, perplexity и learning rate.
- `src/text_summarizer/encoder.py` - encoder модели: token embeddings, positional encoding и Transformer encoder.
- `notebooks/kaggle_text_summarizer_pipeline.ipynb` - self-contained Kaggle notebook с Dataset, DataLoader, моделью, тренировкой, графиками, AMP, DataParallel для двух GPU и тестовым inference. Настроен на CNN/DailyMail: `/kaggle/input/datasets/gowrishankarp/newspaper-text-summarization-cnn-dailymail`, файлы `train.csv`, `validation.csv`, `test.csv`, колонки `article` и `highlights`.

## Быстрая проверка

```bash
python3 -m pip install -e ".[dev]"
python3 -m pytest
```

## Dry run тренировки

Проверка training step на синтетическом батче, без Dataset/DataLoader:

```bash
python3 scripts/train.py --dry-run --d-model 64 --num-heads 4 --dim-feedforward 128 --num-encoder-layers 2 --num-decoder-layers 2
```

## Графики обучения

После появления `runs/baseline/metrics.jsonl`:

```bash
python3 -m pip install -e ".[viz]"
python3 scripts/plot_training_metrics.py
```
