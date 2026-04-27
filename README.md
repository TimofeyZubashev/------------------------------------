# Трансформер для саммаризации текстов

Python-проект для обучения и инференса собственного text summarizer на базе Transformer encoder-decoder архитектуры.

Проект переписан из финального Kaggle/Jupyter notebook `sum-sum (2).ipynb` в модульную структуру. Notebook оставлен как исходный экспериментальный артефакт, а основной код теперь находится в `src/` и `scripts/`.

## Что внутри

- CNN/DailyMail loader для датасета `gowrishankarp/newspaper-text-summarization-cnn-dailymail`;
- простой word-level tokenizer;
- Train / Validation / Test Dataset и DataLoader;
- Transformer encoder;
- Transformer decoder summarizer;
- обучение с AMP и `DataParallel`;
- продолжение обучения с checkpoint;
- inference по произвольному тексту или validation samples;
- построение графиков loss/perplexity.

## Структура

```text
src/text_summarizer/
  config.py        # конфиги датасета, обучения и fine-tuning
  tokenizer.py     # SimpleTokenizer
  data.py          # Dataset/DataLoader и загрузка CNN/DailyMail
  encoder.py       # Transformer encoder
  model.py         # Transformer summarizer + checkpoint loading
  training.py      # train/evaluate loop, AMP, DataParallel helpers
  inference.py     # загрузка модели и генерация summary
  plotting.py      # графики и статистики

scripts/
  train.py         # обучение с нуля
  fine_tune.py     # продолжение обучения с checkpoint
  infer.py         # инференс
  plot_history.py  # график по history.json

checkpoints/
  transformer_summarizer_checkpoint/
    summarizer_checkpoint.pt
```

## Важная правка после Kaggle ошибки

Ошибка:

```text
RuntimeError: grad can be implicitly created only for scalar outputs
```

возникала из-за `nn.DataParallel`: когда модель возвращает scalar loss с каждой GPU, PyTorch собирает их в вектор. Поэтому в `src/text_summarizer/training.py` loss всегда приводится к скаляру через:

```python
loss = output.loss.mean()
```

Это исправляет `backward()` на двух T4.

## Установка

```bash
python3 -m pip install -e ".[dev]"
```

На Kaggle зависимости `torch`, `pandas` и `matplotlib` обычно уже доступны.

## Обучение с нуля

```bash
python3 scripts/train.py \
  --data-dir /kaggle/input/datasets/gowrishankarp/newspaper-text-summarization-cnn-dailymail \
  --epochs 100 \
  --batch-size 32 \
  --gradient-accumulation-steps 1
```

Если будет `CUDA out of memory`, начните с:

```bash
python3 scripts/train.py \
  --batch-size 16 \
  --gradient-accumulation-steps 2
```

## Продолжить обучение с checkpoint

```bash
python3 scripts/fine_tune.py \
  --checkpoint checkpoints/transformer_summarizer_checkpoint/summarizer_checkpoint.pt \
  --epochs 5 \
  --batch-size 8
```

## Инференс

По тексту:

```bash
python3 scripts/infer.py \
  --checkpoint checkpoints/transformer_summarizer_checkpoint/summarizer_checkpoint.pt \
  --text "Your article text here"
```

По validation examples:

```bash
python3 scripts/infer.py \
  --checkpoint checkpoints/transformer_summarizer_checkpoint/summarizer_checkpoint.pt \
  --num-samples 5
```

## Графики

```bash
python3 scripts/plot_history.py \
  --history /kaggle/working/summarizer/history.json \
  --output /kaggle/working/summarizer/history.png \
  --no-show
```

## Kaggle GPU

Для двух T4 используется `nn.DataParallel`. Это делит batch между GPU, но не объединяет память двух карт в одну. Если не хватает памяти, уменьшайте:

1. `batch_size`;
2. `max_source_length`;
3. `max_target_length`;
4. `d_model` или количество слоев.

