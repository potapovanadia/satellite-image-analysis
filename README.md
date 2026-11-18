# Обробка та аналіз супутникових знімків класичними методами

STEM-проєкт з дисципліни «Комп'ютерна графіка та візуалізація», КНУ імені Тараса Шевченка.
Виконала: **Потапова Надія**, ІПЗ-23. Формат: індивідуально, варіант 2 (два пайплайни).

Програма обробляє аерофотознімки з набору UC Merced Land Use двома пайплайнами
класичних методів, рахує метрики якості та порівнює пайплайни між собою за різних
рівнів шуму. Усі алгоритми обробки реалізовано вручну на NumPy, без OpenCV і без
методів машинного навчання.

| Пайплайн | Стадії (категорії методів) | Задача |
|---|---|---|
| **№1** | Кувахара (фільтрація) → RGB→HSV (колір) → GLCM (текстура) → k-means (сегментація) | поділ знімка на області: поле, ліс, вода, забудова |
| **№2** | CLAHE (гістограма) → Кенні (контури) → закриття + чистка (морфологія) → Хаф (прямі) | виділення доріг і прямолінійних меж |

Детальний опис алгоритмів і схеми стадій: [`docs/pipelines.md`](docs/pipelines.md).
Обґрунтування параметрів: [`docs/parameters.md`](docs/parameters.md).

## Структура проєкту

```
satellite-image-analysis/
├── config.yaml          # усі параметри методів, шуму, порогів перевірки
├── requirements.txt     # залежності
├── satimg/              # пакет програми (опис модулів — docs/pipelines.md)
├── tests/               # автотести pytest (110 тестів)
├── data/
│   ├── dataset/         # вибірка: 5 класів × 20 знімків 256×256
│   ├── manifest.csv     # опис вибірки
│   └── README.md        # джерело та правило відбору
├── docs/                # документація та результати для звіту
└── results/             # створюється програмою (не зберігається в Git)
```

## Встановлення

Потрібен Python 3.10 або новіший.

```bash
git clone https://github.com/potapovanadia/satellite-image-analysis.git
cd satellite-image-analysis
python -m pip install -r requirements.txt
```

## Інструкція користувача

Усі команди виконуються з кореневої папки проєкту.

### 1. Перевірити, що все працює

```bash
python -m pytest              # 110 тестів, ~10 с
python -m satimg dataset      # перевірка вибірки: 5 класів по 20 знімків
```

### 2. Запустити пайплайни

```bash
python -m satimg run --pipeline p1            # пайплайн №1 на всій вибірці
python -m satimg run --pipeline p2            # пайплайн №2
python -m satimg run --pipeline all           # обидва (повний експеримент, ~15–25 хв)
python -m satimg run --pipeline all --limit 5 # швидка перевірка на 5 знімках
python -m satimg run --pipeline p1 --per-class 2 --noise none gaussian_20
```

Для кожного знімка програма виконує прогін без шуму та з шумом (гаусів σ = 10, 20, 30,
імпульсний 5 %) і зберігає:

- `results/metrics_p1.csv`, `results/metrics_p2.csv` (+ `.json`) — метрики кожного запуску;
- `results/<pipeline>/<шум>/<клас>/<знімок>/` — вхід і результат кожної стадії
  (`00_input.png`, `01_…png` … `04_…png`) для умов зі списку `output.save_noise`;
- `results/errors.csv` — файли, які не вдалося прочитати (якщо такі були).

### 3. Застосувати один метод до одного знімка

```bash
python -m satimg method canny --image data/dataset/freeway/freeway00.tif
python -m satimg method kuwahara --image data/dataset/forest/forest00.tif --noise gaussian_20
```

Доступні методи: `kuwahara`, `hsv`, `glcm`, `kmeans`, `clahe`, `canny`, `morphology`, `hough`.
Результат зберігається в `results/methods/`.

### 4. Аналіз і перевірка коректності

```bash
python -m satimg analyze   # таблиці та графіки -> results/analysis/
python -m satimg verify    # еталонні сценарії (C1) -> results/verify_results.csv
```

### 5. Зміна параметрів без зміни коду

Параметри задаються в `config.yaml`. Будь-який з них можна перевизначити з командного
рядка через `--set` (можна кілька разів):

```bash
python -m satimg run --pipeline p1 --set pipeline1.kuwahara.window=5 --set pipeline1.kmeans.k=3
python -m satimg config --set pipeline2.canny.sigma=2.0   # показати підсумкову конфігурацію
python -m satimg run --config my_config.yaml               # інший файл конфігурації
```

Підбір параметра на частині вибірки:

```bash
python -m satimg sweep --pipeline p2 --param pipeline2.canny.sigma --values 1.0 1.4 2.0
```

Випадкові компоненти (шум, ініціалізація k-means) використовують фіксований `seed: 42`,
тому результати відтворювані.

## Якість коду

```bash
python -m pytest                  # автотести: кожен метод перевіряється на зображеннях 3×3–7×7
python -m pylint satimg tests     # статичний аналіз: 10.00/10
```

## Основні результати

Повний аналіз — у звіті та в [`docs/results/`](docs/results/).

- Перевірка коректності на 72 еталонних тест-кейсах: **64 пройдено (89 %)**.

## Вибірка

UC Merced Land Use Dataset (Yang & Newsam, 2010) — опис і правило відбору в
[`data/README.md`](data/README.md).
