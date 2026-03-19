# Гипотеза 3: Для каких типов рядов какая трансформация ряда полезнее?

## Структура

- `run_experiment.py`: запускает эксперимент
- `config.py`: всякие разные константы
- `src/`:
    - `cluster.py`: кластеризация рядов по признакам
    - `data.py`: загрузка и разбиение данных
    - `features.py`: создание признаков для катбуста
    - `models.py`: обучаем и оцениванеим модельки
    - `metrics.py`: рассчет метрик smape и mase
- `results/`:
    - `analysis_results.ipynb`: визуализация результатов

## Как запустить

```bash
pip3 install -r requirements.txt
python3 run_experiment.py
```

Затем надо открыть и запустить юпитер ноут из `results/analysis_results.ipynb`

## Метрики

- sMAPE
- MASE
