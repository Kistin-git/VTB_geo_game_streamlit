# VTB Geo Game Streamlit

Игровое приложение, в котором пользователь кликом на карте выбирает место для банкомата, а модель показывает более сильную альтернативу.

## Запуск

```bash
python3 -m pip install -r requirements.txt
streamlit run app.py
```

Перед запуском нужны артефакты в `data/`, которые экспортируются из основного репозитория командой:

```bash
python3 ../VTB_geo_project_streamlit/scripts/build_all.py --game-repo .
```
