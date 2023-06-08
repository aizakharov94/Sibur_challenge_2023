## Sibur challenge 2023 (Прогноза количества продуктов в производстве. 2-ая задача)

Top 3 private solution (1.720% public and 1.724% private MAPE) <br/>
Event Link: [Sibur challenge 2023 Event](https://ai-community.com/sibur-challenge-2023) <br/>
Competition Link: [Sibur challenge 2023](https://platform.aitoday.ru/event/9) <br/>

#### Структура репозитория:

1) Pipeline.ipynb - обучение моделей, получение обученных моделей, получение файла инференса predict.py. <br/>
&nbsp;&nbsp;&nbsp;&nbsp; На выходе получаем: <br/>
|-- predict.py - файл с инференсом <br/>
|-- model_\*.pkl - Модели регрессии для различных таргетов (продуктов) и различного типа газа <br/>

2) predict.py - файл с инференсом, который принимает модели с Pipeline.ipynb и применяет для предсказания количества продусков в производстве. <br/>
3) top3_solution.zip - решение (файл predict.py и модели) на отправку <br/>
4) train.parquet - train data <br/>
