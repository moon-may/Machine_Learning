# Predicting
Выпускная квалификационная работа. Прогнозирование количества отчисленных студентов с помощью ML

## Данные
Для обучения модели использовался набор данных с Kaggle [Predict students' dropout and academic success](https://www.kaggle.com/datasets/thedevastator/higher-education-predictors-of-student-retention/). 

Оригинальное исследование и авторство: [Predict students' dropout and academic success](https://zenodo.org/records/5777340#.Y7FJotJBwUE). 

Набор данных распространяется по лицензии CC0. 

## Структура проекта

`notebooks` – каталог содержит файлы Jupyter Notebook: 
- `preprocessing` – предварительна обработка данных
- `models_learning` – обучение моделей
    
`datasets` – коллекция использованных наборов данных: 
- `data_processed.csv` – данные после предобработки
- `test.csv` – набор, использованный при отладке
- `new_data.csv` – набор данных для демонстрации работы приложения

`src` – каталог для вспомогательных файлов программного кода

Запуск приложения производится из файла `main.py`. 

Обученная модель сохранена в `model.keras`.

`requirements.txt` содержит версии библиотек, использованных при разработке. 
