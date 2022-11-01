# Архитектурные и тактические решения проекта:
___
* ### Под разные моменты работы модели были созданы  разные сущности для выполнения соответствующих задач( реализовано всё в виде наследования сущностей , каждый следующий шаг опирается на прошлый)

  + ### ___Preprocess.py___  - содержит класс ___PreprocessingModel___ для (чтения датасета, разбиения выборки на тестовую и обучающую и выделение целевого признака) все операции происходят в конструкторе
  + ### ___Transform.py___ -  содержит класс ___TransformModel___ для создания пайплайна с числовыми фичами датасета и стандартизацией выборок(все операции происходят аналогично  в конструкторе как и в следующих классах)
  + ### ___Train.py___ - содержит класс ___TrainModel___ для обучения модели/ей , предсказания , создания пайплайна и получения метрик на выходе
  + ### ___Exporting.py___ - содержит класс ___TrainModel___ для выгрузки метрик и моделей в фалы а также для сериализации модели через  pickle

___
 ### __Где и как конкретно они используются в коде?__
    В методе run_train_pipeline происходит последовательное создание сущностей:

        params_from_yaml=read_training_pipeline_params(mcfg,cfg,name)
        preproc =  PreprocessingModel(params_from_yaml)
        tf =  TransformModel(params_from_yaml)
        train =  TrainModel(params_from_yaml)
        export =  ExportModel(params_from_yaml)
Переменная params_from_yaml содержит в себе распарщенные значения yaml файла.
И дальше они передаются сущностям и те уже с ними работают

## Для валидации использовались ___Датаклассы___ и библиотека Marshmellow
    Изначально провел эксперимент с Hydra , но всё ломалось и я забил.
    Все описания и манипуляции производились в файле cfg.py.
    Для создания и валидации схем использовался метод read_training_pipeline_params

### По какому принципу работает метод read_training_pipeline_params в cfg.py?
    Подразумевается что cfg файлы состоят из двух видов :
    * models.yaml содержит только названия моделей
    * остальные yaml файлы в папке configs содержат информацию уже необходимую для обучения , валидации ,  разбиения и тп
#### read_training_pipeline_params содержит путь до cfg файла с названиями моделей models.yaml(В данном случае их 2 можно использовать для масштабирования придется добавлять в cfg.py дополнительные условия и датаклассы) ,  cfg файл для описания модели и само название модели name которую хотим юзать . 
___
# Как запустить и использовать?
 
  + # Установка (Ubuntu/Unix)
      ```
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
      ```
  + # Использование DecisionTreeClassifier
    ```
    
    python3 ml_project/train_pipeline.py configs/models.yaml configs/DecisionTreeConfig.yaml DecisionTreeClassifier
    
  + # Использование CatBoostClassifier
    ```
     python3 ml_project/train_pipeline.py configs/models.yaml configs/CatBoostConfig.yaml CatBoostClassifier
    ```
  + # Результаты (метрики и готовые модели лежат в models)
  

    
    
    

    
    



