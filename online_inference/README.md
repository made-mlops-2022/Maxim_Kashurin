# Инструкция по развертке Docker локально
~~~
docker build -t hw2:latest .
docker run --rm -p 8080:8080 hw2:latest
python3 make_request.py 


~~~
# Тесты рабочие , но почему то вместо 400 возвращатеся 422 , подскажите пж как это пофиксить?
~~~
сначала запускаем контейнер , потом тесты
tests/tests.py
~~~

# Docker: получить образ с Docker Hub
~~~
docker pull maxkashh/hw2:latest
docker run --rm -p 8080:8080 maxkashh/hw2:latest
~~~
# Применение:

Метод   /health чтобы поверить  проверить, готов ли сервис к вызову /predict

После запуска контейнера Docker с онлайн-моделью вывода запустите скрипт, запрашивающий службу с помощью 
~~~
python3 make_request.py 
~~~
Если данные введены неверно по идее должен вернуться 400 Error , но у меня почему то возвращается 422


# Docker Оптимизации:

    Добавил несколько файлов в .dockerignore - особо не выиграл, было 2.50GB стало 2.49GB Когда же выбрал другую начальную среду (вместо python:3.9 python:3.6-slim-stretch) образ стал весить 2.38GB
    стоило перетащить команду COPY . /project вниз, а

COPY requirements.txt /project
RUN pip install --no-cache-dir -r requirements.txt

наверх в Dockerfile и теперь его пересборка стала в несколько раз быстрее