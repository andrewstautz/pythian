docker build -t stevemcquaid/pythian:latest .
docker run -it -p 80:8050 --rm stevemcquaid/pythian:latest python app.py

