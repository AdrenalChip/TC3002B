# TC3002B

## Actividad 2.1 y Actividad 2.2
### Generación o Selección del Set de Datos y Preprocesado de los datos 
Se obtuvo un dataset de la plataforma _Kaggle_ que contiene imagenes de diferentes videojuegos. Este dataset cuenta con 10 clases y se tiene alrededor de 1000 imágenes por clase.
Las clases son las siguientes:
* Terraria
* Roblox
* Minecraft
* God of War
* Genshin Impact
* Free Fire
* Forza Horizon
* Fortnite
* Apex Legends
* Among Us

Para realizar la separación de los sets de prueba y entrenamiento se uso el _script_ [*movetraintestfiles.py*](https://github.com/AdrenalChip/TC3002B/blob/main/movetraintestfiles.py) que mueve de una carpeta a otra las imagenes de manera aleatoria, respetando la relacion 80% - 20% por cada clase existente.
En el archivo [*preprocess.py*](https://github.com/AdrenalChip/TC3002B/blob/main/preprocess.py) carga las diferentes imágenes, tanto de entrenamiento como validación y normaliza los valores entre 0 y 1. 

Más información del dataset dentro del archivo [*gameplay_images.txt*](https://github.com/AdrenalChip/TC3002B/blob/main/gameplay_images.txt).   

[Link al Drive de las Imágenes](https://drive.google.com/drive/folders/11SkaT7sGMPT6QlXJ7xYzSnjdPyowvUmV?usp=sharing)

[Link al Dataset de Kaggle](https://www.kaggle.com/datasets/aditmagotra/gameplay-images)
