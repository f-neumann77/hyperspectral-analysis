## Утилита для анализа гиперспектральных изображений

### Формат хранения гиперспектральных изображений
Гиперспектральные изображения хранятся в .mat файлах. Маски для гиперспектральных изображений так же хранятся в .mat файлах.

#### структура .mat файла для ГСИ:

'image': гиперспектральное изображение (numpy array)


#### структура .mat файла для маски ГСИ:

'mask': маска для гиперспектрального изображения (numpy array)

'labels': словарь значений маски и их названий ( {0: void, 1: background, etc} )

