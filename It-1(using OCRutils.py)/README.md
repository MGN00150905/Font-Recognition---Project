# Importing ocr_utils.py functions

A utility python program for accessing the dataset was offered with the dataset. This was a Github Repo with the utility program along side 40 different machine learning programs from the book Python Machine Learning.

The particular function I was looking for from 'examples/ocr_utils.py' was 'montage' and 'read_data'.

```py
def montage(X, maxChars = 2500, title=''):

    count,h, w = np.shape(X)    

    separator_size = 5
    count = min(maxChars,count)

    nCol = int(math.ceil(math.sqrt(count)))
```

Unfortunately these functions were deprecated and outdated. I was unable to display a square matrix of characters with the 'montage' function now was I able to read in the data with the read_data function.

The next step was now to figure out how to load the data myself by creating my own utility functions.
