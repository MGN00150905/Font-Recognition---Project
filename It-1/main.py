from examples import ocr_utils
import os


cwd = os.getcwd()

filter = dict({'ARIAL.csv':'scanned'})
df = ocr_utils.read_file(cwd+'/fonts.zip', filter)


# Entry = df.head()
# index location. comma means all rows, only 10 columns
examples = df.head().iloc[:,10:]
print(examples)

ocr_utils.montage(examples)