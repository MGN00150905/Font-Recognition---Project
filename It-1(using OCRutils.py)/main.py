from examples import ocr_utils
import os


# Getting current directory
cwd = os.getcwd()


filter = dict({'ARIAL.csv':'scanned'})

# Read file function
df = ocr_utils.read_file(cwd+'/fonts.zip', filter)


# Entry = df.head()
# index location. comma means all rows, but only from 10th column onwards
examples = df.head().iloc[:,10:]
print(examples)

ocr_utils.montage(examples)
