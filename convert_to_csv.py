# dependencies
# OS modules
import os
# Pandas
import pandas as pd
# In-built time module
import time
# tqdm for progress bars
from tqdm import tqdm
# Pillow Image Library
from PIL import Image
# Numpy module
import numpy as np

# A list for column names of csv
columnNames = list()
# A column for label
columnNames.append('label')
# Other pixels column
# replace 784 with your image size, here it is 28x28=784
# iterate and build headers
for i in range(784):
    pixel = str(i)
    columnNames.append(pixel)

# Create a Pandas dataframe for storing data
train_data = pd.DataFrame(columns = columnNames)

# calculates the total number of images in the dataset initially 0
num_images = 0

# iterate through every folder of the dataset
for i in range(0, 58):

    # print messeage
    print("Iterating: " + str(i) + " folder")

    # itreate through every image in the folder
    # tqdm shows progress bar
    for file in tqdm(os.listdir(str(i))):
        # open image using PIL Image module
        img = Image.open(os.path.join(str(i), file))
        # resize to 28x28, replace with your size
        img = img.resize((28, 28), Image.NEAREST)
        # load image  
        img.load()
        # create a numpy array for image pixels
        imgdata = np.asarray(img, dtype="int32")
       
        # temporary array to store pixel values
        data = []
        data.append(str(i))
        for y in range(28):
            for x in range(28):
                data.append(imgdata[x][y])

        # add the data row to training data dataframe
        train_data.loc[num_images] = data

        # increment the number of images
        num_images += 1

# write the dataframe to the CSV file
train_data.to_csv("train_converted.csv", index=False)