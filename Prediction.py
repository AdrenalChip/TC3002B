import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

base_dir = '.\\Dataset'
queries_dir = os.path.join(base_dir,'Queries')
batch_size = 13

queries_ds = ImageDataGenerator(
    rescale = 1./255,
    )

queries_generator = queries_ds.flow_from_directory(
    queries_dir,
    target_size = (50,50),
    batch_size = batch_size,
    class_mode = 'categorical',
    color_mode = 'grayscale'
)

from tensorflow.keras import optimizers
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.models import load_model

savedModel = load_model('videogame_img_v3.h5')

queries_imgs = queries_generator[0][0]
queries_labels = queries_generator[0][1]

predictions = savedModel.predict(queries_imgs)
classes_x = np.argmax(predictions,axis=1)
classes_x
queries_labels_y = np.argmax(queries_labels,axis=1)

from tensorflow.math import confusion_matrix

mat = confusion_matrix(classes_x, queries_labels_y, num_classes= 10)
print('                   ', 'Among Us ',                             'Apex Legends ',                   'Fortnite ',                    'Forza Horizon ',                 'Free Fire ',                 'Genshin Impact ',                       'God of War ',                      'Minecraft ',                     'Roblox ',                   'Terraria') 
print('pred Among Us      ',"   ", np.array(mat[0][0]),"        ", np.array(mat[0][1]), "          ", np.array(mat[0][2]), "        ", np.array(mat[0][3]), "           ", np.array(mat[0][4]), "           ", np.array(mat[0][5]), "          ", np.array(mat[0][6]), "        ", np.array(mat[0][7]), "        ", np.array(mat[0][8]), "     ", np.array(mat[0][9])) 
print('pred Apex Legends  ',"   ", np.array(mat[1][0]),"        ", np.array(mat[1][1]), "          ", np.array(mat[1][2]), "        ", np.array(mat[1][3]), "           ", np.array(mat[1][4]), "           ", np.array(mat[1][5]), "          ", np.array(mat[1][6]), "        ", np.array(mat[1][7]), "        ", np.array(mat[1][8]), "     ", np.array(mat[1][9]))
print('pred Fortnite      ',"   ", np.array(mat[2][0]),"        ", np.array(mat[2][1]), "          ", np.array(mat[2][2]), "        ", np.array(mat[2][3]), "           ", np.array(mat[2][4]), "           ", np.array(mat[2][5]), "          ", np.array(mat[2][6]), "        ", np.array(mat[2][7]), "        ", np.array(mat[2][8]), "     ", np.array(mat[2][9]))
print('pred Forza Horizon ',"   ", np.array(mat[3][0]),"        ", np.array(mat[3][1]), "          ", np.array(mat[3][2]), "        ", np.array(mat[3][3]), "           ", np.array(mat[3][4]), "           ", np.array(mat[3][5]), "          ", np.array(mat[3][6]), "        ", np.array(mat[3][7]), "        ", np.array(mat[3][8]), "     ", np.array(mat[3][9]))
print('pred Free Fire     ',"   ", np.array(mat[4][0]),"        ", np.array(mat[4][1]), "          ", np.array(mat[4][2]), "        ", np.array(mat[4][3]), "           ", np.array(mat[4][4]), "           ", np.array(mat[4][5]), "          ", np.array(mat[4][6]), "        ", np.array(mat[4][7]), "        ", np.array(mat[4][8]), "     ", np.array(mat[4][9]))
print('pred Genshin Impact',"   ", np.array(mat[5][0]),"        ", np.array(mat[5][1]), "          ", np.array(mat[5][2]), "        ", np.array(mat[5][3]), "           ", np.array(mat[5][4]), "           ", np.array(mat[5][5]), "          ", np.array(mat[5][6]), "        ", np.array(mat[5][7]), "        ", np.array(mat[5][8]), "     ", np.array(mat[5][9]))
print('pred God of War    ',"   ", np.array(mat[6][0]),"        ", np.array(mat[6][1]), "          ", np.array(mat[6][2]), "        ", np.array(mat[6][3]), "           ", np.array(mat[6][4]), "           ", np.array(mat[6][5]), "          ", np.array(mat[6][6]), "        ", np.array(mat[6][7]), "        ", np.array(mat[6][8]), "     ", np.array(mat[6][9]))
print('pred Minecraft     ',"   ", np.array(mat[7][0]),"        ", np.array(mat[7][1]), "          ", np.array(mat[7][2]), "        ", np.array(mat[7][3]), "           ", np.array(mat[7][4]), "           ", np.array(mat[7][5]), "          ", np.array(mat[7][6]), "        ", np.array(mat[7][7]), "        ", np.array(mat[7][8]), "     ", np.array(mat[7][9]))
print('pred Roblox        ',"   ", np.array(mat[8][0]),"        ", np.array(mat[8][1]), "          ", np.array(mat[8][2]), "        ", np.array(mat[8][3]), "           ", np.array(mat[8][4]), "           ", np.array(mat[8][5]), "          ", np.array(mat[8][6]), "        ", np.array(mat[8][7]), "        ", np.array(mat[8][8]), "     ", np.array(mat[8][9]))
print('pred Terraria      ',"   ", np.array(mat[9][0]),"        ", np.array(mat[9][1]), "          ", np.array(mat[9][2]), "        ", np.array(mat[9][3]), "           ", np.array(mat[9][4]), "           ", np.array(mat[9][5]), "          ", np.array(mat[9][6]), "        ", np.array(mat[9][7]), "        ", np.array(mat[9][8]), "     ", np.array(mat[9][9]))

