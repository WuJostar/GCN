#察看模型內的參數值

import os
from tensorflow.python import pywrap_tensorflow

model_dir = '/home/fan/test2/實驗12 老師新構想 batch_size 16/model.ckpt'

reader = pywrap_tensorflow.NewCheckpointReader(model_dir)
var_to_shape_map = reader.get_variable_to_shape_map()

for key in var_to_shape_map:
    if 'connected' in key: #使用'關鍵字'找
        print("tensor_name: ", key)
        print(reader.get_tensor(key).shape)
