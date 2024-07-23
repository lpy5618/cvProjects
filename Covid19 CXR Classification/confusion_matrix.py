import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.xception import preprocess_input
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# 设置模型路径和数据路径
model_path = 'Covid19 CXR Classification/covid.h5'
data_path = 'Covid19 CXR Classification/COVID-19_Radiography_Dataset'
classes = os.listdir(data_path)

# 加载模型
model = tf.keras.models.load_model(model_path)
print("Model loaded.")

# 获取验证数据生成器
datagen = image.ImageDataGenerator(preprocessing_function=preprocess_input, validation_split=0.2)
validation_generator = datagen.flow_from_directory(
    data_path,
    target_size=(299, 299),
    batch_size=32,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# 获取验证数据和标签
validation_generator.reset()
y_pred = model.predict(validation_generator, steps=validation_generator.samples // validation_generator.batch_size + 1, verbose=1)
y_pred_classes = np.argmax(y_pred, axis=1)

# 获取真实标签
y_true = validation_generator.classes[:len(y_pred_classes)]

# 计算混淆矩阵
cm = confusion_matrix(y_true, y_pred_classes)
cmd = ConfusionMatrixDisplay(cm, display_labels=classes)

# 可视化混淆矩阵
plt.figure(figsize=(10, 10))
cmd.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
