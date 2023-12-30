import tensorflow as tf 
import matplotlib.pyplot as plt
import numpy as np

'''Loading the fashion data from datatset'''
mnist = tf.keras.datasets.fashion_mnist


'''Creating the train and test data'''
(train_images,train_labels),(test_images,test_labels) = mnist.load_data()

'''Defining the class names'''

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


print('value for images',train_images.shape)

'''Presentation of image data in matplot library'''
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()
'''Creating the images by dividing the 255 which model can understand'''
train_images,test_images = train_images/255.0,test_images/255
'''Image presenation'''
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i])
    plt.xlabel(class_names[train_labels[i]])
plt.show()
'''Tensorflow Model creation'''
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(28,activation='relu'),
    tf.keras.layers.Dense(10,activation='softmax')
])
'''Model compile with loss and optimizer'''
model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),optimizer='adam',metrics=['accuracy'])

model.fit(train_images,train_labels,epochs=10)
'''Evaluate model'''
test_loss,test_acc = model.evaluate(test_images,test_labels)

print('test accuarcy',test_loss)
print('test loss',test_acc)
'''Prediction of test data'''
prediction_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
predict = prediction_model.predict(test_images)

print('Raw predicted value',predict[0])

print('Predicted value',np.argmax(predict[0]))

print('Test value',test_labels[0])

'''-----------------------------------------------------------------------------------------------'''
'''Visual presenation'''
def plot_image(i, predictions_array, true_label, img):
  true_label, img = true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  true_label = true_label[i]
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')



# Plot the first X test images, their predicted labels, and the true labels.
# Color correct predictions in blue and incorrect predictions in red.
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predict[i], test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predict[i], test_labels)
plt.tight_layout()
plt.show()