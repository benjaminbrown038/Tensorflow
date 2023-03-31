import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adadelta
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

x_train,y_train),x_test,y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0],1,28,28)
x_test = x_test.reshape(x_test.shape[0],1,28,28)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= x_train
x_test /= y_test
y_train = to_categorical(y_train,10)
y_test = to_categorical(y_test,10)

model = Sequential()
model.add(Conv2D(8,(3,3),padding='same',activation = 'relu',input_shape=(1,28,28)))
model.add(MaxPool2D((2,2),padding = 'same'))
model.add(Conv2D(64,(3,3),padding='same',activation='relu'))
model.add(MaxPool2D((4,4),padding='same'))
model.add(Conv2D(10,(3,3),padding='same',activation = 'relu'))
model.add(MaxPool2D((2,2),padding='same'))
model.add(Conv2D(10,(3,3),padding='same',activation = 'softmax'))
model.add(MaxPool2D((4,4),padding = 'same'))
model.add(Flatten())

# optimizers
ada = Adadelta(lr = .01)
adam = Adam()
sgd1 = SGD(lr = .01)
sgd2 = SGD(lr = .001)

# loss
loss = CategoricalCrossentropy()

# metrics
ac = Accuracy()

model.compile(loss = loss, optimizer = opt, accuracy = ac)

test_batch_size = len(y_test)
train_batch_size = len(y_train)
batch_size = test_batch_size + train_batch_size

model.fit(x_train,y_train,
          validation_data = (x_test,y_test),
          epochs = 25,
          batch_size = batch_size)


model.load_weights("model.h5")

#reshaping input for model
img = np.expand_dims(img,axis=0)
img_class = model.predict_classes(img)

# testing
prediction = img_class[0]
classname = img_class[0]

# predictions
print("Class: ",classname)
model.save('./model.h5')

deploy_model = load_model('./model.h5',compile=True)
