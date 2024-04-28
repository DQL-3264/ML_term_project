import numpy as np
from utils import load_data,show_train_history,show_prediction

from sklearn.metrics import classification_report,accuracy_score

from keras.models import Sequential 
from keras.callbacks import ModelCheckpoint
from keras.models import load_model 
from keras.utils import to_categorical
from keras.layers import Dense,Flatten,Dropout,BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adamax

# load dataset
x_train,y_train,x_test,y_test=load_data()
x_train.astype(np.float32)
x_test.astype(np.float32)

# dataset normlization
x_train_norm=x_train/255
x_test_norm=x_test/255

# Change to one hoe encoding
y_train_onehot =to_categorical(y_train,num_classes=2)
y_test_onehot=to_categorical(y_test,num_classes=2)

# Model's architecture
model=Sequential(name="CNN_ver1")

model.add(Conv2D(32,7, activation="relu", padding="same" ,input_shape=(128,128,1)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())

model.add(Conv2D(32,5, activation="relu", padding="same"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))
model.add(BatchNormalization())

model.add(Conv2D(64,5, activation="relu", padding="same"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())

model.add(Conv2D(64,5, activation="relu", padding="same"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))
model.add(BatchNormalization())

model.add(Conv2D(64,3, activation="relu", padding="same"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))
model.add(BatchNormalization())

model.add(Conv2D(64,3, activation="relu", padding="same"))
model.add(Dropout(0.5))
model.add(BatchNormalization())

model.add(Conv2D(128,3, activation="relu", padding="same"))

model.add(Flatten())
model.add(Dense(512, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(512, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(2, activation="softmax"))
print(model.summary())

model.compile(loss='binary_crossentropy',metrics=['accuracy'],optimizer='adam')

# call back function
model_checkpoint_callback =ModelCheckpoint(
    filepath="bestmodel.h5",
    monitor='val_accuracy',
    mode='max',
    save_best_only=True,
    save_weights_only=True,
    verbose=1
    )

# Train model
history=model.fit(x_train_norm,y_train_onehot,
                  epochs=50,
                  validation_split=0.2,
                  batch_size=10)

# show the training history
show_train_history(history)

# Predict testset
model.load_weights("bestmodel.h5")
pred=model.predict(x_test_norm)
pred=np.argmax(pred,axis=1)
print(classification_report(y_test, pred))
print(accuracy_score(y_test, pred))

# Predict ours pictures
model.load_weights("bestmodel.h5")
pred=np.load("photos.npy")
pred.astype(np.float32)
pred=pred/255
res=model.predict(pred)
res=np.argmax(res,axis=1)
show_prediction(pred,res)