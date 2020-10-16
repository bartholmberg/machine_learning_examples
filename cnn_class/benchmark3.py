import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import os
import numpy as np
@tf.function
def comp(a,b,c):
    d= a*b+c
    e=a*b*c
    return d,e

tf.enable_eager_execution();
digitsIn = keras.Input(shape=(784,), name="digits")
x = layers.Dense(64, activation="relu", name="dense_1")(digitsIn)
x = layers.Dense(64, activation="relu", name="dense_2")(x)
pOut = layers.Dense(1, activation="linear", name="pOut")(x)
# 1 output here , so can't use SparseCategoricalCrossentropy for loss which has probability vector output
# use 10 (or more) for probabilty vector (of labeled outputs)
model = keras.Model(inputs=digitsIn, outputs=pOut)

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Preprocess the data (these are NumPy arrays)
x_train = x_train.reshape(60000, 784).astype("float32") / 255
x_test = x_test.reshape(10000, 784).astype("float32") / 255

y_train = y_train.astype("float32")
y_test = y_test.astype("float32")

# Reserve 10,000 samples for validation
x_val = x_train[-10000:]   # [start:stop:step], can omit any, so [start:], take last 10000
y_val = y_train[-10000:]   # [-10000:], take last 10000
#x_val2 = x_train[10000:]   # [10000:], take all but first 10000
#x_val2= x_train[:10000]    # [:10000],take first 10k
#y_val2= y_train[:10000]    # [:10000], take first 10k
x_train = x_train[:-10000] # [:-10000]  , take all but last 10000
y_train = y_train[:-10000]
A,B=tf.constant(3.0), tf.constant(6.0)
X= tf.Variable(20.0)
loss = tf.math.abs(A*X-B)
def train_step():
    with tf.GradientTape() as tape:
        loss=tf.math.abs(A*X-B)
    dX=tape.gradient(loss,X)
    print('X = {:0.2f}  dX = {:2f} loss = {:2f}'.format(X.numpy() ,dX,loss))
    X.assign(X-dX)
    #X= X-dX
print( )
model.summary()
for i in range(10):
    train_step()
a=tf.constant([1,2,3],'int8')

b=tf.constant([0,1,3],'int8')
c=tf.add(a,b)
d=comp(a,b,c)
print('c,d:',c,d)
opt=tf.keras.optimizers.SGD(lr=0.01,momentum=0.9)
model.compile(loss='mse',optimizer=opt)
#model.compile(
#    optimizer=keras.optimizers.RMSprop(),
#    loss='mse',
#    metrics=[tf.keras.metrics.RootMeanSquaredError()])
#model.compile(
   # optimizer=keras.optimizers.RMSprop(),  # Optimizer
    # Loss function to minimize
    #loss=keras.losses.SparseCategoricalCrossentropy(),
    #loss=tf.keras.losses.Loss(),
    # List of metrics to monitor
    #metrics=[keras.metrics.SparseCategoricalAccuracy()],
  #  metrics=[tf.keras.metrics.RootMeanSquaredError()],
#)

print("Fit model on training data")
if not os.path.isfile("bench3.index"): 
    history = model.fit(
      x_train,
      y_train,
      batch_size=64,
      epochs=90,
    # We pass some validation for
    # monitoring validation loss and metrics
    # at the end of each epoch
      validation_data=(x_val, y_val),
    )
else :
    model.load_weights("bench3") 
    history = model.fit(
      x_train,
      y_train,
      batch_size=64,
      epochs=1,
    # We pass some validation for
    # monitoring validation loss and metrics
    # at the end of each epoch
      validation_data=(x_val, y_val),
    )
 # Evaluate the model on the test data using `evaluate`
print("Evaluate on test data")
model.summary()
model.save_weights("bench3")
results = model.evaluate(x_test, y_test, batch_size=128)
for k in range(20):
    ind = np.random.random_integers(0,10000)
    yhat = model.predict(x_test[ind,:].reshape(1,784))
    zhat = max(np.round( yhat))
    print ("prediction at rand ind, pred, label: ",ind,zhat , y_test[ind])
 
    XtestImage = np.squeeze(x_test[ind,:]).reshape(28,28,1)
    plt.imshow(XtestImage)
    plt.show(block=False)
    plt.draw()
    plt.pause(1.5)
    plt.cla()

print("test loss, test acc:", results)

# Generate predictions (probabilities -- the output of the last layer)
# on new data using `predict`
print("Generate predictions for 3 samples")
predictions = model.predict(x_test[:3])
print("predictions shape:", predictions.shape)