import numpy as np
import keras
#from rnn_utils import *
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.utils import np_utils


from utils import get_data

T = 100 #number of timesteps taken in account for X and Y samples for the unconditionally



# WARNING : it can take several hours to run.
def get_X_Y_unconditionally():
    strokes, texts = get_data()  # We will consider the strokes only, and not the text
    # X is composed of sequences of T points (from a timestep t).
    # Y is composed of sequences of T points (from a timestep T + 1 relatively to X)
    # a point is composed of (b, x1, x2) with b = {0,1} if the pen is on contact with the paper or not, (x1, x2) the coordinate of the pen.
    # a stroke correspond to a text example. We can extract several samples for X and Y for each stroke.
    #(nb_stroke,_ ) = strokes.shape
    X = np.array([])
    Y = np.array([])
    for idx_stroke in range(0,550): #normally it is "in range(nb_stroke = 6000)". But it takes me too long.
        stroke = strokes[idx_stroke]
        (nb_timesteps, tmp) = stroke.shape
        
        for t in range(nb_timesteps - T - 1):
            X = np.append(X, [stroke[t : t + T]])
            Y = np.append(Y, [stroke[t + T + 1]])
        
    X = np.reshape(X, [-1, T, 3])
    Y = np.reshape(Y, [-1, 3])
    
    ## if you want to save them into a file (as it takes some times to generate...), uncomment these lines:
    #np.save("inputX_550", X)
    #np.save("inputY_550", Y)
    return X,Y

# return the model I used, before training.
def rebuild_model():
    hidden_size = 300
    
    model = Sequential()
    model.add(LSTM(hidden_size, return_sequences=True, input_shape=(T, 3)))
    model.add(LSTM(hidden_size, return_sequences=True))
    model.add(LSTM(hidden_size, return_sequences=True))
    model.add(LSTM(hidden_size, return_sequences=False))
    model.add(Dense(3 , activation='linear'))
    model.compile(loss = 'mean_squared_error', optimizer='rmsprop', metrics = ['accuracy'])
    print(model.summary())
    return model


def unconditionally_model():
    
    import keras
    ## If you want to rebuild the model uncomment the following line:
    # model = rebuild_model
    
    ## If you want to load a pretrained model uncomment the following line: 
    model = keras.models.load_model("../models/model2.hdf5")
    
    ## If you want to train more the model, uncomment the following lines:
    ## - if you want to preprocess the strokes data, you can uncomment this line. But it last a lot of time for me
    # X , Y = get_X_Y_unconditionally()  # get the X and Y samples used for training.
    ## - Or you can load the X and Y data that I saved. I correspond to the 550 first strokes.
    #X = np.load("..\data\inputX_550.npy")
    #Y = np.load("..\data\inputY_550_ones.npy")
    ## training :
    # history = model.fit(X, Y, batch_size = 32, nb_epoch = 3,  verbose = 1)
    
    ## if you want to save the model into a datafile, uncomment this line :
    # model.save("..\models\model.hdf5")
    
    return model



def predictSeq(cpt, model):
    X_ = np.array([1.,0.,0.] * 100)
    X_ = np.reshape(X_, [T, 3])
    X_ = np.array([X_])
    res = []
    
    for tmp in range(cpt):
        y = model.predict(X_)
        y[0][0] = int(y[0][0])
        res = np.append(res, y[0])
        X_ = np.reshape(np.append(X_[0][1 : T], y[0]), (-1, T, 3))
    res = np.reshape(res, [-1, 3])
    res[cpt-1][0] = 1 # we release the pen at the end. Otherwise we can't display the result
    return res


def generate_unconditionally(random_seed=1):
    # Input:
    #   random_seed - integer

    # Output:
    #   stroke - numpy 2D-array (T x 3)
    np.random.seed(18)
    
    # The model takes a sequance of T points and predict the next point.
    # We will give an empty sequence {[1, 0, 0]}*T 
    # Then we will predict (point by point) the 500 first points
    model = unconditionally_model()
    stroke_result = predictSeq(500, model)
    
    return stroke_result


def generate_conditionally(text='welcome to lyrebird', random_seed=1):
    # Input:
    #   text - str
    #   random_seed - integer

    # Output:
    #   stroke - numpy 2D-array (T x 3)
    return stroke


def recognize_stroke(stroke):
    # Input:
    #   stroke - numpy 2D-array (T x 3)

    # Output:
    #   text - str
    return 'welcome to lyrebird'