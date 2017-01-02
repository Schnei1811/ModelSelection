import os
import tensorflow as tf
import numpy as np
import pickle
import time
from tabulate import tabulate
from sklearn.model_selection import train_test_split
from sklearn import neighbors, tree, svm, metrics
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from tensorflow.python.ops import rnn, rnn_cell
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse=False)
snnepochcount = 0


if not os.path.exists("./data"):
    print("\nA data folder has been created in this directory.\nPlease place a supervised learning dataset in this folder. All outputs must be in the far right columns.\n")
    os.makedirs("./data")

def TrainKNearestNeighbor(train_x, test_x, train_y, test_y):
    starttime = time.time()
    print('\nTraining...\n')
    knnclf = neighbors.KNeighborsClassifier(n_jobs=-1)
    knnclf.fit(train_x, train_y)
    trainaccuracy = knnclf.score(train_x, train_y)
    testaccuracy = knnclf.score(test_x, test_y)
    f1 = metrics.f1_score(test_y, knnclf.predict(test_x), average='weighted')
    runtimeseconds = round((time.time() - starttime), 2)
    modelname = 'KNN'
    modelparam = np.array([testaccuracy, f1, runtimeseconds])
    ans = Save(knnclf, trainaccuracy, testaccuracy, f1, modelname, knnprevaccuracy, knnprevf1score, runtimeseconds, modelparam)
    return

def TrainRandomForest(train_x, test_x, train_y, test_y):
    starttime = time.time()
    print('\nTraining...\n')
    rfclf = tree.DecisionTreeClassifier()
    rfclf.fit(train_x, train_y)
    trainaccuracy = rfclf.score(train_x, train_y)
    testaccuracy = rfclf.score(test_x, test_y)
    f1 = metrics.f1_score(test_y, rfclf.predict(test_x), average='weighted')
    runtimeseconds = round((time.time() - starttime), 2)
    modelname = 'RF'
    modelparam = np.array([testaccuracy, f1, runtimeseconds])
    ans = Save(rfclf, trainaccuracy, testaccuracy, f1, modelname, rfprevaccuracy, rfprevf1score, runtimeseconds, modelparam)
    return

def TrainMultinomialNaiveBayes(train_x, test_x, train_y, test_y):
    starttime = time.time()
    mnbclf = MultinomialNB()
    print('\nTraining...\n')
    try: mnbclf.fit(train_x, train_y)
    except ValueError:
        print('Multinomial Naive Bayes does not work with negative feature values')
        RepeatedModelSelect()
    trainaccuracy = mnbclf.score(test_x, test_y)
    testaccuracy = mnbclf.score(train_x, train_y)
    f1 = metrics.f1_score(test_y, mnbclf.predict(test_x), average='weighted')
    runtimeseconds = round((time.time() - starttime), 2)
    modelname = 'MNB'
    modelparam = np.array([testaccuracy, f1, runtimeseconds])
    ans = Save(mnbclf,trainaccuracy,testaccuracy, f1, modelname, mnbprevaccuracy, mnbprevf1score, runtimeseconds, modelparam)
    return

def TrainAdaBoost(train_x, test_x, train_y, test_y):
    starttime = time.time()
    print('\nTraining...\n')
    adaclf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2), n_estimators=300)
    adaclf.fit(train_x, train_y)
    trainaccuracy = adaclf.score(train_x, train_y)
    testaccuracy = adaclf.score(test_x, test_y)
    f1 = metrics.f1_score(test_y, adaclf.predict(test_x), average='weighted')
    runtimeseconds = round((time.time() - starttime), 2)
    modelname = 'ADA'
    modelparam = np.array([testaccuracy, f1, runtimeseconds])
    ans = Save(adaclf, trainaccuracy, testaccuracy, f1, modelname, adaprevaccuracy, adaprevf1score, runtimeseconds, modelparam)
    return

def TrainSVM(train_x, test_x, train_y, test_y, C):
    starttime = time.time()
    print('\nTraining...\n')
    svmclf = svm.SVC(C=C, kernel='rbf', max_iter = -1)
    svmclf.fit(train_x, train_y)
    trainaccuracy = svmclf.score(train_x, train_y)
    testaccuracy = svmclf.score(test_x, test_y)
    f1 = metrics.f1_score(test_y, svmclf.predict(test_x), average='weighted')
    runtimeseconds = round((time.time() - starttime), 2)
    modelname = 'SVM'
    modelparam = np.array([testaccuracy, f1, C, runtimeseconds])
    ans = Save(svmclf, trainaccuracy, testaccuracy, f1, modelname, svmprevaccuracy, svmprevf1score, runtimeseconds, modelparam)
    return

def TrainLogisticRegression(train_x, test_x, train_y, test_y, C):
    starttime = time.time()
    print('\nTraining...\n')
    logitclf = LogisticRegression(C = C)
    logitclf.fit(train_x, train_y)
    trainaccuracy = logitclf.score(test_x, test_y)
    testaccuracy = logitclf.score(train_x, train_y)
    f1 = metrics.f1_score(test_y, logitclf.predict(test_x), average='weighted')
    runtimeseconds = round((time.time() - starttime), 2)
    modelname = 'LOGIT'
    modelparam = np.array([testaccuracy, f1, C, runtimeseconds])
    ans = Save(logitclf, trainaccuracy, testaccuracy, f1, modelname, logitprevaccuracy, logitprevf1score, runtimeseconds, modelparam)
    return

def TrainSimpleNeuralNetwork(train_x, test_x, train_y, test_y, epochs, hidden_size, regularization, optimizername):
    global snnepochcount
    starttime = time.time()
    train_x, test_x = np.matrix(train_x), np.matrix(test_x)
    train_y_onehot = np.matrix(encoder.fit_transform(train_y))
    params = (np.random.random(size=hidden_size * (n_features + 1) + n_classes * (hidden_size + 1)) - 0.5) * 0.25
    fmin = minimize(fun=backprop, x0=params, args=(n_features, hidden_size, n_classes, train_x, train_y_onehot, regularization, epochs),
                    method=optimizername, jac=True, options={'maxiter': epochs})
    theta1 = np.matrix(np.reshape(fmin.x[:hidden_size * (n_features + 1)], (hidden_size, (n_features + 1))))
    theta2 = np.matrix(np.reshape(fmin.x[hidden_size * (n_features + 1):], (n_classes, (hidden_size + 1))))

    a1, z2, a2, z3, h = forward_propagate(train_x, theta1, theta2)
    y_predtrain = np.array(np.argmax(h, axis=1) + 1)
    traincorrect = [1 if a == b else 0 for (a, b) in zip(y_predtrain, train_y)]
    trainaccuracy = (sum(map(int, traincorrect)) / len(traincorrect))

    a1, z2, a2, z3, h = forward_propagate(test_x, theta1, theta2)
    y_predtest = np.array(np.argmax(h, axis=1) + 1)
    testcorrect = [1 if a == b else 0 for (a, b) in zip(y_predtest, test_y)]
    testaccuracy = (sum(map(int, testcorrect)) / len(testcorrect))

    f1 = metrics.f1_score(test_y, y_predtest, average='weighted')
    snnepochcount = 0
    runtimeseconds = round((time.time() - starttime), 2)
    modelname = 'SNN'
    if optimizername in ['TNC']: optimizername = 'Newton'
    if optimizername in ['trust-ncg']: optimizername = 'Newton-TR'
    modelparam = np.array([testaccuracy, f1, epochs, hidden_size, regularization, runtimeseconds, snnoptimizerworddict[optimizername]])
    ans = Save(y_predtest, trainaccuracy, testaccuracy, f1, modelname, snnprevaccuracy, snnprevf1score, runtimeseconds, modelparam)
    if ans in ['y']:
        savepickle = open('modelparameters/{}/SNNpickle.pickle'.format(dataname), 'wb')
        pickle.dump(fmin, savepickle)
    return

def TrainDeepNeuralNetwork(train_x, test_x, train_y, test_y, epochs, hiddenlayersize1, hiddenlayersize2, optimizer, optimizername):
    starttime = time.time()
    train_y_onehot, test_y_onehot = np.matrix(encoder.fit_transform(train_y)), np.matrix(encoder.fit_transform(test_y))
    batch_size = 100

    x = tf.placeholder('float', [None, n_features])
    y = tf.placeholder('float')

    prediction = deep_neural_network_model(x, hiddenlayersize1, hiddenlayersize2, n_classes, n_features)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, y))
    optimizer = optimizer.minimize(cost)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        estimatedruntime, earlyexitcount, earlyexit = time.time(), 0, 0
        for epoch in range(epochs):
            epoch_loss, i = 0, 0
            while i < len(train_x):
                start = i
                end = i + batch_size
                batch_x = np.array(train_x[start:end])
                batch_y = np.array(train_y_onehot[start:end])
                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
                epoch_loss += c
                i += batch_size
            if epoch == 0:
                estimatedruntime, timemetric= TimeConversion((time.time() - estimatedruntime) * epochs)
                print('\nEstimated Model Running Time:', round(estimatedruntime, 2), timemetric, '\n')
            earlyexitcount, epoch, earlyexit = dnnexitcheck(earlyexitcount, epoch, epoch_loss, earlyexit)
            if epoch == -1: break
            print('Epoch', epoch + 1, 'completed out of', epochs, 'loss:', epoch_loss)
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        y_p = tf.argmax(prediction, 1)
        val_accuracy, y_pred = sess.run([accuracy, y_p], feed_dict={x: test_x, y: test_y_onehot})
        y_pred = y_pred+1
        trainaccuracy = accuracy.eval({x: train_x, y: train_y_onehot})
        testaccuracy = accuracy.eval({x: test_x, y: test_y_onehot})
        f1 = metrics.f1_score(test_y, y_pred, average='weighted')
        runtimeseconds = round((time.time() - starttime), 2)
        modelname = 'DNN'
        modelparam = [testaccuracy, f1, epochs, hiddenlayersize1, hiddenlayersize2, runtimeseconds, dnnoptimizerworddict[optimizername]]
        ans = Save(y_pred, trainaccuracy, testaccuracy, f1, modelname, dnnprevaccuracy, dnnprevf1score, runtimeseconds, modelparam)
        if ans in ["y"]:
            saver.save(sess, "modelparameters/{}/DNNmodel.ckpt".format(dataname))
    sess.close()
    tf.reset_default_graph()
    return

def TrainRecurrentNeuralNetwork(train_x, test_x, train_y, test_y, epochs, hiddenlayersize, optimizer, optimizername):
    starttime = time.time()
    train_y_onehot, test_y_onehot = np.matrix(encoder.fit_transform(train_y)), np.matrix(encoder.fit_transform(test_y))

    for i in range(256, 0, -1):
        if len(train_x[:]) % i == 0:
            batch_size = i
            break

    for i in range(round((n_features) ** 0.5), 1, -1):
        if (n_features) % i == 0:
            sequence_size = i
            n_sequence = int((n_features) / sequence_size)
            break

    print("Batch Size: ", batch_size, "\nNumber of Sequences: ", n_sequence, "\nSequence Size: ", sequence_size)

    x = tf.placeholder('float', [None, n_sequence, sequence_size])
    y = tf.placeholder('float')

    prediction = recurrent_neural_network(x, hiddenlayersize, n_sequence, sequence_size)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, y))
    optimizer = optimizer.minimize(cost)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        estimatedruntime, earlyexitcount, earlyexit = time.time(), 0, 0
        for epoch in range(epochs):
            epoch_loss, i = 0, 0
            while i < len(train_x):
                start = i
                end = i + batch_size
                batch_x = np.array(train_x[start:end])
                batch_x = batch_x.reshape((batch_size, n_sequence, sequence_size))
                batch_y = np.array(train_y_onehot[start:end])
                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
                epoch_loss += c
                i += batch_size
            if epoch == 0:
                estimatedruntime, timemetric= TimeConversion((time.time() - estimatedruntime) * epochs)
                print('\nEstimated Model Running Time:', round(estimatedruntime, 2), timemetric, '\n')
            earlyexitcount, epoch, earlyexit = dnnexitcheck(earlyexitcount, epoch, epoch_loss, earlyexit)
            if epoch == -1: break
            print('Epoch', epoch + 1, 'completed out of', epochs, 'loss:', epoch_loss)
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        y_p = tf.argmax(prediction, 1)
        val_accuracy, y_pred = sess.run([accuracy, y_p],feed_dict={x: test_x.reshape((-1, n_sequence, sequence_size)), y: test_y_onehot})
        y_pred = y_pred+1
        trainaccuracy = accuracy.eval({x: test_x.reshape((-1, n_sequence, sequence_size)), y: test_y_onehot})
        testaccuracy = accuracy.eval({x: test_x.reshape((-1, n_sequence, sequence_size)), y: test_y_onehot})
        f1 = metrics.f1_score(test_y, y_pred, average='weighted')
        runtimeseconds = round((time.time() - starttime), 2)
        modelname = 'RNN'
        modelparam = np.array([testaccuracy, f1, epochs, hiddenlayersize, n_sequence, sequence_size, runtimeseconds, dnnoptimizerworddict[optimizername]])
        ans = Save(y_pred, trainaccuracy, testaccuracy, f1, modelname, rnnprevaccuracy, rnnprevf1score, runtimeseconds, modelparam)
        if ans in ["y"]:
            saver.save(sess, "modelparameters/{}/RNNmodel.ckpt".format(dataname))
    sess.close()
    tf.reset_default_graph()
    return

def TrainBiDirectionalRecurrentNeuralNetwork(train_x, test_x, train_y, test_y, epochs, hiddenlayersize, optimizer, optimizername):
    starttime = time.time()
    train_y_onehot, test_y_onehot = np.matrix(encoder.fit_transform(train_y)), np.matrix(encoder.fit_transform(test_y))

    for i in range(256, 0, -1):
        if len(train_x[:]) % i == 0:
            batch_size = i
            break

    for i in range(round((n_features) ** 0.5), 1, -1):
        if (n_features) % i == 0:
            sequence_size = i
            n_sequence = int((n_features) / sequence_size)
            break

    print("Batch Size: ", batch_size, "\nNumber of Sequences: ", n_sequence, "\nSequence Size: ", sequence_size)

    x = tf.placeholder("float", [None, n_sequence, sequence_size])
    y = tf.placeholder("float", [None, n_classes])

    prediction = bidirectional_neural_network(x, sequence_size, n_sequence, hiddenlayersize)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, y))
    optimizer = optimizer.minimize(cost)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        estimatedruntime, earlyexitcount, earlyexit = time.time(), 0, 0
        for epoch in range(epochs):
            epoch_loss, i = 0, 0
            while i < len(train_x):
                start = i
                end = i + batch_size
                batch_x = np.array(train_x[start:end])
                batch_x = batch_x.reshape((batch_size, sequence_size, n_sequence))
                batch_y = np.array(train_y_onehot[start:end])
                batch_x = batch_x.reshape((batch_size, n_sequence, sequence_size))
                _, c = sess.run([optimizer,cost], feed_dict={x: batch_x, y: batch_y})
                epoch_loss += c
                i += batch_size
            if epoch == 0:
                estimatedruntime, timemetric= TimeConversion((time.time() - estimatedruntime) * epochs)
                print('\nEstimated Model Running Time:', round(estimatedruntime, 2), timemetric, '\n')
            earlyexitcount, epoch, earlyexit = dnnexitcheck(earlyexitcount, epoch, epoch_loss, earlyexit)
            if epoch == -1: break
            print('Epoch', epoch + 1, 'completed out of', epochs, 'loss:', epoch_loss)
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        y_p = tf.argmax(prediction, 1)
        val_accuracy, y_pred = sess.run([accuracy, y_p],feed_dict={x: test_x.reshape((-1, n_sequence, sequence_size)), y: test_y_onehot})
        y_pred = y_pred + 1
        trainaccuracy = accuracy.eval({x: test_x.reshape((-1, n_sequence, sequence_size)), y: test_y_onehot})
        testaccuracy = accuracy.eval({x: test_x.reshape((-1, n_sequence, sequence_size)), y: test_y_onehot})
        f1 = metrics.f1_score(test_y, y_pred, average='weighted')
        runtimeseconds = round((time.time() - starttime), 2)
        modelname = 'BiNN'
        modelparam = np.array([testaccuracy, f1, epochs, hiddenlayersize, n_sequence, sequence_size, runtimeseconds, dnnoptimizerworddict[optimizername]])
        ans = Save(y_pred, trainaccuracy, testaccuracy, f1, modelname, binnprevaccuracy, binnprevf1score, runtimeseconds, modelparam)
        if ans in ["y"]:
            saver.save(sess, "modelparameters/{}/BiNNmodel.ckpt".format(dataname))
    sess.close()
    tf.reset_default_graph()
    return

def TrainConvolutionalNeuralNetwork(train_x, test_x, train_y, test_y, epochs, optimizer, optimizername):
    starttime = time.time()
    train_y_onehot = np.matrix(encoder.fit_transform(train_y))
    test_y_onehot = np.matrix(encoder.fit_transform(test_y))
    convolution_size = int(n_features ** 0.5)
    dropout = 1.0

    for i in range(256, 0, -1):
        if len(train_x[:]) % i == 0:
            batch_size = i
            break

    x = tf.placeholder(tf.float32, [None, n_features])
    y = tf.placeholder(tf.float32, [None, n_classes])
    keep_prob = tf.placeholder(tf.float32)

    prediction = convolutional_neural_network(x, keep_prob)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, y))
    optimizer = optimizer.minimize(cost)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        estimatedruntime, earlyexitcount, earlyexit = time.time(), 0, 0
        for epoch in range(epochs):
            epoch_loss, i = 0, 0
            while i < len(train_x):
                start = i
                end = i + batch_size
                batch_x = np.array(train_x[start:end])
                batch_x = batch_x.reshape((batch_size, convolution_size, convolution_size))
                batch_y = np.array(train_y_onehot[start:end])
                batch_x = batch_x.reshape((batch_size, n_features))
                _, c = sess.run([optimizer,cost], feed_dict={x: batch_x, y: batch_y, keep_prob: dropout})
                epoch_loss += c
                i += batch_size
            if epoch == 0:
                estimatedruntime, timemetric = TimeConversion((time.time() - estimatedruntime) * epochs)
                print('\nEstimated Model Running Time:', round(estimatedruntime, 2), timemetric, '\n')
            earlyexitcount, epoch, earlyexit = dnnexitcheck(earlyexitcount, epoch, epoch_loss, earlyexit)
            if epoch == -1: break
            print('Epoch', epoch + 1, 'completed out of', epochs, 'loss:', epoch_loss)
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        y_p = tf.argmax(prediction, 1)
        val_accuracy, y_pred = sess.run([accuracy, y_p], feed_dict={x: test_x.reshape((-1, n_features)), y: test_y_onehot, keep_prob: dropout})
        y_pred = y_pred + 1
        trainaccuracy = accuracy.eval({x: test_x.reshape((-1, n_features)), y: test_y_onehot, keep_prob: dropout})
        testaccuracy = accuracy.eval({x: test_x.reshape((-1, n_features)), y: test_y_onehot, keep_prob: dropout})
        f1 = metrics.f1_score(test_y, y_pred, average='weighted')
        runtimeseconds = round((time.time() - starttime), 2)
        modelname = 'ConvNN'
        modelparam = np.array([testaccuracy, f1, epochs, convolution_size, runtimeseconds, dnnoptimizerworddict[optimizername]])
        ans = Save(y_pred, trainaccuracy, testaccuracy, f1, modelname, convnnprevaccuracy, convnnprevf1score, runtimeseconds, modelparam)
        if ans in ["y"]:
            saver.save(sess, "modelparameters/{}/ConvNNmodel.ckpt".format(dataname))
    sess.close()
    tf.reset_default_graph()
    return

def sigmoid(z):
    return expit(z)

def sigmoid_gradient(z):
    return np.multiply(sigmoid(z), (1 - sigmoid(z)))

def forward_propagate(X, theta1, theta2):
    m = X.shape[0]
    a1 = np.insert(X, 0, values=np.ones(m), axis=1)
    z2 = a1 * theta1.T
    a2 = np.insert(sigmoid(z2), 0, values=np.ones(m), axis=1)
    z3 = a2 * theta2.T
    h = sigmoid(z3)
    return a1, z2, a2, z3, h

def backprop(params, n_features, hidden_size, n_classes, X, y, learning_rate, epochs):
    global snnepochcount
    estimatedruntime = time.time()
    m = X.shape[0]
    J = 0
    X, y = np.matrix(X), np.matrix(y)
    theta1 = np.matrix(np.reshape(params[:hidden_size * (n_features + 1)], (hidden_size, (n_features + 1))))
    theta2 = np.matrix(np.reshape(params[hidden_size * (n_features + 1):], (n_classes, (hidden_size + 1))))
    a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)
    delta1, delta2 = np.zeros(theta1.shape), np.zeros(theta2.shape)
    for i in range(m):
        try: first_term = np.multiply(-y[i, :], np.log(h[i, :]))
        except ValueError: print("Check if you have classifications that are labelled 0. If so, +1 to Classifications and rerun")
        try: second_term = np.multiply((1 - y[i, :]), np.log(1 - h[i, :]))
        except ValueError: print("Check if you have input data without a labelled classifications")
        J += np.sum(first_term - second_term)
    J = J / m
    J += (float(learning_rate) / (2 * m)) * (np.sum(np.power(theta1[:, 1:], 2)) + np.sum(np.power(theta2[:, 1:], 2)))
    for t in range(m):
        a1t = a1[t, :]
        z2t = z2[t, :]
        a2t = a2[t, :]
        ht = h[t, :]
        yt = y[t, :]
        d3t = ht - yt
        z2t = np.insert(z2t, 0, values = np.ones(1))
        d2t = np.multiply((theta2.T * d3t.T).T, sigmoid_gradient(z2t))
        delta1 = delta1 + (d2t[:, 1:]).T * a1t
        delta2 = delta2 + d3t.T * a2t
    delta1, delta2 = delta1 / m, delta2 / m
    delta1[:, 1:] = delta1[:, 1:] + (theta1[:, 1:] * learning_rate) / m
    delta2[:, 1:] = delta2[:, 1:] + (theta2[:, 1:] * learning_rate) / m
    grad = np.concatenate((np.ravel(delta1), np.ravel(delta2)))
    if snnepochcount == 0:
        estimatedruntime, timemetric = TimeConversion((time.time() - estimatedruntime) * epochs)
        print('\nEstimated Model Running Time:', round(estimatedruntime,2), timemetric, '\n')
    snnepochcount = snnepochcount + 1
    print("Iteration:", snnepochcount, " Cost:", J)
    return J, grad

def deep_neural_network_model(x, hiddenlayersize1, hiddenlayersize2, n_classes, n_features):
    hidden_1_layer = {'f_fum': hiddenlayersize1,
                      'weight': tf.Variable(tf.random_normal([n_features, hiddenlayersize1])),
                      'bias': tf.Variable(tf.random_normal([hiddenlayersize1]))}
    hidden_2_layer = {'f_fum': hiddenlayersize2,
                      'weight': tf.Variable(tf.random_normal([hiddenlayersize1, hiddenlayersize2])),
                      'bias': tf.Variable(tf.random_normal([hiddenlayersize2]))}
    output_layer = {'f_fum': None,
                    'weight': tf.Variable(tf.random_normal([hiddenlayersize2, n_classes])),
                    'bias': tf.Variable(tf.random_normal([n_classes])), }
    l1 = tf.add(tf.matmul(x, hidden_1_layer['weight']), hidden_1_layer['bias'])
    l1 = tf.nn.relu(l1)
    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weight']), hidden_2_layer['bias'])
    l2 = tf.nn.relu(l2)
    output = tf.matmul(l2, output_layer['weight']) + output_layer['bias']
    return output

def recurrent_neural_network(x, hiddenlayersize, n_sequence, sequence_size):
    layer = {'weights': tf.Variable(tf.random_normal([hiddenlayersize, n_classes])),
             'biases': tf.Variable(tf.random_normal([n_classes]))}
    x = tf.transpose(x, [1, 0, 2])
    x = tf.reshape(x, [-1, sequence_size])
    x = tf.split(0, n_sequence, x)
    lstm_cell = rnn_cell.BasicLSTMCell(hiddenlayersize, state_is_tuple=True, forget_bias = 0.9)
    outputs, states = rnn.rnn(lstm_cell, x, dtype=tf.float32)
    output = tf.matmul(outputs[-1], layer['weights']) + layer['biases']
    return output

def bidirectional_neural_network(x, sequence_size, n_sequence, hiddenlayersize):
    layer = {'weights': tf.Variable(tf.random_normal([2*hiddenlayersize, n_classes])),
             'biases': tf.Variable(tf.random_normal([n_classes]))}
    x = tf.transpose(x, [1, 0, 2])
    x = tf.reshape(x, [-1, sequence_size])
    x = tf.split(0, n_sequence, x)
    lstm_fw_cell = rnn_cell.BasicLSTMCell(hiddenlayersize, state_is_tuple=True, forget_bias = 0.9)
    lstm_bw_cell = rnn_cell.BasicLSTMCell(hiddenlayersize, state_is_tuple=True, forget_bias = 0.9)
    outputs, _, _ = rnn.bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x, dtype=tf.float32)
    return tf.matmul(outputs[-1], layer['weights']) + layer['biases']

def conv2d(x, W, b, strides=1):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def maxpool2d(x, k):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

def convolutional_neural_network(x, dropout):
    weights = {'wc1': tf.Variable(tf.random_normal([5, 5, 1, 50])),
               'wc2': tf.Variable(tf.random_normal([5, 5, 50, 100])),
               'wd1': tf.Variable(tf.random_normal([5 * 5 * 100, 1024])),
               'out': tf.Variable(tf.random_normal([1024, n_classes]))}
    biases = {'bc1': tf.Variable(tf.random_normal([50])),
              'bc2': tf.Variable(tf.random_normal([100])),
              'bd1': tf.Variable(tf.random_normal([1024])),
              'out': tf.Variable(tf.random_normal([n_classes]))}
    x = tf.reshape(x, shape=[-1, 20, 20, 1])
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    conv1 = maxpool2d(conv1, k=2)
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    conv2 = maxpool2d(conv2, k=2)
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout(fc1, dropout)
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out

def isPrime(n):
    if n == 2: return True
    if n == 3: return True
    if n % 2 == 0: return False
    if n % 3 == 0: return False
    i = 5
    w = 2
    while i * i <= n:
        if n % i == 0: return False
        i += w
        w = 6 - w
    return True

def Save(clf, trainaccuracy, testaccuracy, f1score, modelname, prevaccuracy, prevf1score, runtimeseconds, modelparam):
    runtime, timemetric = TimeConversion(runtimeseconds)
    print('\n\nTime: {}{}'.format(round(runtime,2), timemetric))
    print('Testing Set Accuracy:', testaccuracy)
    print('Accuracy Change from Saved Model:', testaccuracy - prevaccuracy)
    print('\nF1 Score:', f1score)
    print('F1 Score Change from Saved Model:', f1score - prevf1score)
    if (testaccuracy - prevaccuracy) > 0 and (f1score - prevf1score) > 0:
        print('\nYour results have improved!')
    print('\nTraining Set Accuracy:', trainaccuracy)
    print('Training Set Accuracy - Testing Set Accuracy:', trainaccuracy - testaccuracy)
    if modelname in ['SVM', 'SNN', 'DNN', 'RNN']:
        if trainaccuracy < 0.9:
            print('\nModel suffering from high bias. Consider decreasing regularization, a polynomial expansion dataset or getting more features')
        if trainaccuracy >= 0.9 and testaccuracy < 0.9:
            print('\nModel suffering from high variance. Consider increasing regularization, getting more data, or reducing features')
    if modelname in ['KNN','RF','MNB','SVM','LOGIT','ADA']:
        print('\nConfusion Matrix:\n', metrics.confusion_matrix(test_y, clf.predict(test_x)))
    elif modelname in['SNN', 'DNN', 'RNN', 'BiNN', 'ConvNN']:
        print('\nConfusion Martrix:\n', metrics.confusion_matrix(test_y, clf))
    ans = input("\nWould you like to save the new parameters? (y/n): ")
    if ans in ["y"]:
        if (testaccuracy - prevaccuracy) < 0 and (f1score - prevf1score) < 0:
            ans = input("\nModel performed worse than saved model. Are you sure you want to save? (y/n): ")
            if ans in ['n']:
                print('Model Unsaved\n')
                return ans
        np.savetxt('modelparameters/{}/{}modelaccuracy.txt'.format(dataname, modelname), modelparam, delimiter=',')
        if modelname in ['KNN','RF','MNB','SVM','LOGIT','ADA']:
            savepickle = open('modelparameters/{}/{}pickle.pickle'.format(dataname, modelname), 'wb')
            pickle.dump(clf, savepickle)
        print("Model Saved")
    elif ans in ['n']: print("Model Unsaved")
    else:
        print("Improper Command")
        Save(clf, trainaccuracy, testaccuracy, f1score, modelname, prevaccuracy, prevf1score, runtimeseconds)
    return ans

def dnnexitcheck(j, epoch, epoch_loss, earlyexit):
    i = 100
    if epoch % i == 0:
        earlyexit = epoch_loss
        j = epoch
    if j + (i-1) == epoch:
        if round(earlyexit - epoch_loss, 3) == 0: epoch = -1
    return j, epoch, earlyexit

def TimeConversion(runtimeseconds):
    timemetric = "s"
    if runtimeseconds >= 60 and runtimeseconds < 3600:
        totaltimer = runtimeseconds / 60
        timemetric = "m"
    elif runtimeseconds >= 3600:
        totaltimer = runtimeseconds / 3600
        timemetric = "h"
    else: totaltimer = runtimeseconds
    return totaltimer, timemetric

def dnn_parameters(ans):
    optimizername = str(input('\nUse which optimizer? (GradDescent, AdaDelta, AdaGrad, Momentum, Adam, Ftrl, ProxGD, ProxAda, RMSProp, or help, back, exit) '))
    if optimizername in ['help']:
        print('\nGradient Descent: Iteratively take the derivative of a point and move by learning rate to local minimum.\n'
              'AdaGrad: Adapts learning rate to parameters with larger rates for infrequent and smaller for frequent. Learning rate can become infintely small.\n'
              'AdaDelta: AdaDelta operates similar to Ada Grad but restricts the learning rate to a fixed size.\n'
              'Momentum: Uses the exponential decaying average of past gradients to accelerate SGD in the relevant direction. Gain faster convergence with reduced oscillation.\n'
              'Adaptive Moment Estimation (Adam): Combination of AdaDelta and Momentum.\n'
              'Follow-the-Regularized-Leader: When regularization is needed, outperforms.\n'
              'Proximal Gradient: Gradient Descent where regularization is solved for.\n'
              'Proximal AdaGrad: AdaGrad where regularization is solved for.\n'
              'RMS Prop: Divide gradient by a running average of its recent magnitude.\n')
        dnn_parameters(ans)
        return
    elif optimizername in ['back']: ModelSelect()
    elif optimizername in ['exit']: exit()
    elif optimizername in ['GradDescent']: optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.001)
    elif optimizername in ['AdaDelta']: optimizer = tf.train.AdadeltaOptimizer(learning_rate = 0.001)
    elif optimizername in ['AdaGrad']: optimizer = tf.train.AdagradOptimizer(learning_rate = 0.001)
    elif optimizername in ['Momentum']: optimizer = tf.train.MomentumOptimizer(learning_rate = 0.001, momentum = 1.0)
    elif optimizername in ['Adam']: optimizer = tf.train.AdamOptimizer(learning_rate = 0.001)
    elif optimizername in ['Ftrl']: optimizer = tf.train.FtrlOptimizer(learning_rate = 0.001)
    elif optimizername in ['ProxGD']: optimizer = tf.train.ProximalGradientDescentOptimizer(learning_rate = 0.001)
    elif optimizername in ['ProxAda']: optimizer = tf.train.ProximalAdagradOptimizer(learning_rate = 0.001)
    elif optimizername in ['RMSProp']: optimizer = tf.train.RMSPropOptimizer(learning_rate = 0.001)
    else:
        print('Unrecognizable command')
        ModelSelect()
        return
    try: epochs = int(input('How many epochs? '))
    except ValueError:
        print('Integer not entered')
        ModelSelect()
        return
    if ans in ['convnn']: return optimizer, optimizername, epochs
    elif ans in ['dnn']:
        try: hiddenlayersize1 = int(input(('Hidden Layer Size of First Layer? (Recommended at least ' + str(2 * n_features) + ') ')))
        except ValueError: print('Integer not entered'), ModelSelect()
        try: hiddenlayersize2 = int(input(('Hidden Layer Size of Second Layer? (Recommended at least ' + str(2 * n_features) + ') ')))
        except ValueError: print('Integer not entered'), ModelSelect()
        print(optimizer, optimizername, epochs, hiddenlayersize1, hiddenlayersize2)
        return optimizer, optimizername, epochs, hiddenlayersize1, hiddenlayersize2
    else:
        try: hiddenlayersize = int(input(('Hidden Layer Size? (Recommended at least ' + str(2 * n_features) + ') ')))
        except ValueError: print('Integer not entered'), ModelSelect()
        return optimizer, optimizername, epochs, hiddenlayersize

def ModelSelect():
    ans = input('\nTrain which model? (knn, rf, mnb, ada, svm, logit, snn, dnn, rnn, binn, convnn, or help, exit): ')
    if ans in ['knn']: TrainKNearestNeighbor(train_x, test_x, train_y, test_y)
    elif ans in ['rf']: TrainRandomForest(train_x, test_x, train_y, test_y)
    elif ans in ['mnb']: TrainMultinomialNaiveBayes(train_x, test_x, train_y, test_y)
    elif ans in ['ada']: TrainAdaBoost(train_x, test_x, train_y, test_y)
    elif ans in ['svm']:
        C = float(input('Enter Value for C (1 default):'))
        TrainSVM(train_x, test_x, train_y, test_y, C)
    elif ans in ['logit']:
        C = float(input('Enter Value for C (1 default):'))
        TrainLogisticRegression(train_x, test_x, train_y, test_y, C)
    elif ans in ['snn']:
        optimizername = str(input('\nUse which optimizer? (Newton, Newton-CG, BFGS, L-BFGS-B, Nelder-Mead, Powell, COBYLA, SLSQP, or help, back, exit) '))
        if optimizername in ['help']:
            print('\nTruncated Newton: Iteratively solving for the minimum of the second derivative of an inverse Hessian matrix\n'
                  'Truncated Newton Conjugate Gradient: Newton method where the conjugate direction is determined for each step\n'
                  'Broyden-Fletcher-Goldfarb-Shanno (Quasi-Newton): Approximates Newton\'s Method using the secant method to solve for first order roots.\n'
                  'Limited-memory BFGS Bounded: BFGS storing only necessary vectors and bounded by lower and upper bound of each feature.\n'
                  'Nelder-Mead: Utilizes a simplex structure rolling to minimum point with the ability to shrink, expand, or reflect given the most recent step\n'
                  'Powell: Bidirectional vector search refining vectors by combining bisection and secant methods. Strong when no mathematical relationship (No derivatives)\n'
                  'Constrained Optimization by Linear Approximation: Option when derivative unsolvable. Iteratively solving for problem with linear programming problems\n'
                  'Sequential Least Squares Quadratic Programming: Uses least squares to solve for minimum convex function\n')
            ModelSelect()
            return
        elif optimizername in ['back']:
            ModelSelect()
            return
        elif optimizername in ['exit']: exit()
        elif optimizername in ['Newton']: optimizername = 'TNC'
        elif optimizername in ['Newton-CG','BFGS','L-BFGS-B','Nelder-Mead','Powell','COBYLA','SLSQP']: optimizername = optimizername
        else:
            print('Unrecognizable command')
            ModelSelect()
            return
        if optimizername in ['Newton-CG','L-BFGS-B','Nelder-Mead','Powell','SLSQP']:
            ans = input('\nWarning: Selected optimizer must run to completion. Running time unknown. Continue? (y/n) ')
            if ans in ['y']: epochs = 1
            elif ans in ['n']:
                print('Newton, BFGS, and COBYLA are the snn optimzers with set epochs')
                ModelSelect()
                return
            else:
                print('Unrecognizable Command')
                ModelSelect()
                return
        else:
            try: epochs = int(input('How many epochs? '))
            except ValueError:
                print('Integer not entered')
                ModelSelect()
                return
        try: hiddenlayersize = int(input('Hidden Layer Size? (Recommended at least ' + str(2 * n_features) + ') '))
        except ValueError:
            print('Integer not entered')
            ModelSelect()
            return
        try: regularization = float(input('Enter Value for Regularization (1 default) '))
        except ValueError:
            print('Float not entered')
            ModelSelect()
            return
        TrainSimpleNeuralNetwork(train_x, test_x, nntrain_y, test_y, epochs, hiddenlayersize, regularization, optimizername)
    elif ans in ['dnn']:
        optimizer, optimizername, epochs, hiddenlayersize1, hiddenlayersize2 = dnn_parameters(ans)
        TrainDeepNeuralNetwork(train_x, test_x, nntrain_y, test_y, epochs, hiddenlayersize1, hiddenlayersize2, optimizer, optimizername)
    elif ans in ['rnn']:
        optimizer, optimizername, epochs, hiddenlayersize = dnn_parameters(ans)
        TrainRecurrentNeuralNetwork(train_x, test_x, nntrain_y, test_y, epochs, hiddenlayersize, optimizer, optimizername)
    elif ans in ['binn']:
        optimizer, optimizername, epochs, hiddenlayersize = dnn_parameters(ans)
        TrainBiDirectionalRecurrentNeuralNetwork(train_x, test_x, nntrain_y, test_y, epochs, hiddenlayersize, optimizer, optimizername)
    elif ans in ['convnn']:
        if (n_features ** 0.5).is_integer() == False:
            print('For Convolutional Neural Networks Num Features must have an Integer Square Root')
            ModelSelect()
            return
        optimizer, optimizername, epochs = dnn_parameters(ans)
        TrainConvolutionalNeuralNetwork(train_x, test_x, nntrain_y, test_y, epochs, optimizer, optimizername)
    elif ans in ['help']:
        print('\nModel Descriptions:\nknn:\tK-Nearest Neighbour\nrf: \tRandom Forest\nmnb:\tMultinomial Naive Bayes\nada:\tAda Boost Classifier\nsvm:\tRadial Basis Function Support Vector Machine\n'
              'logit:\tLogistic Regression\nsnn:\tSimple Neural Network\ndnn:\tDeep Neural Network (Multilayer Percepetron)\nrnn:\tLong Short-Term Memory Recurrent Neural Network'
              '\nbinn:\tBiDirectional Long-Short Term Memory Recurrent Neural Network\nconvnn:\tConvolutional Neural Network')
        ModelSelect()
        return
    elif ans in ['exit']: exit()
    else:
        print('Improper Selection')
        ModelSelect()
        return

def RepeatedModelSelect():
    while True:
        ans = input("\nTrain Data with Alternative Model? (y/n): ")
        if ans in ["y"]: ModelSelect()
        else: exit()

def LoadData():
    dirs = os.listdir("data/")
    print("\nAvailable Datasets: \n")
    for file in dirs: print(file)
    datarun = input("\nEnter a Dataset: ")
    return datarun

def CheckDataName():
    while True:
        datarun = LoadData()
        dataname = datarun[:-4]
        try:
            DataSet = np.loadtxt('data/{}'.format(datarun), delimiter=",")
            return DataSet, dataname
            break
        except FileNotFoundError:
            try:
                DataSet, dataname = np.loadtxt('data/{}.txt'.format(datarun), delimiter=","), datarun
                return DataSet, dataname
                break
            except FileNotFoundError:
                try:
                    DataSet, dataname = np.loadtxt('data/{}.csv'.format(datarun), delimiter=","), datarun
                    return DataSet, dataname
                    break
                except FileNotFoundError:
                    print('Dataset Not Found')

DataSet, dataname = CheckDataName()

cols = DataSet.shape[1]
X = DataSet[:, 0:cols - 1]
y = DataSet[:, cols - 1:cols]

lengthdata = len(X)
n_features = len(X[0])
n_classes = int(max(y) - min(y) + 1)

train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2)
nntrain_y = train_y
train_y = np.ravel(train_y)

if not os.path.exists("./modelparameters/{}".format(dataname)): os.makedirs("./modelparameters/{}".format(dataname))
file = open("modelparameters/{}/nclassesfeatures.txt".format(dataname), "w")
file.write(str(n_classes) + "," + str(n_features) + "," + str(lengthdata))
file.flush()

try:
    knnmodelaccuracy = np.loadtxt("modelparameters/{}/KNNmodelaccuracy.txt".format(dataname), delimiter=",")
    knnprevaccuracy, knnprevf1score, knntime = round(knnmodelaccuracy[0], 5), round(knnmodelaccuracy[1], 5), round(knnmodelaccuracy[2], 5)
    knntime, knntimemetric = TimeConversion(knntime)
except FileNotFoundError: knnprevaccuracy, knnprevf1score, knntime, knntimemetric = 0, 0, 0, ''
try:
    rfmodelaccuracy = np.loadtxt("modelparameters/{}/RFmodelaccuracy.txt".format(dataname), delimiter=",")
    rfprevaccuracy, rfprevf1score, rftime = round(rfmodelaccuracy[0], 5), round(rfmodelaccuracy[1], 5), round(rfmodelaccuracy[2], 5)
    rftime, rftimemetric = TimeConversion(rftime)
except FileNotFoundError: rfprevaccuracy, rfprevf1score, rftime, rftimemetric = 0, 0, 0, ''
try:
    mnbmodelaccuracy = np.loadtxt("modelparameters/{}/MNBmodelaccuracy.txt".format(dataname), delimiter=",")
    mnbprevaccuracy, mnbprevf1score, mnbtime = round(mnbmodelaccuracy[0], 5), round(mnbmodelaccuracy[1], 5), round(mnbmodelaccuracy[2], 5)
    mnbtime, mnbtimemetric = TimeConversion(mnbtime)
except FileNotFoundError: mnbprevaccuracy, mnbprevf1score, mnbtime, mnbtimemetric = 0, 0, 0, ''
try:
    adamodelaccuracy = np.loadtxt("modelparameters/{}/ADAmodelaccuracy.txt".format(dataname), delimiter=",")
    adaprevaccuracy, adaprevf1score, adatime = round(adamodelaccuracy[0], 5), round(adamodelaccuracy[1], 5), round(adamodelaccuracy[2], 5)
    adatime, adatimemetric = TimeConversion(adatime)
except FileNotFoundError: adaprevaccuracy, adaprevf1score, adatime, adatimemetric = 0, 0, 0, ''
try:
    svmmodelaccuracy = np.loadtxt("modelparameters/{}/SVMmodelaccuracy.txt".format(dataname), delimiter=",")
    svmprevaccuracy, svmprevf1score, svmC, svmtime = round(svmmodelaccuracy[0], 5), round(svmmodelaccuracy[1], 5), round(svmmodelaccuracy[2], 5), round(svmmodelaccuracy[3], 5)
    svmtime, svmtimemetric = TimeConversion(svmtime)
except FileNotFoundError: svmprevaccuracy, svmprevf1score, svmC, svmtime, svmtimemetric = 0, 0, 0, 0, ''
try:
    logitmodelaccuracy = np.loadtxt("modelparameters/{}/LOGITmodelaccuracy.txt".format(dataname), delimiter=",")
    logitprevaccuracy, logitprevf1score, logitC, logittime = round(logitmodelaccuracy[0], 5), round(logitmodelaccuracy[1], 5), round(logitmodelaccuracy[2], 5), round(logitmodelaccuracy[3], 5)
    logittime, logittimemetric = TimeConversion(logittime)
except FileNotFoundError: logitprevaccuracy, logitprevf1score, logitC, logittime, logittimemetric = 0, 0, 0, 0, ''
try:
    snnmodelaccuracy = np.loadtxt("modelparameters/{}/SNNmodelaccuracy.txt".format(dataname), delimiter=",")
    snnprevaccuracy, snnprevf1score, snnepochs, snnhiddenlayersize, snnregularization, snntime, snnoptimizer = round(snnmodelaccuracy[0], 5), round(snnmodelaccuracy[1], 5), int(snnmodelaccuracy[2]), int(snnmodelaccuracy[3]), round(snnmodelaccuracy[4], 5), round(snnmodelaccuracy[5], 5), snnmodelaccuracy[6]
    snntime, snntimemetric = TimeConversion(snntime)
except FileNotFoundError: snnprevaccuracy, snnprevf1score, snnepochs, snnhiddenlayersize, snnregularization, snntime, snntimemetric, snnoptimizer = 0, 0, 0, 0, 0, 0, '', 0
try:
    dnnmodelaccuracy = np.loadtxt("modelparameters/{}/DNNmodelaccuracy.txt".format(dataname), delimiter=",")
    dnnprevaccuracy, dnnprevf1score, dnnepochs, dnnhiddenlayersize1, dnnhiddenlayersize2, dnntime, dnnoptimizer = round(dnnmodelaccuracy[0], 5), round(dnnmodelaccuracy[1], 5), int(dnnmodelaccuracy[2]), int(dnnmodelaccuracy[3]), int(dnnmodelaccuracy[4]), round(dnnmodelaccuracy[5], 5), dnnmodelaccuracy[6]
    dnntime, dnntimemetric = TimeConversion(dnntime)
except FileNotFoundError: dnnprevaccuracy, dnnprevf1score, dnnepochs, dnnhiddenlayersize1, dnnhiddenlayersize2, dnntime, dnntimemetric, dnnoptimizer = 0, 0, 0, 0, 0, 0, '', 0
try:
    rnnmodelaccuracy = np.loadtxt("modelparameters/{}/RNNmodelaccuracy.txt".format(dataname), delimiter=",")
    rnnprevaccuracy, rnnprevf1score, rnnepochs, rnnhiddenlayersize, rnnnsequence, rnnsequencesize, rnntime, rnnoptimizer = round(rnnmodelaccuracy[0], 5), round(rnnmodelaccuracy[1], 5), int(rnnmodelaccuracy[2]), int(rnnmodelaccuracy[3]), int(rnnmodelaccuracy[4]), int(rnnmodelaccuracy[5]), round(rnnmodelaccuracy[6], 5), rnnmodelaccuracy[7]
    rnntime, rnntimemetric = TimeConversion(rnntime)
except FileNotFoundError: rnnprevaccuracy, rnnprevf1score, rnnepochs, rnnhiddenlayersize,rnnnsequence, rnnsequencesize,  rnntime, rnntimemetric, rnnoptimizer = 0, 0, 0, 0, 0, 0, 0, '', 0
try:
    binnmodelaccuracy = np.loadtxt("modelparameters/{}/BiNNmodelaccuracy.txt".format(dataname), delimiter=",")
    binnprevaccuracy, binnprevf1score, binnepochs, binnhiddenlayersize, binnnsequence, binnsequencesize, binntime, binnoptimizername = round(binnmodelaccuracy[0], 5), round(binnmodelaccuracy[1], 5), int(binnmodelaccuracy[2]), int(binnmodelaccuracy[3]), int(binnmodelaccuracy[4]), int(binnmodelaccuracy[5]), round(binnmodelaccuracy[6], 5), binnmodelaccuracy[7]
    binntime, binntimemetric = TimeConversion(binntime)
except FileNotFoundError: binnprevaccuracy, binnprevf1score, binnepochs, binnhiddenlayersize, binnnsequence, binnsequencesize, binntime, binntimemetric, binnoptimizer = 0, 0, 0, 0, 0, 0, 0, '', 0
try:
    convnnmodelaccuracy = np.loadtxt("modelparameters/{}/ConvNNmodelaccuracy.txt".format(dataname), delimiter=",")
    convnnprevaccuracy, convnnprevf1score, convnnepochs, convnnsize, convnntime, convnnoptimizername = round(convnnmodelaccuracy[0], 5), round(convnnmodelaccuracy[1], 5), int(convnnmodelaccuracy[2]), int(convnnmodelaccuracy[3]), round(convnnmodelaccuracy[4], 5), convnnmodelaccuracy[5]
    convnntime, convnntimemetric = TimeConversion(convnntime)
except FileNotFoundError: convnnprevaccuracy, convnnprevf1score, convnnepochs, convnnsize, convnntime, convnntimemetric, convnnoptimizer = 0, 0, 0, 0, 0, '', 0

snnoptimizernumdict = {0:'NA', 1:'Newton', 2:'Newton-CG', 3:'Newton-TR', 4:'BFGS', 5:'L-BFGS-B', 6:'Nelder-Mead', 7:'Powell', 8:'COBYLA', 9:'SLSQP', 10:'dogleg'}
dnnoptimizernumdict = {0:'NA', 1:'GradDescent', 2:'AdaDelta', 3:'AdaGrad', 4:'Momentum', 5:'Adam', 6:'Ftrl', 7:'ProxGD', 8:'ProxAda', 9:'RMSProp'}
snnoptimizerworddict = {v: k for k, v in snnoptimizernumdict.items()}
dnnoptimizerworddict = {v: k for k, v in dnnoptimizernumdict.items()}

print('\nData Length:', lengthdata)
print('Number of Features:', n_features)
print('Number of Classes:', n_classes)
print("\nCurrent Scores:\n")

print(tabulate([['K-Nearest Neighbour', round(knnprevaccuracy, 3), round(knnprevf1score,3), str(round(knntime,2))+knntimemetric,
                str('NA'), int(-1), str('NA'), str('NA'), str('NA')],
                ['Random Forest', round(rfprevaccuracy, 3), round(rfprevf1score,3), str(round(rftime,2))+rftimemetric,
                str('NA'), int(-1), str('NA'), str('NA'), str('NA')],
                ['Multinomial Naive Bayes', round(mnbprevaccuracy, 3), round(mnbprevf1score,3), str(round(mnbtime,2))+mnbtimemetric,
                 str('NA'), int(-1), str('NA'), str('NA'), str('NA')],
                ['Ada Boost Classifier', round(adaprevaccuracy, 3), round(adaprevf1score,3), str(round(adatime,2))+adatimemetric,
                 str('NA'), int(-1), str('NA'), str('NA'), str('NA')],
                ['RBF SVM', round(svmprevaccuracy, 3), round(svmprevf1score,3), str(round(svmtime,2))+svmtimemetric,
                 svmC, int(-1), str('NA'), str('NA'), str('NA')],
                ['Logistic Regression', round(logitprevaccuracy, 3), round(logitprevf1score, 3), str(round(logittime,2))+logittimemetric,
                 logitC, int(-1), str('NA'), str('NA'), str('NA')],
                ['Simple Neural Network', round(snnprevaccuracy, 3), round(snnprevf1score, 3), str(round(snntime,2))+snntimemetric,
                 snnregularization, snnepochs, snnhiddenlayersize, str('NA'), str(snnoptimizernumdict[snnoptimizer])],
                ['Deep Neural Network', round(dnnprevaccuracy, 3), round(dnnprevf1score, 3), str(round(dnntime,2))+dnntimemetric,
                 float(1.0), dnnepochs, str(dnnhiddenlayersize1)+'x'+str(dnnhiddenlayersize2), str('NA'), str(dnnoptimizernumdict[dnnoptimizer])],
                ['LSTM RNN', round(rnnprevaccuracy, 3), round(rnnprevf1score, 3), str(round(rnntime,2))+rnntimemetric,
                 float(1.0), rnnepochs, rnnhiddenlayersize, str(rnnnsequence)+'x'+str(rnnsequencesize), str(dnnoptimizernumdict[rnnoptimizer])],
                ['BiDirectional LSTM RNN', round(binnprevaccuracy, 3), round(binnprevf1score, 3), str(round(binntime,2))+binntimemetric,
                 float(1.0), binnepochs, binnhiddenlayersize, str(binnnsequence)+'x'+str(binnsequencesize), str(dnnoptimizernumdict[binnoptimizer])],
                ['Convolutional NN', round(convnnprevaccuracy, 3), round(convnnprevf1score, 3), str(round(convnntime,2))+convnntimemetric,
                 float(1.0), convnnepochs, str('NA'), str(convnnsize)+'x'+str(convnnsize), str(dnnoptimizernumdict[convnnoptimizer])]],
                headers=['Model Name', 'Accuracy', 'F1 Score','Run Time','Reg.','Epochs','Hidden Size','Seq. Num x Size','Optimizer'], tablefmt='orgtbl'))

ModelSelect()
RepeatedModelSelect()
