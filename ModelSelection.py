import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import numpy as np
import sklearn
import pickle
import time
from numpy import genfromtxt
from sklearn import datasets, neighbors, svm, tree, metrics
from sklearn.svm import SVC
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures
from sklearn.cross_validation import train_test_split
from scipy.optimize import minimize
import sklearn
import pandas as pd
import os
import io

if not os.path.exists("./data"):
	print("A data folder has been created in this directory.\nPlease place a supervised learning dataset in this folder.\n")
	os.makedirs("./data")
	print("A predictiondata folder has been created in this directory.\nPlease place the prediction data in this folder.\n The classifications must be in the right-most column for all data")
	os.makedirs("./predictiondata")

def printresult(totaltimer,timemetric,prevmodelaccuracy,prevmodelf1,accuracy,f1,test_y,test_x,clf,accuracytrain):
	print('Time: {}{}'.format(totaltimer, timemetric))
	print('Prior Accuracy:', prevmodelaccuracy)
	print('Prior F1:', prevmodelf1)
	print('Accuracy:', accuracy)
	print('Accuracy Change:', accuracy - prevmodelaccuracy)
	print('F1 Score:', f1)
	print('F1 Score Change:', f1 - prevmodelf1)
	if modelselect in ["knn","rf","svm"]:
		print(metrics.confusion_matrix(test_y, clf.predict(test_x)))
		print('Train Accuracy:',clf.score(train_x,train_y))
		print('Train - Test Accuracy:', accuracytrain- accuracy)
	elif modelselect in ["snn"]:
		print(metrics.confusion_matrix(test_y, clf))
		print('Train Accuracy:',accuracytrain)
		print('Train - Test Accuracy:', accuracytrain- accuracy)
	elif modelselect in ["dnn","rnn"]:
		print(metrics.confusion_matrix(test_y, clf))

def knn(knnmodelaccuracy):
	starttime = time.time()
	timemetric = "s"
	knnclf = neighbors.KNeighborsClassifier(n_jobs=-1)
	print("Training...")
	knnclf.fit(train_x,train_y)
	accuracy = knnclf.score(test_x,test_y)
	accuracytrain = knnclf.score(train_x,train_y)
	f1 = metrics.f1_score(test_y,knnclf.predict(test_x))
	totaltimer = round((time.time()-starttime),2)
	refinedtimer, timemetric = modeltimer(totaltimer)

	printresult(refinedtimer, timemetric,knnmodelaccuracy[0],knnmodelaccuracy[1],accuracy,f1,test_y,test_x,knnclf,accuracytrain)

	ans = input("Would you like to save the new parameters? (y/n): ")

	if ans in ["y"]:
		newmodelaccuracy = open("modelparameters/%s/knnmodelaccuracy.txt" %dataname,"w")
		newmodelaccuracy.write(str(accuracy) + "," + str(f1) + "," + str(totaltimer))
		newmodelaccuracy.flush()
		savepickle = open('modelparameters/%s/KNNpickle.pickle' %dataname,'wb')
		pickle.dump(knnclf,savepickle)
		print("Model Saved")
	else:
		print("Model Unsaved")

def rf(rfmodelaccuracy):
	starttime = time.time()
	timemetric = "s"
	rfclf = tree.DecisionTreeClassifier()
	print("Training...")
	rfclf.fit(train_x,train_y)
	rfpredictprob = rfclf.predict_proba(train_x)
	rfpredictscores = np.zeros((n_classes,),dtype=np.int)
	i = 0 
	while i < n_classes:
		sumprob = sum(rfpredictprob[:,i])
		rfpredictscores[i] = sumprob
		i = i+1
	np.save("modelparameters/%s/rfprobscores" %dataname,np.array(rfpredictscores))
	accuracy = rfclf.score(test_x,test_y)
	accuracytrain = rfclf.score(train_x,train_y)
	f1 = metrics.f1_score(test_y,rfclf.predict(test_x))
	totaltimer = round((time.time()-starttime),2)
	refinedtimer, timemetric = modeltimer(totaltimer)

	printresult(totaltimer, timemetric,rfmodelaccuracy[0],rfmodelaccuracy[1],accuracy,f1,test_y,test_x,rfclf,accuracytrain)

	ans = input("Would you like to save the new parameters? (y/n): ")
	
	if ans in ["y"]:
		newmodelaccuracy = open("modelparameters/%s/rfmodelaccuracy.txt" %dataname,"w")
		newmodelaccuracy.write(str(accuracy) + "," + str(f1) + "," + str(totaltimer))
		newmodelaccuracy.flush()
		savepickle = open("modelparameters/%s/RFpickle.pickle" %dataname,'wb')
		pickle.dump(rfclf,savepickle)
		print("Model Saved")
	else:
		print("Model Unsaved")

def svm(c,svmmodelaccuracy):
	starttime = time.time()
	timemetric = "s"
	svmclf = sklearn.svm.SVC(C=float(c),kernel='rbf',max_iter=1000)
	print("Training...")
	svmclf.fit(train_x, train_y)
	accuracy = svmclf.score(test_x,test_y)
	accuracytrain = svmclf.score(train_x,train_y)
	f1 = metrics.f1_score(test_y,svmclf.predict(test_x))
	totaltimer = round((time.time()-starttime),2)
	refinedtimer, timemetric = modeltimer(totaltimer)

	printresult(refinedtimer, timemetric,svmmodelaccuracy[0],svmmodelaccuracy[1],accuracy,f1,test_y,test_x,svmclf,accuracytrain)

	ans = input("Would you like to save the new parameters? (y/n): ")

	if ans in ["y"]:
		newmodelaccuracy = open("modelparameters/%s/svmmodelaccuracy.txt" %dataname,"w")
		newmodelaccuracy.write(str(accuracy) + "," + str(f1) + "," + str(C) + "," + str(totaltimer))
		newmodelaccuracy.flush()
		savepickle = open("modelparameters/%s/SVMpickle.pickle" %dataname,'wb')
		pickle.dump(svmclf,savepickle)
		print("Model Saved")
	else:
		print("Model Unsaved")

def snn(regularization,hm_epochs,hidden_size,degreepoly,snnmodelaccuracy):
	starttime = time.time()
	timemetric = "s"
	degreepoly = int(degreepoly)
	poly = PolynomialFeatures(degreepoly)
	hidden_size = int(hidden_size)
	regularization = int(regularization)
	maxiter = int(hm_epochs)
	encoder = OneHotEncoder(sparse=False)
	num_labels = int(n_classes)

	X = poly.fit_transform(train_x)
	input_size = X.shape[1]
	X = np.matrix(X)

	train_y_onehot = encoder.fit_transform(nntrain_y)
	test_y_onehot = encoder.fit_transform(test_y)
	y_onehot = np.matrix(train_y_onehot)

	params = (np.random.random(size=hidden_size * (input_size + 1) + num_labels * (hidden_size + 1)) - 0.5) * 0.25
	theta1 = np.matrix(np.reshape(params[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
	theta2 = np.matrix(np.reshape(params[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))
	a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)
	print("Training...") 
	J, grad = backprop(params, input_size, hidden_size, num_labels, X, y_onehot, regularization)
	fmin = minimize(fun=backprop, x0=params, args=(input_size, hidden_size, num_labels, X, y_onehot, regularization),
					method='TNC', jac=True, options={'maxiter': maxiter})
	X = np.matrix(X)
	theta1 = np.matrix(np.reshape(fmin.x[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
	theta2 = np.matrix(np.reshape(fmin.x[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))
	a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)
	y_pred = np.array(np.argmax(h, axis=1) + 1)
	correcttrain = [1 if a == b else 0 for (a, b) in zip(y_pred, train_y)]

	X = poly.fit_transform(test_x)
	X = np.matrix(X)
	a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)
	y_pred = np.array(np.argmax(h, axis=1) + 1)
	correct = [1 if a == b else 0 for (a, b) in zip(y_pred, test_y)]

	accuracy = (sum(map(int, correct)) / float(len(correct)))
	accuracytrain = (sum(map(int, correcttrain)) / float(len(correcttrain)))
	f1 = metrics.f1_score(test_y,y_pred)
	totaltimer = round((time.time()-starttime),2)
	refinedtimer, timemetric = modeltimer(totaltimer)

	printresult(refinedtimer,timemetric,snnmodelaccuracy[0],snnmodelaccuracy[1],accuracy,f1,test_y,test_x,y_pred,accuracytrain)
	ans = input("Would you like to save the new parameters? (y/n): ")

	if ans in ["y"]:
		newmodelaccuracy = open("modelparameters/%s/snnmodelaccuracy.txt" %dataname,"w")
		newmodelaccuracy.write(str(accuracy) + "," + str(f1) + "," + str(regularization) + "," + str(maxiter) + "," + str(hidden_size) + "," + str(degreepoly) + "," + str(totaltimer))
		newmodelaccuracy.flush()
		savepickle = open('modelparameters/%s/fminSNNpickle.pickle' %dataname,'wb')
		pickle.dump(fmin,savepickle)
		savepickle = open('modelparameters/%s/ypredSNNpickle.pickle' %dataname,'wb')
		pickle.dump(y_pred,savepickle)
		print("Model Saved")
	else:
		print("Model Unsaved")

def dnn(hm_epochs,hidden_size,degreepoly,dnnmodelaccuracy,n_features,n_classes):
	train_x, train_y,train_y_onehot,test_x,test_y,test_y_onehot, n_features, n_classes = createNNdata(dataname,degreepoly)
	starttime = time.time()

	x = tf.placeholder('float', [None, n_features])
	y = tf.placeholder('float')

	batch_size = 100

	prediction = deep_neural_network_model(x,hidden_size,n_classes,n_features)
	cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(prediction,y) )
	optimizer = tf.train.AdamOptimizer().minimize(cost)
	saver = tf.train.Saver()

	with tf.Session() as sess:
		sess.run(tf.initialize_all_variables())
		for epoch in range(int(hm_epochs)):
			epoch_loss = 0
			i = 0
			while i<len(train_x):
				start = i
				end = i+batch_size
				batch_x = np.array(train_x[start:end])
				batch_y = np.array(train_y_onehot[start:end])
				_, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
				epoch_loss += c
				i += batch_size
			print('Epoch', epoch+1, 'completed out of',hm_epochs,'loss:',epoch_loss)
		saver.save(sess, "modelparameters/%s/DNNmodel.ckpt" %dataname)
		correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
		accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
		
		y_p = tf.argmax(prediction, 1)
		val_accuracy, y_pred = sess.run([accuracy, y_p], feed_dict={x:test_x, y:test_y_onehot})

		accuracy = accuracy.eval({x:test_x, y:test_y_onehot})
		accuracytrain = accuracy
		f1 = metrics.f1_score(test_y,y_pred)
		totaltimer = round((time.time()-starttime),2)
		refinedtimer, timemetric = modeltimer(totaltimer)

		printresult(refinedtimer,timemetric,dnnmodelaccuracy[0],dnnmodelaccuracy[1],accuracy,f1,test_y,test_x,y_pred,accuracytrain)
		ans = input("Would you like to save the new parameters? (y/n)")

		if ans in ["y"]:
			newmodelaccuracy = open("modelparameters/%s/dnnmodelaccuracy.txt" %dataname,"w")
			newmodelaccuracy.write(str(accuracy) + "," + str(f1) + "," + str(hm_epochs) + "," + str(hidden_size) + "," + str(degreepoly) + "," + str(totaltimer))
			newmodelaccuracy.flush()
			print("Model saved")
		else:
			print("Model Unsaved")

def recurrentnn(hm_epochs,degreepoly,rnnmodelaccuracy,n_features,n_classes):
	train_x, train_y,train_y_onehot,test_x,test_y,test_y_onehot, n_features, n_classes = createNNdata(dataname,degreepoly)
	starttime = time.time()
	i = 128
	for i in range(128,0,-1):
		if len(train_x[:]) % i == 0:
			batch_size = i
			break
	print("Batch Size: ",batch_size)
	i = 0
	for i in range(round(n_features**0.5),1,-1):
		if n_features % i == 0:
			chunk_size = i
			n_chunks = int(n_features / chunk_size)				#nfeatures must be evenly divisible by batch size. 	#chunk size and nchunks must multiply to nfeatures
			break
	print("Number of Chunks: ",n_chunks)
	print("Chunks Size: ",chunk_size)
	
	rnn_size = 256

	x = tf.placeholder('float', [None, n_chunks,chunk_size])
	y = tf.placeholder('float')

	prediction = recurrent_neural_network(x,n_classes,n_features,rnn_size,n_chunks,chunk_size)
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction,y))
	optimizer = tf.train.AdamOptimizer().minimize(cost)
	saver = tf.train.Saver()

	with tf.Session() as sess:
		sess.run(tf.initialize_all_variables())
		for epoch in range(int(hm_epochs)):
			epoch_loss = 0
			i = 0
			while i<len(train_x):
				start = i
				end = i+batch_size
				batch_x = np.array(train_x[start:end])
				batch_x = batch_x.reshape((batch_size,n_chunks,chunk_size))
				batch_y = np.array(train_y_onehot[start:end])
				_, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
				epoch_loss += c
				i += batch_size
			print('Epoch', epoch+1, 'completed out of',hm_epochs,'loss:',epoch_loss)
		saver.save(sess, "modelparameters/%s/RNNmodel.ckpt" %dataname)
		correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
		accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
		
		y_p = tf.argmax(prediction, 1)
		val_accuracy, y_pred = sess.run([accuracy, y_p], feed_dict={x:test_x.reshape((-1,n_chunks,chunk_size)), y:test_y_onehot})

		accuracy = accuracy.eval({x:test_x.reshape((-1,n_chunks,chunk_size)), y:test_y_onehot})
		accuracytrain = accuracy
		f1 = metrics.f1_score(test_y,y_pred)
		totaltimer = round((time.time()-starttime),2)
		refinedtimer, timemetric = modeltimer(totaltimer)

		printresult(refinedtimer,timemetric,rnnmodelaccuracy[0],rnnmodelaccuracy[1],accuracy,f1,test_y,test_x,y_pred,accuracytrain)
		ans = input("Would you like to save the new parameters? (y/n)")

		if ans in ["y"]:
			newmodelaccuracy = open("modelparameters/%s/rnnmodelaccuracy.txt" %dataname,"w")
			newmodelaccuracy.write(str(accuracy) + "," + str(f1) + "," + str(hm_epochs) + "," + str(degreepoly) + "," + str(totaltimer))
			newmodelaccuracy.flush()
			print("Model saved")
		else:
			print("Model Unsaved")

def modeltimer(totaltimer):
	timemetric = "s"
	if totaltimer > 60 and totaltimer < 3600:
		totaltimer = totaltimer / 60
		timemetric = "m"
	elif totaltimer > 3600:
		totaltimer = totaltimer / 3600
		timemetric = "h"
	return totaltimer, timemetric

def summaryscores():
	try:
		knnmodelaccuracy = np.loadtxt("modelparameters/%s/knnmodelaccuracy.txt" %dataname,delimiter=",")
	except FileNotFoundError:
		knnmodelaccuracy = [0,0,0]
	try:
		rfmodelaccuracy = np.loadtxt("modelparameters/%s/rfmodelaccuracy.txt" %dataname,delimiter=",")
	except FileNotFoundError:
		rfmodelaccuracy = [0,0,0]
	try:
		svmmodelaccuracy = np.loadtxt("modelparameters/%s/svmmodelaccuracy.txt" %dataname,delimiter=",")
	except FileNotFoundError:
		svmmodelaccuracy = [0,0,0,0]
	try:
		snnmodelaccuracy = np.loadtxt("modelparameters/%s/snnmodelaccuracy.txt" %dataname,delimiter=",")
	except FileNotFoundError:
		snnmodelaccuracy = [0,0,0,0,0,0,0]
	try:
		dnnmodelaccuracy = np.loadtxt("modelparameters/%s/dnnmodelaccuracy.txt" %dataname,delimiter=",")
	except FileNotFoundError:
		dnnmodelaccuracy = [0,0,0,0,0,0]
	try:
		rnnmodelaccuracy = np.loadtxt("modelparameters/%s/rnnmodelaccuracy.txt" %dataname,delimiter=",")
	except FileNotFoundError:
		rnnmodelaccuracy = [0,0,0,0,0]
	knntime,knnmetric = modeltimer(knnmodelaccuracy[2])
	rftime,rfmetric = modeltimer(rfmodelaccuracy[2])
	svmtime,svmmetric = modeltimer(svmmodelaccuracy[3])
	snntime,snnmetric = modeltimer(snnmodelaccuracy[6])
	dnntime,dnnmetric = modeltimer(dnnmodelaccuracy[5])
	rnntime,rnnmetric = modeltimer(rnnmodelaccuracy[4])
	print("\nCurrent Scores:")
	print("K-Nearest Neighbour: Accuracy", round(knnmodelaccuracy[0],5), " F1", round(knnmodelaccuracy[1],5), " Duration {}{}".format(round(knntime,2),knnmetric))
	print("Random Forest: Accuracy", round(rfmodelaccuracy[0],5), " F1", round(rfmodelaccuracy[1],5), " Duration {}{}".format(round(rftime,2),rfmetric))
	print("RBF SVM: Accuracy", round(svmmodelaccuracy[0],5), " F1", round(svmmodelaccuracy[1],5), " C", svmmodelaccuracy[2], " Duration {}{}".format(round(svmtime,2),svmmetric))
	print("Simple NN: Accuracy ", round(snnmodelaccuracy[0],5), " F1", round(snnmodelaccuracy[1],5), " Epochs",snnmodelaccuracy[3], " HLSize", snnmodelaccuracy[4]," Reg", snnmodelaccuracy[2]," DegPoly", snnmodelaccuracy[5], " Duration {}{}".format(round(snntime,2),snnmetric))
	print("Deep NN: Accuracy", round(dnnmodelaccuracy[0],5), " F1", round(dnnmodelaccuracy[1],5), " Epochs", dnnmodelaccuracy[2], " HLSize", dnnmodelaccuracy[3], " DegPoly",dnnmodelaccuracy[4], " Duration {}{}".format(round(dnntime,2),dnnmetric))
	print("LSTM RNN: Accuracy", round(rnnmodelaccuracy[0],5), " F1", round(rnnmodelaccuracy[1],5), " Epochs", rnnmodelaccuracy[2], " DegPoly",rnnmodelaccuracy[3], " Duration {}{}".format(round(rnntime,2),rnnmetric))
	return knnmodelaccuracy,rfmodelaccuracy,svmmodelaccuracy,snnmodelaccuracy,dnnmodelaccuracy,rnnmodelaccuracy

def sigmoid(z):
	return 1 / (1 + np.exp(-z))

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

def backprop(params, input_size, hidden_size, num_labels, X, y, learning_rate):
	global snnepoch
	m = X.shape[0]
	X = np.matrix(X)
	y = np.matrix(y)
	theta1 = np.matrix(np.reshape(params[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
	theta2 = np.matrix(np.reshape(params[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))
	a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)
	J = 0
	delta1 = np.zeros(theta1.shape)
	delta2 = np.zeros(theta2.shape)
	for i in range(m):
		try:
			first_term = np.multiply(-y[i,:], np.log(h[i,:]))
		except ValueError:
			print("Check if you have classifications that are 0")
		second_term = np.multiply((1 - y[i,:]), np.log(1 - h[i,:]))
		J += np.sum(first_term - second_term)
	J = J / m
	J += (float(learning_rate) / (2 * m)) * (np.sum(np.power(theta1[:,1:], 2)) + np.sum(np.power(theta2[:,1:], 2)))
	for t in range(m):
		a1t = a1[t,:]
		z2t = z2[t,:]
		a2t = a2[t,:]
		ht = h[t,:]
		yt = y[t,:]
		d3t = ht - yt
		z2t = np.insert(z2t, 0, values=np.ones(1))
		d2t = np.multiply((theta2.T * d3t.T).T, sigmoid_gradient(z2t))
		delta1 = delta1 + (d2t[:,1:]).T * a1t
		delta2 = delta2 + d3t.T * a2t
	delta1 = delta1 / m
	delta2 = delta2 / m
	delta1[:,1:] = delta1[:,1:] + (theta1[:,1:] * learning_rate) / m
	delta2[:,1:] = delta2[:,1:] + (theta2[:,1:] * learning_rate) / m
	grad = np.concatenate((np.ravel(delta1), np.ravel(delta2)))
	print("Iteration:",snnepoch, " Cost:",J)
	snnepoch = snnepoch +1
	return J, grad

def cost(params, input_size, hidden_size, num_labels, X, y, learning_rate):
	m = X.shape[0]
	X = np.matrix(X)
	y = np.matrix(y)
	theta1 = np.matrix(np.reshape(params[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
	theta2 = np.matrix(np.reshape(params[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))
	a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)
	J = 0
	for i in range(m):
		first_term = np.multiply(-y[i,:], np.log(h[i,:]))
		second_term = np.multiply((1 - y[i,:]), np.log(1 - h[i,:]))
		J += np.sum(first_term - second_term)
	J = J / m
	J += (float(learning_rate) / (2 * m)) * (np.sum(np.power(theta1[:, 1:], 2)) + np.sum(np.power(theta2[:, 1:], 2)))
	return J

def deep_neural_network_model(data, hidden_size,n_classes,n_features):
	n_nodes_hl1 = int(hidden_size)
	n_nodes_hl2 = int(hidden_size)
	hidden_1_layer = {'f_fum':n_nodes_hl1,
					  'weight':tf.Variable(tf.random_normal([n_features, n_nodes_hl1])),
					  'bias':tf.Variable(tf.random_normal([n_nodes_hl1]))}
	hidden_2_layer = {'f_fum':n_nodes_hl2,
					  'weight':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
					  'bias':tf.Variable(tf.random_normal([n_nodes_hl2]))}
	output_layer = {'f_fum':None,
					'weight':tf.Variable(tf.random_normal([n_nodes_hl2, n_classes])),
					'bias':tf.Variable(tf.random_normal([n_classes])),}
	l1 = tf.add(tf.matmul(data,hidden_1_layer['weight']), hidden_1_layer['bias'])
	l1 = tf.nn.relu(l1)
	l2 = tf.add(tf.matmul(l1,hidden_2_layer['weight']), hidden_2_layer['bias'])
	l2 = tf.nn.relu(l2)
	output = tf.matmul(l2,output_layer['weight']) + output_layer['bias']

	return output

def recurrent_neural_network(x,n_classes,n_features,rnn_size,n_chunks,chunk_size):
	layer = {'weights':tf.Variable(tf.random_normal([rnn_size,n_classes])),
			 'biases':tf.Variable(tf.random_normal([n_classes]))}
	x = tf.transpose(x, [1,0,2])
	x = tf.reshape(x, [-1, chunk_size])
	x = tf.split(0, n_chunks, x)
	lstm_cell = rnn_cell.BasicLSTMCell(rnn_size,state_is_tuple=True)
	outputs, states = rnn.rnn(lstm_cell, x, dtype=tf.float32)
	output = tf.matmul(outputs[-1],layer['weights']) + layer['biases']

	return output

def createNNdata(dataname,degreepoly):
	info = [dataname,degreepoly]
	try:
		data = genfromtxt('modelparameters/{0}/{1}NNTraining.csv'.format(info[0],info[1]),delimiter=',')
		print("Loading Dataset...")
	except OSError:
		print("Building Dataset...")
		buildNNData(degreepoly,info)        
		data = genfromtxt('modelparameters/{0}/{1}NNTraining.csv'.format(info[0],info[1]),delimiter=',')
	test_data = genfromtxt('modelparameters/{0}/{1}NNTesting.csv'.format(info[0],info[1]),delimiter=',')

	print("Converting Training Set OneHot...")
	train_x=np.array([ i[1::] for i in data])
		#train_x=np.array([ i[::data.shape[0]-1] for i in data])
	train_y,train_y_onehot = convertOneHot(data)
	print("Converting Test Set OneHot...")
	test_x=np.array([ i[1::] for i in test_data])
	test_y,test_y_onehot = convertOneHot(test_data)
	print("Training...")
	n_features = data.shape[1]-1
	n_classes = len(test_y_onehot[0])
	return train_x, train_y,train_y_onehot,test_x,test_y,test_y_onehot,n_features,n_classes

def isPrime(n):		#Return True if Prime
	if n == 2:
		return True
	if n == 3:
		return True
	if n % 2 == 0:
		return False
	if n % 3 == 0:
		return False
	i = 5
	w = 2
	while i * i <= n:
		if n % i == 0:
			return False
		i += w
		w = 6 - w
	return True

def buildNNData(degreepoly,info):
	degreepoly = int(degreepoly)
	poly = PolynomialFeatures(degreepoly)
	dnntrain_x = poly.fit_transform(train_x)
	dnntest_x = poly.fit_transform(test_x)
	lenPrime = isPrime(len([train_x]))
	if lenPrime == True:
		dnntrain_x = dnntrain_x[0:dnntrain_x.shape[0]-1,:]
	f=open('modelparameters/{0}/{1}NNTraining.csv'.format(info[0],info[1]),'w')
	for i,j in enumerate(dnntrain_x):
		j = j[1:cols]
		k=np.append(np.array(train_y[i]),j)
		f.write(",".join([str(s) for s in k]) + '\n')
	f.close()
	f=open('modelparameters/{0}/{1}NNTesting.csv'.format(info[0],info[1]),'w')
	for i,j in enumerate(dnntest_x):
		j = j[1:cols]
		k=np.append(np.array(test_y[i]),j)
		f.write(",".join([str(s) for s in k]) + '\n')
	f.close()

def convertOneHot(data):
	y=np.array([int(i[0]) for i in data])
	y_onehot=[0]*len(y)
	for i,j in enumerate(y):
		y_onehot[i]=[0]*(y.max() + 1)
		y_onehot[i][j]=1
	return (y,y_onehot)

def use_simple_neural_network(input_data):
	X = input_data
	fmin = pd.read_pickle('modelparameters/%s/SNNpickle.pickle' %dataname)
	snnmodelaccuracy = np.loadtxt("modelparameters/%s/snnmodelaccuracy.txt" %dataname,delimiter=",")
	hidden_size = int(snnmodelaccuracy[4])
	degreepoly = int(snnmodelaccuracy[5])
	if degreepoly > 1:
		poly = PolynomialFeatures(degreepoly)
		X = poly.fit_transform(X)
	theta1 = np.matrix(np.reshape(fmin.x[:hidden_size * (n_features + 1)], (hidden_size, (n_features + 1))))
	theta2 = np.matrix(np.reshape(fmin.x[hidden_size * (n_features + 1):], (n_classes, (hidden_size + 1))))
	a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)
	snnPredict = np.array(np.argmax(h, axis=1))
	print("SNN Predict: ", snnPredict)

def use_deep_neural_network(input_data):
	x = tf.placeholder('float', [None, n_features])
	dnnmodelaccuracy = np.loadtxt("modelparameters/%s/dnnmodelaccuracy.txt" %dataname,delimiter=",")
	hidden_size = int(dnnmodelaccuracy[3])
	prediction = deep_neural_network_model(x,hidden_size,n_classes,n_features)
	saver = tf.train.Saver()

	with tf.Session() as sess:
		sess.run(tf.initialize_all_variables())
		saver.restore(sess,"modelparameters/%s/model.ckpt" %dataname)
		result = (sess.run(tf.argmax(prediction.eval(feed_dict={x:[input_data]}),1)))
		print(result)

snnepoch = -1

ans = input("\nRetrain or Use a model? (r/u): ")

if ans in ["r"]:
	dirs = os.listdir("data/")
	print("\nAvailable Datasets: \n")
	for file in dirs:
		print(file)
	datarun = input("\nEnter a Dataset:")
	dataname = datarun[:-4]
	print("Loading Data...")

	df = np.loadtxt('data/%s' %datarun, delimiter=",")
	cols = df.shape[1]
	X = df[:,0:cols-1]
	y = df[:,cols-1:cols]
	
	if not os.path.exists("./modelparameters/%s" %dataname):
		os.makedirs("./modelparameters/%s" %dataname)
	with io.FileIO("modelparameters/%s/nclassesfeatures.txt" %dataname,"w") as file:
		file = open("modelparameters/%s/nclassesfeatures.txt" %dataname,"w")
		file.write(str(int(max(y))) + "," + str(len(X[0])))
		file.flush()
	
	n_classes = max(y)
	n_features = len(X[0])

	train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2)
	nntrain_y = train_y
	train_y = np.ravel(train_y)
	
	print("Length of Data:", len(X[:]))
	print("Number of Features:", n_features)
	print("Number of Classes:", n_classes)

	knnmodelaccuracy, rfmodelaccuracy, svmmodelaccuracy,snnmodelaccuracy,dnnmodelaccuracy, rnnmodelaccuracy = summaryscores()

	modelselect = input("Retrain which model? (knn/rf/svm/snn/dnn/rnn): ")
	if modelselect in ["knn"]:
		knn(knnmodelaccuracy)
	elif modelselect in ["rf"]:
		rf(rfmodelaccuracy)
	elif modelselect in ["svm"]:
		C = input("Enter Value for C: ")
		svm(C, svmmodelaccuracy)
	elif modelselect in ["snn"]:
		regularization = input("Enter Value for Regularization : ")
		hm_epochs = input("How many epochs? ")
		degreepoly = input("Degree Polynomial Feature Expansion: ")
		print("Recommended Hidden Size: ",n_features**(int(degreepoly)))
		hidden_size = input("Enter Size of Hidden Layer: ")
		snn(regularization,hm_epochs,hidden_size,degreepoly,snnmodelaccuracy)
	elif modelselect in ["dnn"]:
		hm_epochs = input("How many epochs? ")
		degreepoly = input("Degree Polynomial Feature Expansion: ")
		print("Recommended Hidden Size: ",n_features**(int(degreepoly))*2)
		hidden_size = input("Enter Size of Hidden Layers: ")
		dnn(hm_epochs,hidden_size,degreepoly,dnnmodelaccuracy,n_features,n_classes)
	elif modelselect in ["rnn"]:
		hm_epochs = input("How many epochs? ")
		degreepoly = 1
		recurrentnn(hm_epochs,degreepoly,rnnmodelaccuracy,n_features,n_classes)
	else:
		print("Unacceptable command")
	summaryscores()

elif ans in ["u"]:
	dirs = os.listdir("modelparameters/")
	print("\nCreated Models: \n")
	for file in dirs:
		print(file)
	dataname = input("\nEnter Model:")
	print("\nAvailable Prediction Data\n")
	dirs = os.listdir("predictiondata/")
	for file in dirs:
		print(file)
	predictionname = input("\nEnter Prediction Data:")
	df = np.loadtxt('predictiondata/%s' %predictionname, delimiter=",")
	modeldetails = np.loadtxt("modelparameters/%s/nclassesfeatures.txt" %dataname,delimiter=",")
	n_classes = int(modeldetails[0])
	n_features = int(modeldetails[1])
	datalength = df.shape[1]
	X = df[0,0:n_features]
	input_data = np.array([X])
	try:
		knnclf = pd.read_pickle('modelparameters/%s/KNNpickle.pickle' %dataname)
		print("KNN Predict: ", knnclf.predict(input_data))
	except FileNotFoundError:
		print("KNN untrained")
	try:
		rfclf = pd.read_pickle('modelparameters/%s/RFpickle.pickle' %dataname)
		print("RF Predict: ", rfclf.predict(input_data))
	except FileNotFoundError:
		print("RF untrained")
	try:
		svmclf = pd.read_pickle('modelparameters/%s/SVMpickle.pickle' %dataname)
		print("SVM Predict: ", svmclf.predict(input_data))
	except FileNotFoundError:
		print("SVM untrained")
	#use_simple_neural_network(input_data)
	input_data = X
	input_data = np.insert(input_data,0,1)
	n_classes = n_classes+1
	n_features = n_features+1
	use_deep_neural_network(input_data)
else:
	print("Unacceptable command")

