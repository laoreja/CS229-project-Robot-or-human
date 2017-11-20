from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle

from new_commons import *

name = "DNN"

tf.logging.set_verbosity(tf.logging.INFO)

def dnn_model_fn(features, labels, mode):

    input_layer = tf.reshape(features["x"], [-1, 29])
    dense1 = tf.layers.dense(inputs=input_layer, units=256, activation=tf.nn.relu)
    # Add dropout operation; 0.6 probability that element will be kept
    dropout1 = tf.layers.dropout(
            inputs=dense1, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    dense2 = tf.layers.dense(inputs=dropout1, units=512, activation=tf.nn.relu)
    dropout2 = tf.layers.dropout(
            inputs=dense2, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
            
#    dense3 = tf.layers.dense(inputs=dropout2, units=256, activation=tf.nn.relu)
#    dropout3 = tf.layers.dropout(
#            inputs=dense3, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    logits = tf.layers.dense(inputs=dropout2, units=2)

    predictions = {
            # Generate predictions (for PREDICT and EVAL mode)
            "classes": tf.argmax(input=logits, axis=1),
            # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
            # `logging_hook`.
            "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=2)
    loss = tf.losses.softmax_cross_entropy(
            onehot_labels=onehot_labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
        train_op = optimizer.minimize(
                loss=loss,
                global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    if mode == tf.estimator.ModeKeys.EVAL:
        eval_metric_ops = {
                "accuracy": tf.metrics.accuracy(
                        labels=labels, predictions=predictions["classes"])}
        return tf.estimator.EstimatorSpec(
                mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

class DNNEstimator():
   
    
    def get_params(self, deep=False):
        return {}

    def fit(self, X, y):
        # Create the Estimator
        self.dnn_classifier = tf.estimator.Estimator(
                model_fn=dnn_model_fn, model_dir="dnn_model")

        train_input_fn = tf.estimator.inputs.numpy_input_fn(
                x={"x": X},
                y=y,
                batch_size=100,
                num_epochs=None,
                shuffle=True)

        # Train the model
        self.dnn_classifier.train(
                input_fn=train_input_fn,
                steps=3000,
    #            hooks=[logging_hook]
                )

            # Evaluate the model and print results
#            eval_results = dnn_classifier.evaluate(input_fn=eval_input_fn)
#            print(eval_results)
    
        return self

#    def predict(self, X):
#        
#        input_fn = tf.estimator.inputs.numpy_input_fn(
#            x={"x": X},
#            num_epochs=1,
#            shuffle=False)
#        prediction = dnn_classifier.predict(input_fn=predict_input_fn)
#        prediction = [float(x['probabilities'][1]) for x in prediction]
#
#        return results
        
    def predict_proba(self, X):
        results = np.zeros([X.shape[0], 2])
        input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": X},
            num_epochs=1,
            shuffle=False)
        prediction = self.dnn_classifier.predict(input_fn=input_fn)
        for i, x in enumerate(prediction):
            results[i, 0] = x['probabilities'][0]
            results[i, 1] = x['probabilities'][1]
        return results
        
dnn = DNNEstimator()
X_train, y_train = prepareTrainData()
evaluateClassifier(dnn, X_train, y_train, name)
printSubmission(dnn, X_train, y_train, name)

