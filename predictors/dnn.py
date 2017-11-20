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
    """Model function for CNN."""
    # Input Layer
    # Reshape X to 4-D tensor: [batch_size, width, height, channels]
    # MNIST images are 28x28 pixels, and have one color channel
    input_layer = tf.reshape(features["x"], [-1, 29])

    # Dense Layer
    # Densely connected layer with 1024 neurons
    # Input Tensor Shape: [batch_size, 7 * 7 * 64]
    # Output Tensor Shape: [batch_size, 1024]
    dense1 = tf.layers.dense(inputs=input_layer, units=128, activation=tf.nn.relu)

    # Add dropout operation; 0.6 probability that element will be kept
    dropout1 = tf.layers.dropout(
            inputs=dense1, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    dense2 = tf.layers.dense(inputs=input_layer, units=512, activation=tf.nn.relu)

    # Add dropout operation; 0.6 probability that element will be kept
    dropout2 = tf.layers.dropout(
            inputs=dense2, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
            
    dense3 = tf.layers.dense(inputs=input_layer, units=256, activation=tf.nn.relu)

    # Add dropout operation; 0.6 probability that element will be kept
    dropout3 = tf.layers.dropout(
            inputs=dense3, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits layer
    # Input Tensor Shape: [batch_size, 1024]
    # Output Tensor Shape: [batch_size, 10]
    logits = tf.layers.dense(inputs=dropout3, units=2)

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
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
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
    


def main(unused_argv):
    X_train, y_train = prepareTrainData()
    tmp = np.hstack([X_train, y_train[:, None]])
##    print(human.shape) # 1881 1881/5 = 376
##    print(robot.shape) # 98, 98/5 = 19    
    human = tmp[y_train == 0, :]
    robot = tmp[y_train == 1, :]
    tmp_train_split = np.vstack([human[:-376, :], robot[:-19,:]])
    tmp_eval_split = np.vstack([human[-376:, :], robot[-19:,:]])
    tmp_train_split = shuffle(tmp_train_split, random_state=0)
    tmp_eval_split = shuffle(tmp_eval_split)
    
    train_data = tmp_train_split[:, :-1]
    train_labels = tmp_train_split[:, -1].astype(np.int32)
    eval_data = tmp_eval_split[:, :-1]
    eval_labels = tmp_eval_split[:, -1].astype(np.int32)
    
    
    # Create the Estimator
    dnn_classifier = tf.estimator.Estimator(
            model_fn=dnn_model_fn, model_dir="dnn_model")

    # Set up logging for predictions
    # Log the values in the "Softmax" tensor with label "probabilities"
#    tensors_to_log = {"probabilities": "softmax_tensor"}
#    logging_hook = tf.train.LoggingTensorHook(
#            tensors=tensors_to_log, every_n_iter=50)

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": train_data},
            y=train_labels,
            batch_size=100,
            num_epochs=None,
            shuffle=True)
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": eval_data},
            y=eval_labels,
            num_epochs=1,
            shuffle=False)


    for i in xrange(10):
        # Train the model
        dnn_classifier.train(
                input_fn=train_input_fn,
                steps=1000,
    #            hooks=[logging_hook]
                )

        # Evaluate the model and print results
        eval_results = dnn_classifier.evaluate(input_fn=eval_input_fn)
        print(eval_results)
    
    common, X_test = prepareTestFeatures()

    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
                x={"x": X_test},
                num_epochs=1,
                shuffle=False)
    prediction = dnn_classifier.predict(input_fn=predict_input_fn)

    prediction = [float(x['probabilities'][1]) for x in prediction]

    predictionDf = pd.DataFrame(data={"prediction": prediction})
    pd.concat([common['bidder_id'], predictionDf], axis=1).to_csv(
        "submissions/{}.csv".format(name),
        index=False,
    )

    
#    #evaluateClassifier(rf, X_train, y_train, name)
#    #printSubmission(rf, X_train, y_train, name)


if __name__ == "__main__":
    tf.app.run()