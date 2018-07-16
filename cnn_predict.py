import tensorflow as tf
import numpy as np
import cv2
import os
from ip import watershed

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def cnn_model(features, labels, mode):
    input_layer = tf.reshape(features["x"], [-1, 224, 224, 3])
    conv1 = tf.layers.conv2d(inputs=input_layer, filters=32, kernel_size=[5,5], padding="same", activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2,2], strides=2)
    conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2,2], strides=2)    
    pool2_flat = tf.reshape(pool2, [-1, 56 * 56 * 64])
    
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(inputs=dense, rate=0.4, training = mode == tf.estimator.ModeKeys.TRAIN)
    
    logits = tf.layers.dense(inputs = dropout, units = 2)
    
    if labels is not None:
        onehot_labels = tf.one_hot(indices = labels, depth = 2)
    
        loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)
    
    predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions, export_outputs={
            'classify': tf.estimator.export.PredictOutput(predictions)})

        
def main(unused_argv):
    saver = tf.train.import_meta_graph('tmp/brandberg_cnn/model.ckpt-1245.meta')

    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint('tmp/brandberg_cnn/'))
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)
        

    img = cv2.imread('/Users/thomas/Desktop/Uni/SoSe18/KI/BOOK-ZID0824731/images-raw/BOOK-0824731-0056.jpg')
    img2 = img.copy()
    
    newWidth = int(img.shape[1]*600/img.shape[0])
    img = cv2.resize(img, (newWidth, 600))
    
    selectiveSearch = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    selectiveSearch.setBaseImage(img)
    selectiveSearch.switchToSelectiveSearchFast()

    rects = selectiveSearch.process()
    print('Total Number of Region Proposals:',len(rects))

    rectsToDelete = []
    
    for i, rect in enumerate(rects):
        x, y, w, h = rect
        area = w * h
        
        if area > 90000 or area < 100 or w < 50 or h < 50:
            rectsToDelete.append(i)
    
    print(len(rects), "-", len(rectsToDelete), "=", len(rects)-len(rectsToDelete))

    for j in reversed(rectsToDelete):
        rects = np.delete(rects, j, 0)
    rects = rects[:500]

    #print(len(rects))
    images = []
    images2 = []
    hscale = img2.shape[0] / img.shape[0]
    wscale = img2.shape[1] / img.shape[1]
    
    for i, rect in enumerate(rects):
        x, y, w, h = rect
        rectImg1 = img2[int(y*hscale):int((y+h)*hscale), int(x*wscale):int((x+w)*wscale)]
        images2.append(rectImg1)
        rectImg1 = cv2.resize(rectImg1, (224, 224), interpolation=cv2.INTER_CUBIC)
        rectImg1 = cv2.cvtColor(rectImg1, cv2.COLOR_BGR2RGB)
        rectImg1 = rectImg1.astype(np.float32)
        
        images.append(rectImg1)

    images = np.asarray(images)

    classifier = tf.estimator.Estimator(
        model_fn=cnn_model, model_dir="/Users/thomas/Desktop/Uni/SoSe18/KI/tmp/brandberg_cnn")

    input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": images},
        num_epochs=1,
        shuffle=False)
    
    predictions = classifier.predict(input_fn = input_fn)
    list_pred = list(predictions)
    
    print(len(images2))
    
    ind = len(list_pred)-1
    deletedImages = 0
    for img in reversed(list_pred):
        probs = img.get('probabilities')
        #print(probs)
        if probs[0] < 0.99999 and probs[1] < 0.99999:
            print(ind)
            images2.pop(ind)
            deletedImages += 1
        ind -= 1
        
    print('length of images:',len(images2))
    print('number of deleted images:', deletedImages)
    print(len(list_pred))
    
    for image in images2:
        watershed.computeWatershed(image)

if __name__ == "__main__":
    tf.app.run()