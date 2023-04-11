import tensorflow as tf
import model as testModel
model3 = testModel.BaseTLClassifier(model=tf.keras.applications.resnet.ResNet50, input_dim = (41,41,3), nClasses=2, trainable = False)
#model3.summary()

