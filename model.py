import tensorflow as tf
from tensorflow.keras import layers



class TLModelBuilder(tf.keras.Model):
  def __init__(self, model=None, image_dim = (41,41,3), nClasses=2, trainable = False, name = "model_0", **kwargs):
    super().__init__()
    self._image_dim = image_dim
    self._input_dim = tuple([None]+list(image_dim))
    self._n_classes = nClasses
    self._name = name
    self.SetTLModel(model, self._image_dim, self._n_classes, trainable, include_top=False) #will need to make it more flexible

    self.input_0 = None
    self.tlmodel_0 = None
    self.flatten_0 = None
    self.dense_0 = None
    self.out = None
    self.model = None

    if self._base_model:
      self.SetLayers()


  def SetLayers(self):
    if self._base_model is not None:
      self.input_0 = tf.keras.Input(shape=self._image_dim)
      self.tlmodel_0 = self._base_model
      self.flatten_0 = layers.Flatten(name="flatten")
      self.dense_0 = layers.Dense(self._n_classes, activation="softmax")
    else:
      print("Base model is not set properly")
        
  def call(self,x):
    for layer in [ self.tlmodel_0, self.flatten_0, self.dense_0]:
        x = layer(x)
    return x

  def GetFunctionalModel(self):
    x = self.input_0
    for layer in [ self.tlmodel_0, self.flatten_0, self.dense_0]:
        x = layer(x)
    # return tf.keras.Model(inputs=self.input_0, outputs=self.call(self.input_0))
    return tf.keras.Model(inputs=self.input_0, outputs=x, name = self._name)


  def SetTLModel(self, model, image_dim, n_classes, trainable = False, include_top=False,  **kwargs):
    try:
      self._base_model = model(input_shape=image_dim,include_top=include_top, classes=n_classes,  **kwargs)
    except:
      print("Set base model failed")

    try:
      for layer in self._base_model.layers:
        layer.trainable = trainable
      self.SetLayers()
    except:
      self._base_model = None
      print("SetModel failed")

