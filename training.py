
from tensorflow.python.distribute import distribution_strategy_context
from tensorflow.python.framework import ops
from tensorflow.python.framework import smart_cond
from tensorflow.python.framework import tensor_util
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.utils import losses_utils
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.keras.utils.generic_utils import deserialize_keras_object
from tensorflow.python.keras.utils.generic_utils import serialize_keras_object
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops.losses import losses_impl
from tensorflow.python.ops.losses import util as tf_losses_util
from tensorflow.python.util.tf_export import keras_export
from tensorflow.tools.docs import doc_controls
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import math_ops

from RNNbased_models import encoder, decoder



# RNNbased 

@tf.function
def train_step(inp, targ, encoder, decoder, enc_hidden, loss_function, optimizer):
  loss = 0

  with tf.GradientTape() as tape:
    enc_output, enc_hidden = encoder(inp, enc_hidden)

    dec_hidden = enc_hidden

    # Teacher forcing - feeding the target as the next input
    for t in range(0, 100):
      # passing enc_output to the decoder
      predictions, dec_hidden, _ = decoder(dec_hidden, enc_output)

      loss += loss_function(targ[:, t], predictions)

      # using teacher forcing
      dec_input = tf.expand_dims(targ[:, t], 1)

  batch_loss = (loss / int(targ.shape[1]))

  variables = encoder.trainable_variables + decoder.trainable_variables

  gradients = tape.gradient(loss, variables)

  optimizer.apply_gradients(zip(gradients, variables))

  return batch_loss




def eval_validation(inp, targ, encoder, decoder, loss_function):
  """
  Evaluate the test sequences
  """
  loss = 0
  enc_hidden = encoder.initialize_hidden_state()
  enc_out, enc_hidden = encoder(inp, enc_hidden)
  dec_hidden = enc_hidden
  

  for t in range(len(inp[0])):
    predictions, dec_hidden, attention_weights = decoder(dec_hidden, enc_out)
    loss += loss_function(targ[:, t], predictions)
    
  return loss / int(targ.shape[1])


# transformers
@tf.function
def transformer_train_step(inp, targ, transformer, loss_function, optimizer):

  with tf.GradientTape() as tape:

    predictions = transformer((inp, targ))

    loss = loss_function(targ, predictions)

  
  #batch_loss = (loss / int(targ.shape[1]))

  variables = transformer.trainable_variables

  gradients = tape.gradient(loss, variables)

  optimizer.apply_gradients(zip(gradients, variables))

  train_loss(loss)

  #return batch_loss




class Loss(object):
  def __init__(self, reduction=losses_utils.ReductionV2.AUTO, name=None):
    losses_utils.ReductionV2.validate(reduction)
    self.reduction = reduction
    self.name = name

  def __call__(self, y_true, y_pred, sample_weight=None):
    scope_name = 'lambda' if self.name == '<lambda>' else self.name
    graph_ctx = tf_utils.graph_context_for_symbolic_tensors(
        y_true, y_pred, sample_weight)
    with K.name_scope(scope_name or self.__class__.__name__), graph_ctx:
      losses = self.call(y_true, y_pred)
      return losses_utils.compute_weighted_loss(
          losses, sample_weight, reduction=self._get_reduction())
          
  def _get_reduction(self):
      """Handles `AUTO` reduction cases and returns the reduction value."""
      if distribution_strategy_context.has_strategy() and (
          self.reduction == losses_utils.ReductionV2.AUTO or
          self.reduction == losses_utils.ReductionV2.SUM_OVER_BATCH_SIZE):
        raise ValueError(
            'Please use `tf.keras.losses.Reduction.SUM` or '
            '`tf.keras.losses.Reduction.NONE` for loss reduction when losses are '
            'used with `tf.distribute.Strategy` outside of the built-in training '
            'loops. You can implement '
            '`tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE` using global batch '
            'size like:\n```\nwith strategy.scope():\n'
            '    loss_obj = tf.keras.losses.CategoricalCrossentropy('
            'reduction=tf.keras.losses.Reduction.NONE)\n....\n'
            '    loss = tf.reduce_sum(loss_obj(labels, predictions)) * '
            '(1. / global_batch_size)\n```\nPlease see '
            'https://www.tensorflow.org/tutorials/distribute/custom_training'
            ' for more details.')

      if self.reduction == losses_utils.ReductionV2.AUTO:
        return losses_utils.ReductionV2.SUM_OVER_BATCH_SIZE
      return self.reduction

    
    

class LossFunctionWrapper(Loss):
  def __init__(self,
               fn,
               reduction=losses_utils.ReductionV2.AUTO,
               name=None,
               **kwargs):
    super(LossFunctionWrapper, self).__init__(reduction=reduction, name=name)
    self.fn = fn
    self._fn_kwargs = kwargs

  def call(self, y_true, y_pred):
    if tensor_util.is_tensor(y_pred) and tensor_util.is_tensor(y_true):
      y_pred, y_true = tf_losses_util.squeeze_or_expand_dimensions(
          y_pred, y_true)
    return self.fn(y_true, y_pred, **self._fn_kwargs)

  def get_config(self):
    config = {}
    for k, v in six.iteritems(self._fn_kwargs):
      config[k] = K.eval(v) if tf_utils.is_tensor_or_variable(v) else v
    base_config = super(LossFunctionWrapper, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))




def nn_sigmoid_cross_entropy_with_logits(
    _sentinel=None,
    labels=None,
    logits=None,
    name=None):
  #nn_ops._ensure_xent_args("sigmoid_cross_entropy_with_logits", _sentinel,
  #                         labels, logits)

  #logits = ops.convert_to_tensor(logits, name="logits")
  #labels = ops.convert_to_tensor(labels, name="labels")
  #with ops.name_scope(name, "logistic_loss", [logits, labels]) as name:
  #  logits = ops.convert_to_tensor(logits, name="logits")
  #  labels = ops.convert_to_tensor(labels, name="labels")
  #  try:
  #    labels.get_shape().merge_with(logits.get_shape())
  #  except ValueError:
  #    raise ValueError("logits and labels must have the same shape (%s vs %s)" %
  #                     (logits.get_shape(), labels.get_shape()))

    
    ####Code de yaniss : pas de softmax #######
    #ratio_1 = 1.8  # les 1 pésent 0.2
    #ratio_0 = 1.2  # les 0 pésent 0.8
    
    #zeros = array_ops.zeros_like(logits, dtype=logits.dtype)
    #cond = (logits >= zeros)
    #relu_logits = array_ops.where(cond, logits, zeros)
    #neg_abs_logits = array_ops.where(cond, -logits, logits)

    #return math_ops.add(
    #    ratio_0 * (relu_logits - logits * labels),
    #    math_ops.log1p(math_ops.exp(neg_abs_logits)) * ((ratio_1-ratio_0) * labels + ratio_0),
    #    name=name)
    ####Fin Code de yaniss : pas de softmax #######


    # ---------------------------------------------------------------------------
    # CODE MODIFIED HERE !!!

    ####Code Normal  #######
    #return math_ops.add(
    #    -labels * math_ops.log1p(logits),
    #    -(1-labels) * math_ops.log1p(1-logits),
    #    name=name)
    ####Fin Code Normal  #######


    ####Code de romain, la sortie du décodeur doit être entre 0 et 1 #######
  #logits = ops.convert_to_tensor(logits, name="logits")
  #labels = ops.convert_to_tensor(labels, name="labels")

  #zeros = array_ops.zeros_like(logits, dtype=logits.dtype)
  #cond = (logits >= zeros)
  #relu_logits = array_ops.where(cond, logits, zeros)
  #neg_abs_logits = array_ops.where(cond, -logits, logits)
  #return math_ops.add(  relu_logits - logits * labels, math_ops.log1p(math_ops.exp(neg_abs_logits)),name=name)  

  #pos_weight=1+math_ops.mean(labels)
  log_weight = 1 + (pos_weight - 1) * labels
  return math_ops.add(  (1 - labels) * logits, log_weight * (math_ops.log1p(math_ops.exp(-math_ops.abs(logits))) +
  nn_ops.relu(-logits)), name=name)
  #return math_ops.add(
  #      -labels * math_ops.log1p(logits),
  #      -(1-labels) * math_ops.log1p(1-logits),
  #      name=name)
  #ratio_1 = 0.2#1.8  # les 1 pésent 0.2
  #ratio_0 = 0.8#1.2  # les 0 pésent 0.8
  #logitsclip = tf.clip_by_value(logits, clip_value_min=0.1, clip_value_max=0.9)
  #labelclip = tf.clip_by_value(labels, clip_value_min=0, clip_value_max=1)
  #val1=labels * math_ops.log1p(logits)
  #val1=-1*val1
  #val0=(1-labels) * math_ops.log1p(1-logits)
  #val0=-1*val0
  #return val1+val0
    #return math_ops.add(
     #   (-1*labels * math_ops.log1p(logitsclip))*ratio_1,
     #   (-1*(1-labels) * math_ops.log1p(1-logitsclip))*ratio_0,
     #   name=name)
    
    
    # --------------------------------------------------------------------------

def K_binary_crossentropy(target, output, from_logits=False):
  return nn_sigmoid_cross_entropy_with_logits(labels=target, logits=output)
  #if not from_logits:
  #  if (isinstance(output, (ops.EagerTensor, variables_module.Variable)) or
  #      output.op.type != 'Sigmoid'):
  #    epsilon_ = _constant_to_tensor(epsilon(), output.dtype.base_dtype)
  #    output = clip_ops.clip_by_value(output, epsilon_, 1. - epsilon_)

      # Compute cross entropy from probabilities.
  #    bce = target * math_ops.log(output + epsilon())
  #    bce += (1 - target) * math_ops.log(1 - output + epsilon())
  #    return -bce
  #  else:
  #    assert len(output.op.inputs) == 1
  #    output = output.op.inputs[0]
  



def binary_crossentropy(y_true, y_pred, from_logits=False, label_smoothing=0):  # pylint: disable=missing-docstring
  #y_pred = ops.convert_to_tensor(y_pred)
  y_true = math_ops.cast(y_true, y_pred.dtype)
  #label_smoothing = ops.convert_to_tensor(label_smoothing, dtype=K.floatx())

  #def _smooth_labels():
  #  return y_true * (1.0 - label_smoothing) + 0.5 * label_smoothing

  #y_true = smart_cond.smart_cond(label_smoothing,
  #                               _smooth_labels, lambda: y_true)
  return K.mean(
      tf.nn.weighted_cross_entropy_with_logits(labels=y_true, logits=y_pred,pos_weight=pos_weight), axis=-1)
      #nn_sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred), axis=-1)

    
    
    
class BinaryCrossentropy(LossFunctionWrapper):
  def __init__(self,
               from_logits=False,
               label_smoothing=0,
               reduction=losses_utils.ReductionV2.AUTO,
               name='binary_crossentropy'):
    super(BinaryCrossentropy, self).__init__(
        binary_crossentropy,
        name=name,
        reduction=reduction,
        from_logits=from_logits,
        label_smoothing=label_smoothing)
    self.from_logits = from_logits
    

def custom_loss_function(actual, predicted):
  return -actual * math_ops.log1p(predicted) - (1-actual) * math_ops.log1p(1-predicted)




def loss_function(real, pred):
  #mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss_ = loss_object_2(real, pred)

  #mask = tf.cast(mask, dtype=loss_.dtype)
  #loss_ *= mask

  return tf.reduce_mean(loss_)






class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, d_model, warmup_steps=4000):
    super(CustomSchedule, self).__init__()

    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)

    self.warmup_steps = warmup_steps

  def __call__(self, step):
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps ** -1.5)

    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)