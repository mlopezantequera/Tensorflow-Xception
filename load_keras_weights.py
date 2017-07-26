import tensorflow as tf

equivalences = {
  'block1_conv1' : 'convolution2d_1',
  'block1_bn1' : 'batchnormalization_1',
  'block1_conv2' : 'convolution2d_2',
  'block1_bn2' : 'batchnormalization_2',
  'block1_res_conv' : 'convolution2d_3',
  'block1_res_bn' : 'batchnormalization_3',
  'block2_dws_conv1' : 'separableconvolution2d_1',
  'block2_bn1' : 'batchnormalization_4',
  'block2_dws_conv2' : 'separableconvolution2d_2',
  'block2_bn2' : 'batchnormalization_5',
  'block2_res_conv' : 'convolution2d_4',
  'block2_res_bn' : 'batchnormalization_6',
  'block3_dws_conv1' : 'separableconvolution2d_3',
  'block3_bn1' : 'batchnormalization_7',
  'block3_dws_conv2' : 'separableconvolution2d_4',
  'block3_bn2' : 'batchnormalization_8',
  'block3_res_conv' : 'convolution2d_5',
  'block3_res_bn' : 'batchnormalization_9',
  'block4_dws_conv1' : 'separableconvolution2d_5',
  'block4_bn1' : 'batchnormalization_10',
  'block4_dws_conv2' : 'separableconvolution2d_6',
  'block4_bn2' : 'batchnormalization_11',
  'block5_dws_conv1' : 'separableconvolution2d_7',
  'block5_bn1' : 'batchnormalization_12',
  'block5_dws_conv2' : 'separableconvolution2d_8',
  'block5_bn2' : 'batchnormalization_13',
  'block5_dws_conv3' : 'separableconvolution2d_9',
  'block5_bn3' : 'batchnormalization_14',
  'block6_dws_conv1' : 'separableconvolution2d_10',
  'block6_bn1' : 'batchnormalization_15',
  'block6_dws_conv2' : 'separableconvolution2d_11',
  'block6_bn2' : 'batchnormalization_16',
  'block6_dws_conv3' : 'separableconvolution2d_12',
  'block6_bn3' : 'batchnormalization_17',
  'block7_dws_conv1' : 'separableconvolution2d_13',
  'block7_bn1' : 'batchnormalization_18',
  'block7_dws_conv2' : 'separableconvolution2d_14',
  'block7_bn2' : 'batchnormalization_19',
  'block7_dws_conv3' : 'separableconvolution2d_15',
  'block7_bn3' : 'batchnormalization_20',
  'block8_dws_conv1' : 'separableconvolution2d_16',
  'block8_bn1' : 'batchnormalization_21',
  'block8_dws_conv2' : 'separableconvolution2d_17',
  'block8_bn2' : 'batchnormalization_22',
  'block8_dws_conv3' : 'separableconvolution2d_18',
  'block8_bn3' : 'batchnormalization_23',
  'block9_dws_conv1' : 'separableconvolution2d_19',
  'block9_bn1' : 'batchnormalization_24',
  'block9_dws_conv2' : 'separableconvolution2d_20',
  'block9_bn2' : 'batchnormalization_25',
  'block9_dws_conv3' : 'separableconvolution2d_21',
  'block9_bn3' : 'batchnormalization_26',
  'block10_dws_conv1' : 'separableconvolution2d_22',
  'block10_bn1' : 'batchnormalization_27',
  'block10_dws_conv2' : 'separableconvolution2d_23',
  'block10_bn2' : 'batchnormalization_28',
  'block10_dws_conv3' : 'separableconvolution2d_24',
  'block10_bn3' : 'batchnormalization_29',
  'block11_dws_conv1' : 'separableconvolution2d_25',
  'block11_bn1' : 'batchnormalization_30',
  'block11_dws_conv2' : 'separableconvolution2d_26',
  'block11_bn2' : 'batchnormalization_31',
  'block11_dws_conv3' : 'separableconvolution2d_27',
  'block11_bn3' : 'batchnormalization_32',
  'block12_dws_conv1' : 'separableconvolution2d_28',
  'block12_bn1' : 'batchnormalization_33',
  'block12_dws_conv2' : 'separableconvolution2d_29',
  'block12_bn2' : 'batchnormalization_34',
  'block12_dws_conv3' : 'separableconvolution2d_30',
  'block12_bn3' : 'batchnormalization_35',
  'block12_res_conv' : 'convolution2d_6',
  'block12_res_bn' : 'batchnormalization_36',
  'block13_dws_conv1' : 'separableconvolution2d_31',
  'block13_bn1' : 'batchnormalization_37',
  'block13_dws_conv2' : 'separableconvolution2d_32',
  'block13_bn2' : 'batchnormalization_38',
  'block14_dws_conv1' : 'separableconvolution2d_33',
  'block14_bn1' : 'batchnormalization_39',
  'block14_dws_conv2' : 'separableconvolution2d_34',
  'block14_bn2' : 'batchnormalization_40',
  'fully_connected' : 'dense_2',}

def get_assign_ops(h5f):
  assign_ops = []
  notfound_vars = []
  unused_layers = [layername for layername in h5f.keys() if len(h5f[layername].keys()) > 0]
  for var in tf.global_variables():
    n = var.op.name
    if n.split('/')[1] in equivalences.keys():
      keras_layername = equivalences[n.split('/')[1]]
      if 'dws' in n:
        if 'depthwise' in n:
          keras_weightname = keras_layername + '_depthwise_kernel:0'
        elif 'pointwise' in n:
          keras_weightname = keras_layername + '_pointwise_kernel:0'
      elif 'fully_connected' in n:
        if 'weights' in n:
          keras_weightname = keras_layername + '_W:0'
        else:
          keras_weightname = keras_layername + '_b:0'
      elif '/weights' in n:  # normal convolutional
        if 'weights' in n:
          keras_weightname = keras_layername + '_W:0'
      elif 'bn' in n:
        if 'beta' in n:
          keras_weightname = keras_layername + '_beta:0'
        elif 'mean' in n:
          keras_weightname = keras_layername + '_running_mean:0'
        elif 'variance' in n:
          keras_weightname = keras_layername + '_running_std:0'
        elif 'gamma' in n:
          keras_weightname = keras_layername + '_gamma:0'

      if keras_layername in h5f.keys():
        value = h5f[keras_layername][keras_weightname][:]
        assign_ops.append(var.assign(value))
        if keras_layername in unused_layers:
          unused_layers.remove(keras_layername)
    else:
      notfound_vars.append(var)

  if len(unused_layers) > 0:
    print("The following weights were found in the file but not matched to any layer in the graph:")
    print(unused_layers)

  if len(notfound_vars) > 0:
    print("The following layers have no matching weights in the weights file. Returning init ops for them:")
    print([var.name for var in notfound_vars])

  init_op = tf.variables_initializer(notfound_vars)

  return assign_ops, init_op

