# -*- coding: utf-8 -*-
"""
@created on: 10/1/18,
@author: Himaprasoon,
@version: v0.0.1

Description:

Sphinx Documentation Status:

"""
import os
import tensorflow as tf


def get_transfered_weights_or_bias(model_path: str, variable_name: str, dimension=None):
    """

    :param model_path: Folder where tf model is saved
    :param variable_name: variable name
    :param dimension: optional, if provided validates the shape with the variable restored
    :return: the value restored
    """

    if not os.path.isdir(model_path):
        raise Exception(f"Model Folder {model_path} doesn't exist")
    if not os.path.exists(f'{model_path}/model.ckpt.meta'):
        raise Exception(f"Model Folder {model_path} doesn't contain model.ckpt.meta file")
    if model_path[-1] != "/":
        model_path += "/"
    model_graph = tf.get_default_graph()
    new_graph = tf.Graph()
    with new_graph.as_default():
        saver = tf.train.import_meta_graph(f'{model_path}model.ckpt.meta', clear_devices=True)
        with tf.Session() as sess:
            saver.restore(sess=sess, save_path=f'{model_path}model.ckpt')
            graph = sess.graph
            # print([(n.name) for n in graph.as_graph_def().node if "Variable" in n.op])
            t = sess.run(graph.get_tensor_by_name(f"{variable_name}:0"))
    model_graph.as_default()
    if dimension and t.shape != tuple(dimension):
        raise Exception(f"Variable {variable_name} has shape {t.shape} but the required shape is {tuple(dimension)}")
    return t