import argparse
import ast
import os
import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def latestckpt(name):
    return int(name.split(".")[-2].split("-")[-1])


def setNodeAttribute(node, tag, shapeArray):
    if (shapeArray is not None):
        if (tag == 'shape'):
            if (len(shapeArray) == 4):
                node.attr[tag].shape.CopyFrom(tf.TensorShape(shapeArray).as_proto())
            elif (len(shapeArray) == 3):
                shapeArray4 = [None] * 4

                shapeArray4[0] = 1
                shapeArray4[1] = shapeArray[1]
                shapeArray4[2] = shapeArray[2]
                shapeArray4[3] = shapeArray[3]
                node.attr[tag].shape.CopyFrom(tf.TensorShape(shapeArray).as_proto())


def main(input_node_name, output_node_name):
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        files = os.listdir(os.getcwd())
        meta_files = [s for s in files if s.endswith('.meta')]
        meta_files = sorted(meta_files, key=latestckpt)
        ckptFile = os.path.basename(meta_files[-1])
        ckpt_with_extension, ckpt_metansion = os.path.splitext(ckptFile)
        currentFolder = os.path.dirname(os.path.realpath(__file__))
        meta_path = os.path.join(currentFolder, ckptFile)
        # Restore the graph
        saver = tf.train.import_meta_graph(meta_path, clear_devices=True)
        graph_def = sess.graph_def
        for node in graph_def.node:
            if node.op == 'RefSwitch':
                node.op = 'Switch'
                for index in range(len(node.input)):
                    if 'moving_' in node.input[index]:
                        node.input[index] = node.input[index] + '/read'
            # replace AssignSub operation with Sub operation and delete 'use_locking' attribute
            elif node.op == 'AssignSub':
                node.op = 'Sub'
                if 'use_locking' in node.attr: del node.attr['use_locking']
            # replace AssignAdd operation with Add operation and delete 'use_locking' attribute
            elif node.op == 'AssignAdd':
                node.op = 'Add'
                if 'use_locking' in node.attr: del node.attr['use_locking']
            # replace Assign operation with Identity operation
            elif node.op == 'Assign':
                node.op = 'Identity'
                if 'use_locking' in node.attr: del node.attr['use_locking']
                if 'validate_shape' in node.attr: del node.attr['validate_shape']
                if len(node.input) == 2:
                    # input0: ref: Should be from a Variable node. May be uninitialized.
                    # input1: value: The value to be assigned to the variable.
                    # delete value to be assigned to variable
                    del node.input[1]
            # delete 'dilations' attribute
            if ('dilations') in node.attr: del node.attr['dilations']
            node.device = ""

        graphDef = optimize_for_inference_lib.optimize_for_inference(
            graph_def,
            [input_node_name],  # an array of the input node(s)
            [output_node_name] if type(output_node_name) is str else [item for item in output_node_name],
            # an array of output nodes
            tf.float32.as_datatype_enum)

        for node in graphDef.node:
            '''Check nodes which have operation as 'Const' with 2 input nodes. 
            These 2 input nodes are AssignMovingAvg and AssignMovingAvg_1 and delete these input nodes'''
            if node.op == 'Const':
                if len(node.input) == 2:
                    del node.input[1]
                    del node.input[0]
            # Replace input node to batchnorm/add node from 'moments/Squeeze_1' with 'moving_variance'
            elif 'batchnorm/add' in node.name:
                for index in range(len(node.input)):
                    if 'Squeeze_1' in node.input[index]:
                        node.input[index] = str(node.input[index]).replace("moments/Squeeze_1", "moving_variance")
            # Replace input node to batchnorm/mul_2 node from 'moments/Squeeze' with 'moving_mean'
            elif 'batchnorm/mul_2' in node.name:
                for index in range(len(node.input)):
                    if 'Squeeze' in node.input[index]:
                        node.input[index] = str(node.input[index]).replace("moments/Squeeze", "moving_mean")
            # add shape attribute to input_node if not present
            elif ((node.op == 'Placeholder' or node.op == 'Reshape') and node.name == input_node_name):
                setNodeAttribute(node, 'shape', [1, 160, 160, 1])

        tf.train.write_graph(graphDef, currentFolder, 'model.pbtxt', as_text=True)
        pbtxt_location = os.path.join(currentFolder, 'model.pbtxt')
        input_checkpoint = os.path.join(currentFolder, ckpt_with_extension)
        pb_location = os.path.join(currentFolder, 'model_frozenforInference.pb')
        freeze_graph.freeze_graph(
            input_graph=pbtxt_location,
            input_saver='',
            input_binary=False,
            input_checkpoint=input_checkpoint,  # an array of the input node(s)
            output_node_names=output_node_name if type(output_node_name) is str else ",".join(output_node_name),
            restore_op_name="save/restore_all",  # Unused.
            filename_tensor_name="save/Const:0",  # Unused.
            output_graph=pb_location,  # an array of output nodes
            clear_devices=True,
            initializer_nodes=''
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_node_name", type=str, required=True, default="batch", help="input node name")
    parser.add_argument("--output_node_name", '--names-list', nargs='+', required=True,default='fire_o/convolution', help="output node(s) name")
    parser.add_argument("--ckpt_path", type=str, required=True, default="", help="Checkpoint path")
    args = parser.parse_args()
    os.chdir(os.path.dirname(os.path.abspath(args.ckpt_path)))
    main(args.input_node_name, args.output_node_name)
