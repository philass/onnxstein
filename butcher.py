import onnx
import onnxruntime as ort
import numpy as np
import sys
import getopt


onnx_model_path = ""
output_model_path = "script_onnx.onnx"

def onnx_type(dtype):
    '''Returns equivalent onnx.TensorProto basetype for a given numpy type
    where dtype can be either a numpy dtype or np.float32, np.int64, etc.'''
    if isinstance(dtype, np.dtype):
        dtype = dtype.type
        return {
            np.float32: onnx.TensorProto.FLOAT,
            np.float64: onnx.TensorProto.DOUBLE,
            np.int32: onnx.TensorProto.INT32,
            np.int64: onnx.TensorProto.INT64,
            np.bool_: onnx.TensorProto.BOOL,
        }[dtype]
    else:
        return onnx.TensorProto.BOOL


def make_constant_node(output_name, tensor):
    tensor = np.asarray(tensor)
    return onnx.helper.make_node(
        "Constant",
        inputs=[],
        outputs=[output_name],
        value=onnx.helper.make_tensor(
            name=output_name,
            data_type=onnx_type(tensor.dtype),
            dims=tensor.shape,
            vals=tensor.flatten(),
        ),
    )

def variable_shape(graph, vname):
    # Check if the vname is an initializer
    for init in graph.initializer:
        if init.name == vname:
            return list(init.dims)
    entries = [e for e in graph.value_info if e.name == vname]
    if len(entries) != 1:
        return None
    [e] = entries
    dims = e.type.tensor_type.shape.dim
    shape = [d.dim_value if not d.dim_param else -1 for d in dims]
    return shape

def variable_type(graph, vname):
    for init in graph.initializer:
        if init.name == vname:
            return init.data_type
    entries = [e for e in graph.value_info if e.name == vname]
    if len(entries) != 1:
        print("THIS WENT WRONG IN VARIABLE TYPE")
        return None
    [e] = entries
    dt = e.type.tensor_type.elem_type
    return dt


def replacement(graph, node):
    if node.op_type == "NonZero":
        shape = variable_shape(graph, node.input[0])
        if shape:
            cast_output = f"cast_{node.output[0]}"
            constant_reshape_output = f"constant_reshape_{node.output}"
            reshape_output = f"reshape_{node.output}"
            constant_expand_shape_output = f"expand_shape_{node.output}"
            cast_node = onnx.helper.make_node("Cast", node.input, [cast_output], to = onnx.TensorProto.INT64, name = node.name) # 
            const_reshape_node = make_constant_node(constant_reshape_output, np.array([-1]))
            reshape_node = onnx.helper.make_node("Reshape", [cast_output, constant_reshape_output], [reshape_output])
            const_expand_shape_node = make_constant_node(constant_expand_shape_output, np.array([len(shape), 1]))
            expand_node = onnx.helper.make_node("Expand", [reshape_output, constant_expand_shape_output], node.output)
            return [cast_node, const_reshape_node, reshape_node, const_expand_shape_node, expand_node]
        else:
            cast_output = f"cast_{node.output[0]}"
            constant_reshape_output = f"constant_reshape_{node.output}"
            reshape_output = f"reshape_{node.output}"
            reshape2_output = f"reshape2_{node.output}"
            shape_output = f"shape_{node.output}"
            size_output = f"size_{node.output}"
            constant_concat_output = f"constant_concat_{node.output}"
            concat_output = f"concat_{node.output}"
            cast_node = onnx.helper.make_node("Cast", node.input, [cast_output], to = onnx.TensorProto.INT64, name = node.name) # 
            const_reshape_node = make_constant_node(constant_reshape_output, np.array([-1]))
            reshape_node = onnx.helper.make_node("Reshape", [cast_output, constant_reshape_output], [reshape_output])
            shape_node = onnx.helper.make_node("Shape", node.input, [shape_output])
            size_node = onnx.helper.make_node("Size", [shape_output], [size_output])
            reshape2_node = onnx.helper.make_node("Reshape", [size_output, constant_reshape_output], [reshape2_output])
            const_concat_node = make_constant_node(constant_concat_output, np.array([1]))
            concat_node = onnx.helper.make_node("Concat", [reshape2_output, constant_concat_output], [concat_output], axis = 0)
            expand_node = onnx.helper.make_node("Expand", [reshape_output, concat_output], node.output)
            return [cast_node, const_reshape_node, reshape_node, shape_node, size_node, reshape2_node, const_concat_node, concat_node, expand_node]

    #elif node.op_type == "ScatterElements":
    #    identity_node = onnx.helper.make_node("Identity", [node.input[0]], node.output, name=node.name)
    #    return [identity_node]
    elif node.op_type == "ScatterElements":
        if len(node.input) != 3:
            print("ScatterElements should have three elements")
            return None
        [input, indices, updates] = node.input
        print(input)
        input_shape = variable_shape(graph, input)
        input_typ = variable_type(graph, input)
        updates_shape = variable_shape(graph, updates)
        indices_shape = variable_shape(graph, indices)
        print("Printing input")
        print(input, input_shape)
        print("Printing updates")
        print(updates, updates_shape)
        print("Printing indices")
        print(indices, indices_shape)

        if not input_shape or -1 in input_shape or not updates_shape or -1 in updates_shape:
            print("ScatterElements Shape is dynamic")
            return [node]
        if len(updates_shape) != len(input_shape):
            print("ScatterElements input Shapes should be same rank")
            return [node]
        cast_output = f"cast_{node.output[0]}"
        resize_indices_output = f"resize_indices_{node.output[0]}"
        resize_updates_output = f"resize_updates_{node.output[0]}"
        constant_resize_shape_output = f"resize_shape_{node.output[0]}"
        first_add_output = f"first_add_{node.output[0]}"
        cast_node = onnx.helper.make_node("Cast", [indices], [cast_output], to = input_typ, name = node.name)
        const_resize_shape_node = make_constant_node(constant_resize_shape_output, np.array(input_shape))
        resize_indices_node = onnx.helper.make_node("Resize", [cast_output, "", "", constant_resize_shape_output], [resize_indices_output], name = node.name + "yeahbuddy")
        resize_updates_node = onnx.helper.make_node("Resize", [updates, "", "", constant_resize_shape_output], [resize_updates_output], name = node.name+ "yeahbuddy1")
        add_node1 = onnx.helper.make_node("Add", [resize_indices_output, resize_updates_output], [first_add_output])
        add_node2 = onnx.helper.make_node("Add", [first_add_output, input], node.output)
        return [cast_node, const_resize_shape_node, resize_indices_node, resize_updates_node, add_node1, add_node2]
    else:
        return [node]

def ort_session(input_path, output_path):
    sess_options = ort.SessionOptions()
    sess_options.optimized_model_filepath = output_path
    sess_options.graph_optimization_level = (ort.GraphOptimizationLevel.ORT_ENABLE_BASIC)
    onnx_session = ort.InferenceSession(input_path, sess_options)


def update_model(onnx_model):
    graph = onnx_model.graph
    list_of_new_nodes = [replacement(graph, node) for node in graph.node]
    new_nodes = []
    for nodes in list_of_new_nodes:
        new_nodes += nodes
    new_graph = onnx.helper.make_graph(new_nodes, graph.name, graph.input, graph.output, graph.initializer)
    new_model = onnx.helper.make_model(new_graph, producer_name=onnx_model.producer_name, ir_version=onnx_model.ir_version)
    onnx.checker.check_model(new_model)
    shaped_model = onnx.shape_inference.infer_shapes(new_model)
    onnx.checker.check_model(shaped_model)
    return shaped_model

argv = sys.argv[1:]
try:
    opts, args = getopt.getopt(argv, "hi:o")
except:
    print("py -i <inputfile> -o <outputfile>")
    sys.exit(2)
for opt, arg in opts:
    if opt == '-i':
        onnx_model_path = arg
    if opt == '-o':
        output_file = arg

