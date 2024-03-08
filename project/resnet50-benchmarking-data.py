# TVM - Tensor Virtual Machine
import tvm
# Relay - intermediate representation (IR) in TVM used for representing deep learning models
from tvm import relay
from tvm.contrib import graph_executor
# This line imports a utility function from TVM that can be used to download test data or models for experimentation
from tvm.contrib.download import download_testdata
import numpy as np
# torch - deep learning framework
import torch
# torchvision - provides datasets, model architectures and common image transformations for computer vision
import torchvision
from torchvision import transforms
from PIL import Image

input_name = "input0"
# 1 - one image at time, 3 - number of channels, in this case RGB, 224 - height, 224 - width
input_shape = [1, 3, 224, 224]
NUM_OF_ITERS = 15

def create_random_image(width, height):
    random_pixels = np.random.rand(height, width, 3) * 255
    random_pixels = np.array(random_pixels, dtype=np.uint8)
    image = Image.fromarray(random_pixels, 'RGB')
    return image

def load_dummy_image():
    img = create_random_image(224, 224)
    my_preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    img = my_preprocess(img)
    img = np.expand_dims(img, 0)
    return img

def load_model():
    # Load a pretrained PyTorch model, model has been trained on ImageNet, a large dataset of images used for image classification
    model_name = "resnet50"
    model = getattr(torchvision.models, model_name)(pretrained=True)
    # setting the model to evaluation mode with model.eval() ensures that these layers behave correctly when you're evaluating your model (like during validation or testing), rather than training it
    model = model.eval()
    return model

def generate_scripted_model(model):
    global input_shape
    # creates a tensor of the specified shape with random numbers drawn from a standard normal distribution
    input_data = torch.randn(input_shape)
    scripted_model = torch.jit.trace(model, input_data).eval()
    return scripted_model

def scripted_model_to_relay(scripted_model):
    global input_name, input_shape
    shape_list = [(input_name, input_shape)]
    mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)
    return mod, params

def quantize_model(mod, params, activation_bit):
    with relay.quantize.qconfig(global_scale=8.0,
                                nbit_input=8,
                                nbit_weight=8,
                                nbit_activation=activation_bit,
                                dtype_input='int8',
                                dtype_weight='int8',
                                dtype_activation='int' + str(activation_bit)):
        quantized_mod = relay.quantize.quantize(mod, params)
    return quantized_mod

def prepare_hardware_configurations(mod, params):
    # set GPU as target device
    target_gpu = tvm.target.Target("cuda", host="llvm")
    target_cpu = tvm.target.Target("llvm", host="llvm")
    dev_gpu = tvm.cuda(0)
    dev_cpu = tvm.cpu(0)
    # build relay
    with tvm.transform.PassContext(opt_level=3):
        lib_gpu = relay.build(mod, target=target_gpu, params=params)
    with tvm.transform.PassContext(opt_level=3):
        lib_cpu = relay.build(mod, target=target_cpu, params=params)

    return [
        [lib_gpu, dev_gpu],
        [lib_cpu, dev_cpu]
    ]

def get_hardware_configurations(mod, params):
    # [unquantized - [[GPU],[CPU]], quantized - [[GPU],[CPU]]]
    hardware_configurations = []
    quantization = False
    activation_bits = [8, 16, 32]
    for _ in range(2):
        if(quantization):
            for activation_bit in activation_bits:
                quantized_mod = quantize_model(mod, params, activation_bit)
                hardware_configurations.append(prepare_hardware_configurations(quantized_mod, params))
        else:
            hardware_configurations.append(prepare_hardware_configurations(mod, params))
        quantization = not quantization
    return hardware_configurations

def get_module(lib, dev):
    return graph_executor.GraphModule(lib["default"](dev))

def get_modules(hardware_configurations):
    # [unquantized_gpu, unquantized_cpu, quantized_gpu8, quantized_cpu8, quantized_gpu16, quantized_cpu16, quantized_gpu32, quantized_cpu32]
    modules = []
    for configuration in hardware_configurations:
        for config in configuration:
            lib, dev = config
            modules.append(get_module(lib, dev))
    return modules

def execute_graph_on_tvm(m, dev, input_name, img):
    # Set inputs
    dtype = "float32"
    m.set_input(input_name, tvm.nd.array(img.astype(dtype)))
    # Benchmarking
    num_runs = 10  # Number of runs to average over
    repeat = 3  # Number of times to repeat the measurement process, taking the minimum time
    timer = m.module.time_evaluator("run", dev, number=num_runs, repeat=repeat)
    timing = np.array(timer().results) * 1000  # Convert to milliseconds
    return timing.mean()

def get_result_dict():
    return {"unquantized": {True: [], False: []},
            "quantized8": {True: [], False: []},
            "quantized16": {True: [], False: []},
            "quantized32": {True: [], False: []}}

def decode(i):
    if(i == 0):return "unquantized"
    if(i == 1):return "quantized8"
    if(i == 2):return "quantized16"
    if(i == 3):return "quantized32"

def update_dictionary(result_dict, time, i, j):
    val = decode(i)
    if(j == 0):
        result_dict[val][True].append(time)
    else:
        result_dict[val][False].append(time)

def main():
    model = load_model()
    scripted_model = generate_scripted_model(model)
    mod, params = scripted_model_to_relay(scripted_model)

    hardware_configurations = get_hardware_configurations(mod, params)
    modules = get_modules(hardware_configurations)

    # repeat as many times as you want
    for iteration in range(NUM_OF_ITERS):
        result_dict = get_result_dict()
        img = load_dummy_image()
        for i in range(len(hardware_configurations)):
            configuration = hardware_configurations[i]
            for j in range(len(configuration)):
                module = modules[i * len(configuration) + j]
                lib, dev = configuration[j]
                time = execute_graph_on_tvm(module, dev, input_name, img)
                update_dictionary(result_dict, time, i, j)
                print(time)
        print(result_dict)
        with open("output-for-plotting.txt", "a") as file:
            file.write(f"Iteration{iteration + 1}:" + str(result_dict) + "\n")

main()

