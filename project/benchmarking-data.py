import tvm
from tvm import relay
import torch
import torchvision
from tvm.contrib.download import download_testdata
from PIL import Image
from torchvision import transforms
import numpy as np

NUMBER_OF_ITERATIONS = 1

def load_model():
    model_name = "resnet50"
    model = getattr(torchvision.models, model_name)(pretrained=True)
    model = model.eval()
    return model

def generate_scripted_model(model):
    input_shape = [1, 3, 224, 224]
    input_data = torch.randn(input_shape)
    scripted_model = torch.jit.trace(model, input_data).eval()
    return scripted_model

def create_random_image(width, height):
    # Generate random RGB values
    random_pixels = np.random.rand(height, width, 3) * 255

    # Convert the pixels to an unsigned byte array
    random_pixels = np.array(random_pixels, dtype=np.uint8)

    # Create an image from the pixel array
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

# quantization_parameters(quantization_executing: bool,weight_value: int, activation_value: int)
def tvm_compile(img, scripted_model, quantization_parameters, use_cuda):
    input_name = "input0"
    shape_list = [(input_name, img.shape)]
    mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)
    if(quantization_parameters[0]):
        with relay.quantize.qconfig(global_scale=8.0,
                                    nbit_input=quantization_parameters[1],
                                    nbit_weight=quantization_parameters[1],
                                    nbit_activation=quantization_parameters[2],
                                    dtype_input='int' + str(quantization_parameters[1]),
                                    dtype_weight='int' + str(quantization_parameters[1]),
                                    dtype_activation='int' + str(quantization_parameters[2]),
                                    rounding='UPWARD'):
            quantized_mod = relay.quantize.quantize(mod, params)

    ######################################################################
    target = None
    dev = None
    if(use_cuda):
        target = tvm.target.Target("cuda", host="llvm")
        dev = tvm.cuda(0)
    else:
        target = tvm.target.Target("llvm", host="llvm")
        dev = tvm.cpu(0)

    with tvm.transform.PassContext(opt_level=3):
        if(quantization_parameters[0]):
            lib = relay.build(quantized_mod, target=target, params=params)
        else:
            lib = relay.build(mod, target=target, params=params)

    ######################################################################
    from tvm.contrib import graph_executor

    dtype = "float32"
    m = graph_executor.GraphModule(lib["default"](dev))
    # Set inputs
    m.set_input(input_name, tvm.nd.array(img.astype(dtype)))

    ######################################################################
    # Benchmarking
    num_runs = 10  # Number of runs to average over
    repeat = 3  # Number of times to repeat the measurement process, taking the minimum time
    min_repeat_ms = 0  # Minimum time in milliseconds to run each measurement for
    timer = m.module.time_evaluator("run", dev, number=num_runs, repeat=repeat,min_repeat_ms=min_repeat_ms)
    timing = np.array(timer().results) * 1000  # Convert to milliseconds

    return timing.mean()

def update_dict(result_dict, quantization_params, cuda_on, time):
    if(not quantization_params[0]):
        result_dict["unquantized"][cuda_on].append(time)
    else:
        if(quantization_params[2] == 8):
            result_dict["quantized8"][cuda_on].append(time)
        elif (quantization_params[2] == 16):
            result_dict["quantized16"][cuda_on].append(time)
        else:
            result_dict["quantized32"][cuda_on].append(time)

def get_result_dict():
    return {"unquantized": {True: [], False: []},
            "quantized8": {True: [], False: []},
            "quantized16": {True: [], False: []},
            "quantized32": {True: [], False: []}}

def get_quantization_params():
    return [[False, 0, 0],
            [True, 8, 8],
            [True, 8, 16],
            [True, 8, 32]]

def main():
    # result_dict = get_result_dict()
    quantization_parameters = get_quantization_params()
    cuda_on = True
    model = load_model()
    scripted_model = generate_scripted_model(model)
    # loop
    reps = 0
    while(reps < NUMBER_OF_ITERATIONS):
        result_dict = get_result_dict()
        img = load_dummy_image()
        for quantization_params in quantization_parameters:
            for _ in range(2):
                time = tvm_compile(img, scripted_model, quantization_params, cuda_on)
                update_dict(result_dict, quantization_params, cuda_on, time)
                cuda_on = not cuda_on

        with open("output-for-plotting.txt", "a") as file:
            file.write(f"Iteration{reps + 1}:" + str(result_dict) + "\n")

        reps += 1


main()