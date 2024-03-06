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
from PIL import Image

def load_model():
    # Load a pretrained PyTorch model, model has been trained on ImageNet, a large dataset of images used for image classification
    model_name = "resnet50"
    model = getattr(torchvision.models, model_name)(pretrained=True)
    # setting the model to evaluation mode with model.eval() ensures that these layers behave correctly when you're evaluating your model (like during validation or testing), rather than training it
    model = model.eval()
    return model

def generate_scripted_model(model):
    # We grab the TorchScripted model via tracing
    # 1 - one image at time, 3 - number of channels, in this case RGB, 224 - height, 224 - width
    input_shape = [1, 3, 224, 224]
    # creates a tensor of the specified shape with random numbers drawn from a standard normal distribution
    input_data = torch.randn(input_shape)
    scripted_model = torch.jit.trace(model, input_data).eval()
    return scripted_model

def load_image():
    ######################################################################
    # Load a test image
    img_url = "https://github.com/dmlc/mxnet.js/blob/main/data/cat.png?raw=true"
    img_path = download_testdata(img_url, "cat.png", module="data")
    img = Image.open(img_path).resize((224, 224))

    # Preprocess the image and convert to tensor
    from torchvision import transforms

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

def generate_relay_graph(img, scripted_model):
    # Import the graph to Relay
    input_name = "relay_graph"
    shape_list = [(input_name, img.shape)]
    mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)
    # Checking TVM code in the compiled module is optional and only for verification purposes
    with open("output-relay-unquantized.txt", "w") as relay_output:
        relay_output.write(str(mod))
    return mod, params, input_name

def hardware_configuration(mod, params):
    # set GPU as a target device
    target = tvm.target.Target("cuda", host="llvm")
    dev = tvm.cuda(0)
    # build relay
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target=target, params=params)
    return lib, dev

def execute_graph_on_tvm(lib, dev, input_name, img):
    # Execute the portable graph on TVM
    dtype = "float32"
    m = graph_executor.GraphModule(lib["default"](dev))
    # Set inputs
    m.set_input(input_name, tvm.nd.array(img.astype(dtype)))
    # Execute
    m.run()
    # Get outputs
    tvm_output = m.get_output(0)
    return tvm_output

def get_key_to_classname():
    # Look up synset name, get a dictionary with human-readable names of the classes
    synset_url = "".join(
        [
            "https://raw.githubusercontent.com/Cadene/",
            "pretrained-models.pytorch/master/data/",
            "imagenet_synsets.txt",
        ]
    )
    synset_name = "imagenet_synsets.txt"
    synset_path = download_testdata(synset_url, synset_name, module="data")
    with open(synset_path) as f:
        synsets = f.readlines()

    synsets = [x.strip() for x in synsets]
    splits = [line.split(" ") for line in synsets]
    key_to_classname = {spl[0]: " ".join(spl[1:]) for spl in splits}
    return key_to_classname

def get_id_to_key():
    class_url = "".join(
        [
            "https://raw.githubusercontent.com/Cadene/",
            "pretrained-models.pytorch/master/data/",
            "imagenet_classes.txt",
        ]
    )
    class_name = "imagenet_classes.txt"
    class_path = download_testdata(class_url, class_name, module="data")
    with open(class_path) as f:
        class_id_to_key = f.readlines()

    class_id_to_key = [x.strip() for x in class_id_to_key]
    return class_id_to_key

def recognize_output(class_id_to_key, key_to_classname, tvm_output, img, model):
    # Get top-1 result for TVM
    top1_tvm = np.argmax(tvm_output.numpy()[0])
    tvm_class_key = class_id_to_key[top1_tvm]

    # Convert input to PyTorch variable and get PyTorch result for comparison
    with torch.no_grad():
        torch_img = torch.from_numpy(img)
        output = model(torch_img)

        # Get top-1 result for PyTorch
        top1_torch = np.argmax(output.numpy())
        torch_class_key = class_id_to_key[top1_torch]

    print("Relay top-1 id: {}, class name: {}".format(top1_tvm, key_to_classname[tvm_class_key]))
    print("Torch top-1 id: {}, class name: {}".format(top1_torch, key_to_classname[torch_class_key]))


def main():
    model = load_model()
    scripted_model = generate_scripted_model(model)
    img = load_image()
    mod, params, input_name = generate_relay_graph(img, scripted_model)
    lib, dev = hardware_configuration(mod, params)
    tvm_output = execute_graph_on_tvm(lib, dev, input_name, img)
    key_to_classname = get_key_to_classname()
    class_id_to_key = get_id_to_key()
    recognize_output(class_id_to_key, key_to_classname, tvm_output, img, model)

main()


