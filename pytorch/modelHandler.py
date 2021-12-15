#Custom Handler for RockNet (for use in Hologram, using the water classification framework)

import os
import io

import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

import json
import logging

from ts.torch_handler.base_handler import BaseHandler

def compute_bbox_coordinates(mask_img, probs, lookup_range, verbose = 0):

    bbox_list = list()


    img = Image.fromarray(np.uint8(mask_img*255), 'L')
    thresh = cv2.threshold(np.array(img),128,255,cv2.THRESH_BINARY)[1]
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    for cntr in contours:
        x,y,w,h = cv2.boundingRect(cntr)
        count = 0
        score = 0
        bbox = {'i_min': y, 'j_min': x, 'i_max': y+h, 'j_max': x+w, 'score': 0}
        arr = (probs[1][y:y+h,x:x+w]).flatten()
        scale = 3
        comp = int(len(arr)/scale) if len(arr) >= scale else 1
        idx = arr.argsort()[-comp:]
        score = float(sum(arr[idx])/comp)
        #for pxl_i in range(y, y+h):
        #    for pxl_j in range(x, x+w):
        #        #if probs[1][pxl_i][pxl_j] > probs[0][pxl_i][pxl_j]:
        #        score += probs[1][pxl_i][pxl_j]
        #        count += 1
        #score = 0 if count == 0 else float(score) / count
        max = 0.6
        score = (score-0.5)/(max-0.5)
        if score > 1.0:
            score = 1.0
        bbox = {'i_min': y, 'j_min': x, 'i_max': y+h, 'j_max': x+w, 'score': score}
        bbox_list.append(bbox)
    return bbox_list

# Main rockHandler class
class rockHandler(BaseHandler):
    """
    A custom model handler implementation
    """

    def __init__(self):
        self._context = None
        self.initialized = False
        self.explain = False
        self.target = 0
        self.input_height = 332
        self.input_width = 514
        self.device = None
        self.map_location = None

    def initialize(self, context):
        """
        Initialize model
        """

        #Load the model
        self._context = context

        self.manifest = context.manifest

        #print(torch.cuda.is_available())

        self.map_location = 'cuda' if torch.cuda.is_available() else 'cpu'
        properties = context.system_properties
        model_dir = properties.get("model_dir")
        self.device = torch.device("cuda:" + str(properties.get("gpu_id")) \
                if torch.cuda.is_available() else "cpu")

        #Read model seralize/pt file
        serialized_file = self.manifest['model']['serializedFile']
        model_pt_path = os.path.join(model_dir, serialized_file)
        if not os.path.isfile(model_pt_path):
            raise RuntimeError("Missing the model.pt file")
        #model_file = self.manifest['model']['modelFile']
        #model_f_path = os.path.join(model_dir, model_file)
        #if not os.path.isfile(model_f_path):
        #    raise RuntimeError("Missing the model file")
        #prePath = self.manifest['model']['baseWeights']
        #pre_path = os.path.join(model_dir, prePath)
        #if not os.path.isfile(prePath):
        #    raise RuntimeError("Missing the pretrained weights")

        import Net_Fcn_Mod

        self.model = Net_Fcn_Mod.Net(NumClasses = 2, PreTrainedModelPath = '', UpdateEncoderBatchNormStatistics = True) #torch.jit.load(model_pt_path)
        self.model.load_state_dict(torch.load(model_pt_path, map_location=self.map_location))
        for param in self.model.parameters():
            param.requires_grad = False

        self.model.to(self.device)
        self.model.eval()
        logging.debug('Model file %s loaded successfully', model_pt_path)
        self.initialized = True

        if self.device.type == 'cuda':
            print(torch.cuda.is_available)
            print(torch.cuda.current_device())
            print(torch.cuda.device(0))
            print(torch.cuda.device_count())
            print(torch.cuda.get_device_name(0))

    # Preprocess a singular input image (normalize, convert to tensor)
    def preprocess_one(self, request):
        """
        Preprocess a single request
        """

        image_data = request.get("data") or request.get("body")

        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        #image_resized = np.array(image.resize((self.input_height, self.input_width)))/255.0

        #tensor_image = torch.from_numpy(image_resized).float().unsqueeze(0)
        #tensor_image = torch.autograd.Variable(tensor_image).to(self.device)
        normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
        transform = transforms.Compose([transforms.ToTensor(), normalize])
        tensor_image = transform(image).unsqueeze(0)

        #print(tensor_image.size())
        #print('presize')
        if torch.cuda.is_available():
            tensor_image = tensor_image.cuda()
        return tensor_image

    def preprocess(self, requests):
        """
        Transform raw input into model input data
        """

        #Take the input data and make it inference ready
        #preprocessed_data = data[0].get("data")
        #if preprocessed_data is None:
        #    preprocessed_data = data[0].get("body")

        return [self.preprocess_one(req) for req in requests]

    # Run the images through the network
    def inference(self, tensors):
        """
        Return a prediction
        """

        output = [self.model(x.permute(0,2,3,1))[0] for x in tensors]
        return output

    # Postprocess a single network output (return the bounding box coordinates and score)
    def postprocess_one(self, output):
        """
        Filter low confidence detections, run NMS for single output
        """
        class_id = 1

        out = output[0] #output.detach().numpy()[0,:,:,:]
        rgb, h, w = out.size()
        out = out.cpu()
        #if torch.cuda.is_available():
        #    sing_im = (torch.from_numpy(np.zeros((h,w)))).cuda()
        #else:
        sing_im = np.zeros((h,w))
        sing_im[out[1] > out[0]] = 1
        #print(out)
        bbox_coords = compute_bbox_coordinates(sing_im, out, lookup_range=5, verbose=0)
        print(bbox_coords)

        finOut = list()
        for box in bbox_coords:
            curBox = [box['j_min'] / w,
                      box['i_min'] / h,
                      box['j_max'] / w,
                      box['i_max'] / h,
                      box['score'],
                      1]
            #if (curBox[0] == curBox[2]) or (curBox[1] == curBox[3]):
                #import pdb
                #pdb.set_trace()
            #else:
            if (curBox[0] != curBox[2]) and (curBox[1] != curBox[3]):
                finOut.append(curBox)

        return finOut

    def postprocess(self, outputs):
        batch_bboxes = [self.postprocess_one(output) for output in outputs]
        return batch_bboxes

    def handle(self, data, context):
        model_input = self.preprocess(data)
        print('Input Complete')
        model_output = self.inference(model_input)
        print('Inference Complete')
        return self.postprocess(model_output)

_service = rockHandler()

def handle(data, context):
    if not _service.initialized:
        _service.initialize(context)

    if data is None:
        return None

    return _service.handle(data, context)

if __name__ == '__main__':

    print('Testing for GPU availability')
    print(torch.cuda.is_available())
    print("Test finish")
    #For testing locally
    from ts.context import Context

    model_name = 'rock'
    manifest = json.loads(open('manifest.json').read())
    model_dir = '.'
    batch_size = 1
    gpu = 0
    mns_version = '1.0'
    context = Context(model_name, model_dir, manifest, batch_size, gpu, mns_version)

    #Initialize the model, similar to how torchserve does
    handle(None, context)

    #Create requests
    image_paths = ['curIm.jpg']
    requests = []
    for path in image_paths:
        request = {}
        with open(path, 'rb') as f:
            bin_data = f.read()
        request = {'data': bin_data}
        requests.append(request)

    out = handle(requests, context)





