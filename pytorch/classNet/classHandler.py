#Custom Handler for RockNet
import os
import io

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

import json

from ts.torch_handler.base_handler import BaseHandler

def compute_bbox_coordinates(mask_img, probs, lookup_range, verbose = 0):

    bbox_list = list()
    visited_pixels = list()

    bbox_found = 0

    for i in range(mask_img.shape[0]):
        for j in range(mask_img.shape[1]):

            if mask_img[i,j] == 1 and (i, j) not in visited_pixels:

                bbox_found += 1

                pixels_to_visit = list()

                count = 0

                bbox = {'i_min': i, 'j_min': j, 'i_max' : i, 'j_max':j, 'score':0}

                pxl_i = i
                pxl_j = j

                while True:
                    visited_pixels.append((pxl_i, pxl_j))
                    if probs[1][pxl_i][pxl_j] > probs[0][pxl_i][pxl_j]:
                        bbox['score'] += probs[1][pxl_i][pxl_j]
                        count += 1

                    bbox['i_min'] = min(bbox['i_min'], pxl_i)
                    bbox['j_min'] = min(bbox['j_min'], pxl_j)
                    bbox['i_max'] = max(bbox['i_max'], pxl_i)
                    bbox['j_max'] = max(bbox['j_max'], pxl_j)

                    i_min = max(0, pxl_i - lookup_range)
                    j_min = max(0, pxl_j - lookup_range)

                    i_max = min(mask_img.shape[0], pxl_i + lookup_range + 1)
                    j_max = min(mask_img.shape[1], pxl_j + lookup_range + 1)

                    for i_k in range(i_min, i_max):
                         for j_k in range(j_min, j_max):

                            if mask_img[i_k, j_k] == 1 and (i_k, j_k) not in visited_pixels and (\
                            i_k, j_k) not in pixels_to_visit:
                                pixels_to_visit.append((i_k, j_k))
                                visited_pixels.append((i_k, j_k))

                    if not pixels_to_visit:
                        break

                    else:
                        pixel = pixels_to_visit.pop()
                        pxl_i = pixel[0]
                        pxl_j = pixel[1]

                bbox['score'] = float(bbox['score'])/count
                bbox_list.append(bbox)

    if verbose:
        print("BBOX Found: %d" % bbox_found)

    return bbox_list

#lookup_range is maximum distance between same cluster pixels
#bbox_coords = compute_bbox_coordinates(image, lookup_range=0, verbose=0)



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

    def initialize(self, context):
        """
        Initialize model
        """

        #Load the model
        self._context = context

        self.manifest = context.manifest

        properties = context.system_properties
        model_dir = properties.get("model_dir")
        self.device = torch.device("cuda:" + str(properties.get("gpu_id")) \
                if torch.cuda.is_available() else "cpu")

        #Read model seralize/pt file
        serialized_file = self.manifest['model']['serializedFile']
        #print(serialized_file)
        #print(model_dir)
        model_pt_path = os.path.join(model_dir, serialized_file)
        #print(model_pt_path)
        if not os.path.isfile(model_pt_path):
            raise RuntimeError("Missing the model.pt file")

        #print(model_dir)
        model_path_DD = os.path.join(model_dir, "darkDark.torch")
        model_path_DL = os.path.join(model_dir, "darkLight.torch")
        model_path_LL = os.path.join(model_dir, "lightLight.torch")

        #print(model_path_DD)
        #model_file = self.manifest['model']['modelFile']
        #model_f_path = os.path.join(model_dir, model_file)
        #if not os.path.isfile(model_f_path):
        #    raise RuntimeError("Missing the model file")
        #prePath = self.manifest['model']['baseWeights']
        #pre_path = os.path.join(model_dir, prePath)
        #if not os.path.isfile(prePath):
        #    raise RuntimeError("Missing the pretrained weights")

        import Net_Fcn_Mod

        self.modelDD = Net_Fcn_Mod.Net(NumClasses = 2, PreTrainedModelPath = '', UpdateEncoderBatchNormStatistics = True) #torch.jit.load(model_pt_path)
        self.modelDD.load_state_dict(torch.load(model_path_DD))
        for param in self.modelDD.parameters():
            param.requires_grad = False
        self.modelDD.eval()

        self.modelDL = Net_Fcn_Mod.Net(NumClasses = 2, PreTrainedModelPath = '', UpdateEncoderBatchNormStatistics = True) #torch.jit.load(model_pt_path)
        self.modelDL.load_state_dict(torch.load(model_path_DL))
        for param in self.modelDL.parameters():
            param.requires_grad = False
        self.modelDL.eval()

        self.modelLL = Net_Fcn_Mod.Net(NumClasses = 2, PreTrainedModelPath = '', UpdateEncoderBatchNormStatistics = True) #torch.jit.load(model_pt_path)
        self.modelLL.load_state_dict(torch.load(model_path_LL))
        for param in self.modelLL.parameters():
            param.requires_grad = False
        self.modelLL.eval()

        num_classes = 18
        self.classModel = models.mobilenet_v2(pretrained = False)
        num_ftrs = self.classModel.classifier[1].in_features
        self.classModel.classifier[1] = nn.Linear(num_ftrs, num_classes)
        for param in self.classModel.parameters():
            param.requires_grad = False
        self.classModel.load_state_dict(torch.load(model_pt_path))
        self.classModel.eval()

        self.classDD = {0,1,2,3,4,5,6}
        self.classDL = {7,8,9,10,11,12}
        self.classLL = {13,14,15,16,17}

        self.initialized = True

    def preprocess_one(self, request):
        """
        Preprocess a single request
        """

        image_data = request.get("data") or request.get("body")

        image = np.array(Image.open(io.BytesIO(image_data)).convert("RGB"))
        #image_resized = np.array(image.resize((self.input_height, self.input_width)))/255.0

        #tensor_image = torch.from_numpy(image_resized).float().unsqueeze(0)
        #tensor_image = torch.autograd.Variable(tensor_image).to(self.device)
        normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
        transform = transforms.Compose([transforms.ToTensor(), normalize])
        tensor_image = transform(image).unsqueeze(0)
        #tensor_image = transform(torch.from_numpy(image)).unsqueeze(0)
        #tensor_image = torch.from_numpy(image.transpose(2,0,1)).float().unsqueeze(0)

        #print(tensor_image.size())
        #print('presize')
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

    def classInference(self, x):
        """
        Return a class output
        """

        imClass = np.argmax(self.classModel(x)) #Should be index of output

        print(self.classModel(x))
        print(imClass)

        if imClass in self.classDD:
            out = self.modelDD(x.permute(0,2,3,1))[0]
        elif imClass in self.classDL:
            out = self.modelDL(x.permute(0,2,3,1))[0]
        elif imClass in self.classLL:
            out = self.modelLL(x.permute(0,2,3,1))[0]
        else:
            out = self.modelDL(x.permute(0,2,3,1))[0]

        return out

    def inference(self, tensors):
        """
        Return a prediction
        """

        output = [self.classInference(x) for x in tensors]
        return output

    def postprocess_one(self, output):
        """
        Filter low confidence detections, run NMS for single output
        """
        class_id = 1

        out = output[0].cpu() #output.detach().numpy()[0,:,:,:]
        rgb, h, w = out.size()
        sing_im = np.zeros((h,w))
        sing_im[out[1] > out[0]] = 1
        #print(out)
        bbox_coords = compute_bbox_coordinates(sing_im, out, lookup_range=5, verbose=0)
        #print(bbox_coords)

        finOut = list()
        for box in bbox_coords:
            curBox = np.zeros(6)
            curBox[0] = box['j_min']/self.input_width
            curBox[1] = box['i_min']/self.input_height
            curBox[2] = box['j_max']/self.input_width
            curBox[3] = box['i_max']/self.input_height
            curBox[4] = box['score']
            curBox[5] = 1
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
        model_output = self.inference(model_input)
        return self.postprocess(model_output)

_service = rockHandler()

def handle(data, context):
    if not _service.initialized:
        _service.initialize(context)

    if data is None:
        return None

    return _service.handle(data, context)

if __name__ == '__main__':

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





