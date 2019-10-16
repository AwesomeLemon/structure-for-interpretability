# from multi_task.models import ResNet, FaceAttributeDecoder, BasicBlock
from models.multi_faces_resnet import ResNet, BasicBlock, FaceAttributeDecoder

import torchvision.models as model_collection


def get_model(params):
    data = params['dataset']

    if 'celeba' in data:
        model = {}
        model['rep'] = ResNet(BasicBlock, [2,2,2,2])
        model['rep'].cuda()
        for t in params['tasks']:
            model[t] = FaceAttributeDecoder()
            model[t].cuda()
        return model

