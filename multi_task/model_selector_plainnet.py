# from multi_task.models import ResNet, FaceAttributeDecoder, BasicBlock
from models.plainnet import PlainNet, BasicBlock, FaceAttributeDecoder

import torchvision.models as model_collection


def get_model(params):
    data = params['dataset']

    if 'celeba' in data:
        model = {}
        model['rep'] = PlainNet(BasicBlock, [1,1,1,1], params['chunks'])
        model['rep'].cuda()
        for t in params['tasks']:
            model[t] = FaceAttributeDecoder()
            model[t].cuda()
        return model

