# from multi_task.models import ResNet, FaceAttributeDecoder, BasicBlock
from models.my_multi_faces_resnet import ResNetSeparated, BasicBlock, FaceAttributeDecoder

def get_model(params):
    data = params['dataset']

    if 'celeba' in data:
        model = {}
        model['rep'] = ResNetSeparated(BasicBlock, [2,2,2,2], params['chunks'])
        model['rep'].cuda()
        for t in params['tasks']:
            model[t] = FaceAttributeDecoder()
            model[t].cuda()
        return model

