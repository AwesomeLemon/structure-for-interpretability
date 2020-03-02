# import multi_task.models.my_multi_faces_resnet as my_resnet
# import multi_task.models.multi_faces_resnet as default_resnet

import models.my_multi_faces_resnet as my_resnet
import models.multi_faces_resnet as default_resnet

# from multi_task.models.my_multi_faces_resnet import ResNetSeparated, BasicBlock, FaceAttributeDecoder

def get_model(params):
    data = params['dataset']
    arc = params['architecture']
    if 'width_mul' in params:
        width_mul = params['width_mul']
    else:
        width_mul = 1
    if 'celeba' in data:
        model = {}
        if arc == 'resnet18':
            #todo: revert from 2 to 1.
            model['rep'] = my_resnet.ResNetSeparated(my_resnet.BasicBlock, [2, 2, 2, 2], params['chunks'], width_mul)
        if arc == 'resnet34':
            model['rep'] = my_resnet.ResNetSeparated(my_resnet.BasicBlock, [3, 4, 6, 3], params['chunks'], width_mul)
        if arc == 'resnet18_vanilla':
            model['rep'] = default_resnet.ResNet(default_resnet.BasicBlock, [2, 2, 2, 2])
        model['rep'].cuda()
        for t in params['tasks']:
            if 'vanilla' in arc:
                model[t] = default_resnet.FaceAttributeDecoder()
            else:
                model[t] = my_resnet.FaceAttributeDecoder()
            model[t].cuda()
        return model
