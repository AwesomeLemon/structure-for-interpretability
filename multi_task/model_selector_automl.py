# import multi_task.models.my_multi_faces_resnet as my_resnet
# import multi_task.models.multi_faces_resnet as default_resnet

import models.my_multi_faces_resnet as my_resnet
import models.multi_faces_resnet as default_resnet
import models.groupconv2_multi_faces_resnet as groupconv_resnet
import models.learnablegroups_multi_faces_resnet as learnablegroups_resnet
import models.maskcon_multi_faces_resnet as maskcon_multi_faces_resnet
import models.maskcon2_multi_faces_resnet as maskcon2_multi_faces_resnet
import models.binmatr_multi_faces_resnet as binmatr_multi_faces_resnet

# from multi_task.models.my_multi_faces_resnet import ResNetSeparated, BasicBlock, FaceAttributeDecoder
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
            model['rep'] = my_resnet.ResNetSeparated(my_resnet.BasicBlock, [2, 2, 2, 2], params['chunks'], width_mul)
        if arc == 'resnet34':
            model['rep'] = my_resnet.ResNetSeparated(my_resnet.BasicBlock, [3, 4, 6, 3], params['chunks'], width_mul)
        if arc == 'resnet18_vanilla':
            model['rep'] = default_resnet.ResNet(default_resnet.BasicBlock, [2, 2, 2, 2])
        if arc =='g_resnext50':#actually resnet18
            kwargs = {'groups': 32, 'width_per_group': 4}
            model['rep'] = groupconv_resnet.G_ResNet(groupconv_resnet.BasicBlockG, [2, 2, 2, 2], **kwargs)
        if arc == 'learnablegroups_resnet18':
            model['rep'] = learnablegroups_resnet.LearnableGroupsResNet(learnablegroups_resnet.BasicBlock, [2, 2, 2, 2])
        if arc == 'maskcon_resnet29':
            model['rep'] = maskcon_multi_faces_resnet.MaskConResNet(maskcon_multi_faces_resnet.Bottleneck, [3, 3, 3, -1], params['chunks'], width_mul)
        if arc == 'maskcon2_resnet29':
            model['rep'] = maskcon2_multi_faces_resnet.MaskConResNet(maskcon2_multi_faces_resnet.Bottleneck, [2, 2, 2, -1], params['chunks'], width_mul)
        if arc == 'maskcon2_resnet29_actual':
            model['rep'] = maskcon2_multi_faces_resnet.MaskConResNet(maskcon2_multi_faces_resnet.Bottleneck, [3, 3, 3, -1], params['chunks'], width_mul)
        if arc == 'binmatr_resnet18':
            model['rep'] = binmatr_multi_faces_resnet.BinMatrResNet(binmatr_multi_faces_resnet.BasicBlock, [2, 2, 2, 2], params['chunks'], width_mul, params['if_fully_connected'])
        model['rep'].to(device)
        for t in params['tasks']:
            if 'vanilla' in arc:
                model[t] = default_resnet.FaceAttributeDecoder()
            else:
                if arc == 'g_resnext50':
                    model[t] = groupconv_resnet.FaceAttributeDecoder()
                elif arc == 'learnablegroups_resnet18':
                    model[t] = learnablegroups_resnet.FaceAttributeDecoder()
                elif arc == 'maskcon_resnet29':
                    model[t] = maskcon_multi_faces_resnet.FaceAttributeDecoder()
                elif 'maskcon2' in arc:
                    model[t] = maskcon2_multi_faces_resnet.FaceAttributeDecoder()
                elif arc == 'binmatr_resnet18':
                    model[t] = binmatr_multi_faces_resnet.FaceAttributeDecoder()
                else:
                    model[t] = my_resnet.FaceAttributeDecoder()

            model[t].to(device)
        return model
