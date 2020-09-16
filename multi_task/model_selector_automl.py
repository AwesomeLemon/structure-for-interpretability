# import multi_task.models.my_multi_faces_resnet as my_resnet
# import multi_task.models.multi_faces_resnet as default_resnet

import models.my_multi_faces_resnet as my_resnet
import models.multi_faces_resnet as default_resnet
import models.groupconv2_multi_faces_resnet as groupconv_resnet
import models.learnablegroups_multi_faces_resnet as learnablegroups_resnet
import models.maskcon_multi_faces_resnet as maskcon_multi_faces_resnet
import models.maskcon2_multi_faces_resnet as maskcon2_multi_faces_resnet
import models.binmatr_multi_faces_resnet as binmatr_multi_faces_resnet
import models.binmatr2_multi_faces_resnet as binmatr2_multi_faces_resnet
import models.binmatr_fullconv_multi_faces_resnet as binmatr_fullconv_multi_faces_resnet
import models.pspnet as pspnet

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
        if arc == 'g_resnext50':  # actually resnet18
            kwargs = {'groups': 32, 'width_per_group': 4}
            model['rep'] = groupconv_resnet.G_ResNet(groupconv_resnet.BasicBlockG, [2, 2, 2, 2], **kwargs)
        if arc == 'learnablegroups_resnet18':
            model['rep'] = learnablegroups_resnet.LearnableGroupsResNet(learnablegroups_resnet.BasicBlock, [2, 2, 2, 2])
        if arc == 'maskcon_resnet29':
            model['rep'] = maskcon_multi_faces_resnet.MaskConResNet(maskcon_multi_faces_resnet.Bottleneck,
                                                                    [3, 3, 3, -1], params['chunks'], width_mul)
        if arc == 'maskcon2_resnet29':
            model['rep'] = maskcon2_multi_faces_resnet.MaskConResNet(maskcon2_multi_faces_resnet.Bottleneck,
                                                                     [2, 2, 2, -1], params['chunks'], width_mul)
        if arc == 'maskcon2_resnet29_actual':
            model['rep'] = maskcon2_multi_faces_resnet.MaskConResNet(maskcon2_multi_faces_resnet.Bottleneck,
                                                                     [3, 3, 3, -1], params['chunks'], width_mul)
        if arc == 'binmatr_resnet18':
            model['rep'] = binmatr_multi_faces_resnet.BinMatrResNet(binmatr_multi_faces_resnet.BasicBlock, [2, 2, 2, 2],
                                                                    params['chunks'], width_mul,
                                                                    params['if_fully_connected'])
        if arc == 'binmatr2_resnet18':
            block_to_use = binmatr2_multi_faces_resnet.BasicBlock

            def check_param_contradiction():
                if not (block_to_use == binmatr2_multi_faces_resnet.BasicBlock): #i.e. already overriden
                    raise ValueError("Contradicting params: which block to use?")

            if_enable_bias = False #note that meaning of this changed from post-hoc bias addition to before-training setting of whether to learn bias
            if 'if_enable_bias' in params:
                if_enable_bias = params['if_enable_bias']
            aux_conns = None
            replace_constants_last_layer_mode = None
            replace_with_avgs_last_layer_mode = None

            if 'input_size' not in params:
                print('Warning! Setting input size to be default')
                params['input_size'] = 'default'

            if 'replace_constants_last_layer_mode' in params:
                replace_constants_last_layer_mode = params['replace_constants_last_layer_mode']

            if 'replace_with_avgs_last_layer_mode' in params:
                replace_with_avgs_last_layer_mode = params['replace_with_avgs_last_layer_mode']

            if 'if_replace_useless_conns_with_bias' in params:
                check_param_contradiction()
                block_to_use = binmatr2_multi_faces_resnet.BasicBlockBiasCreator

            if 'if_replace_useless_conns_with_additives' in params:
                check_param_contradiction()
                if 'this_is_graph_visualization_run' in params:
                    block_to_use = binmatr2_multi_faces_resnet.BasicBlockMockAdditivesUser
                    aux_conns = params['auxillary_connectivities_for_id_shortcut']
                else:
                    if not ('if_additives_user' in params):
                        block_to_use = binmatr2_multi_faces_resnet.BasicBlockAdditivesCreator
                    else:
                        block_to_use = binmatr2_multi_faces_resnet.BasicBlockMockAdditivesUser
            else:
                if 'this_is_graph_visualization_run' in params:
                    check_param_contradiction()
                    block_to_use = binmatr2_multi_faces_resnet.BasicBlockMock
                    aux_conns = params['auxillary_connectivities_for_id_shortcut']

            if 'if_store_avg_activations_for_disabling' in params:
                check_param_contradiction()
                if not ('if_additives_user' in params):
                    block_to_use = binmatr2_multi_faces_resnet.BasicBlockAvgAdditivesCreator
                else:
                    block_to_use = binmatr2_multi_faces_resnet.BasicBlockAvgAdditivesUser

            model['rep'] = binmatr2_multi_faces_resnet.BinMatrResNet(block_to_use,
                                                                     [2, 2, 2, 2], params['chunks'],
                                                                     width_mul, params['if_fully_connected'], False, 40,
                                                                     params['input_size'], aux_conns, if_enable_bias,
                                                                     replace_constants_last_layer_mode,
                                                                     replace_with_avgs_last_layer_mode)
        if arc == 'binmatr_fullconv_resnet18':
            model['rep'] = binmatr_fullconv_multi_faces_resnet.BinMatrFullConvResNet(
                binmatr_fullconv_multi_faces_resnet.BasicBlock, [2, 2, 2, 2], params['chunks'],
                width_mul, params['if_fully_connected'], False)
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
                elif arc == 'binmatr2_resnet18':
                    dim = params['chunks'][-1]
                    model[t] = binmatr2_multi_faces_resnet.FaceAttributeDecoder(dim)
                elif arc == 'binmatr_fullconv_resnet18':
                    model[t] = binmatr_fullconv_multi_faces_resnet.FaceAttributeFullConvDecoder(
                        model['rep'].connectivities[-1][int(t)].unsqueeze(0))
                else:
                    model[t] = my_resnet.FaceAttributeDecoder()

            model[t].to(device)
        return model

    if 'cifar10' in data or 'cifarfashionmnist' in data:
        model = {}
        if arc == 'binmatr2_resnet18':
            model['rep'] = binmatr2_multi_faces_resnet.BinMatrResNet(binmatr2_multi_faces_resnet.BasicBlock,
                                                                     [2, 2, 2, 2],
                                                                     params['chunks'], width_mul,
                                                                     params['if_fully_connected'], True,
                                                                     10 if 'cifar10' in data else 20)
        model['rep'].to(device)
        for t in params['tasks']:
            if arc == 'binmatr2_resnet18':
                if not data == 'cifar10_singletask':
                    model[t] = binmatr2_multi_faces_resnet.FaceAttributeDecoderCifar10()
                else:
                    dim = params['chunks'][-1]
                    model[t] = binmatr2_multi_faces_resnet.FaceAttributeDecoderCifar10SingleTask(dim)
            model[t].to(device)

        return model

    if 'cityscapes' in data:
        model = {}
        if arc == 'binmatr2_resnet50':
            raise NotImplementedError('Need to implement setting GAP size for Cityscapes input size')
            base_resnet = binmatr2_multi_faces_resnet.BinMatrResNet(binmatr2_multi_faces_resnet.Bottleneck,
                                                                    [3, 4, 6, 3], params['chunks'],
                                                                    width_mul, params['if_fully_connected'], False,
                                                                    num_tasks=3)
        model['rep'] = pspnet.ResNetDilated(base_resnet, 8)
        model['rep'].cuda()
        if 'S' in params['tasks']:
            model['S'] = pspnet.SegmentationDecoder(num_class=19, task_type='C')
            model['S'].cuda()
        if 'I' in params['tasks']:
            model['I'] = pspnet.SegmentationDecoder(num_class=2, task_type='R')
            model['I'].cuda()
        if 'D' in params['tasks']:
            model['D'] = pspnet.SegmentationDecoder(num_class=1, task_type='R')
            model['D'].cuda()
        return model
