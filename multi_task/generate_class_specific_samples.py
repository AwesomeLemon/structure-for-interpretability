"""
Based on:
(1) https://github.com/utkuozbulak/pytorch-cnn-visualizations/
(2) https://github.com/tensorflow/lucid
(3) https://github.com/greentfrapp/lucent
"""
from collections import OrderedDict

from pathlib import Path
from torch.nn.functional import softmax
from torch.optim import SGD, Adam, RMSprop
from torchvision import models

from multi_task.load_model import load_trained_model

from multi_task.util.util import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def rfft2d_freqs(h, w):
    """Computes 2D spectrum frequencies."""
    fy = np.fft.fftfreq(h)[:, None]
    # when we have an odd input dimension we need to keep one additional
    # frequency and later cut off 1 pixel
    if w % 2 == 1:
        fx = np.fft.fftfreq(w)[: w // 2 + 2]
    else:
        fx = np.fft.fftfreq(w)[: w // 2 + 1]
    return np.sqrt(fx * fx + fy * fy)

class ClassSpecificImageGenerator():
    """
        Produces an image that maximizes a certain class with gradient ascent
    """
    def __init__(self, model, target, im_size, batch_size=1, use_my_model=True):
        self.target_layer_name, self.neuron_idx = target

        self.model = model
        self.use_my_model = use_my_model

        self.size_0 = im_size[0]
        self.size_1 = im_size[1]

        self.batch_size = batch_size
        self.execute_trunk = lambda img: self.feature_extractor(img)

        if self.use_my_model:
            self.feature_extractor = self.model['rep']
            self.hook_dict = ClassSpecificImageGenerator.hook_model(self.model['rep'])
            self.feature_extractor.eval()
            for param in self.feature_extractor.parameters():
                param.requires_grad_(False)

            if self.target_layer_name == 'label':
                target_class = self.neuron_idx
                self.classifier = self.model[str(target_class)]
                self.classifier.eval()
                for param in self.classifier.parameters():
                    param.requires_grad_(False)
                if False:
                    self.irrelevant_idx = list(range(target_class)) + list(range(target_class + 1, 40))
                    self.irrelevant_idx = list(filter(lambda x: x != 10, self.irrelevant_idx))
                else:
                    self.irrelevant_idx = []
                print(self.irrelevant_idx)
                self.irrelevant_classifiers = {idx: self.model[str(idx)] for idx in self.irrelevant_idx}
                for c in self.irrelevant_classifiers.values():
                    c.eval()

                self.get_target_tensor = lambda img, trunk_features: self.classifier.linear(trunk_features[target_class])[:, 1].mean()
            else:
                self.get_target_tensor = lambda img, _: self.hook_dict[self.target_layer_name].features[:, self.neuron_idx].mean()

        else:
            model.eval()
            self.get_target_tensor = lambda _, trunk_features: trunk_features[0, self.neuron_idx]

        def inner_calculate_loss_f(self, image_transformed, trunk_features):
            output = self.get_target_tensor(image_transformed, trunk_features)
            target_weight = 1
            class_loss = -output * target_weight \
                         + 0.0005 ** 2 * torch.norm(self.image_transformed, 1) \
                         + 0.001 ** 1.5 * total_variation_loss(self.image_transformed, 2)
            return class_loss

        self.calculate_loss_f = lambda img, trunk_features: inner_calculate_loss_f(self, img, trunk_features)

        # with open('configs.json') as config_params:
        #     configs = json.load(config_params)
        # test_dst = CELEBA(root=configs['celeba']['path'], is_transform=False, split='val',
        #                   img_size=(configs['celeba']['img_rows'], configs['celeba']['img_cols']),
        #                   augmentations=None)
        # self.created_image = test_dst.__getitem__(0)[0]

        color_correlation_svd_sqrt = np.asarray([[0.26, 0.09, 0.02],
                                                 [0.27, 0.00, -0.05],
                                                 [0.27, -0.09, 0.03]]).astype("float32")
        if True:
            color_correlation_svd_sqrt = np.asarray([[0.22064119, 0.15331138, 0.10992278],
                                                    [0.15331138, 0.19750742, 0.1486874 ],
                                                    [0.10992278, 0.1486874 , 0.25062584]]).astype("float32")
            # color_correlation_svd_sqrt = np.asarray([[ 9.86012461, -7.94772474,  0.39051937],
            #                                        [-7.94772474, 15.5556668 , -5.74280639],
            #                                        [ 0.39051937, -5.74280639,  7.22573533]]).astype('float32')
            # color_correlation_svd_sqrt = np.asarray([[0.25062584, 0.1486874, 0.10992278],
            #                                          [0.1486874, 0.19750742, 0.15331138],
            #                                          [0.10992278, 0.15331138, 0.22064119]]).astype("float32")
        max_norm_svd_sqrt = np.max(np.linalg.norm(color_correlation_svd_sqrt, axis=0))
        color_correlation_normalized = color_correlation_svd_sqrt / max_norm_svd_sqrt # multiply by more => brighter image
        def _linear_decorrelate_color(tensor):
            t_permute = tensor.permute(0, 2, 3, 1)
            t_permute = torch.matmul(t_permute, torch.tensor(color_correlation_normalized.T).to(device))
            tensor = t_permute.permute(0, 3, 1, 2)
            return tensor

        self.fourier = True
        if self.fourier:
            h, w = self.size_0 *1, self.size_1 *1
            freqs = rfft2d_freqs(h, w)
            channels = 3
            init_val_size = (batch_size, channels) + freqs.shape + (2,)  # 2 for imaginary and real components
            sd = 0.5
            decay_power = 0.6

            spectrum_real_imag_t = (torch.randn(*init_val_size) * sd).to(device).requires_grad_(True)
            # spectrum_real_imag_t = (torch.FloatTensor(*init_val_size).uniform_(-0.5, 0.5)).to(device).requires_grad_(True)

            def inner(spectrum_real_imag_t):
                nonlocal decay_power
                # 2 lines below used to be outside
                scale = 1.0 / np.maximum(freqs, 1.0 / max(w, h)) ** decay_power
                scale = torch.tensor(scale).float()[None, None, ..., None].to(device)
                # decay_power *= 0.9998


                scaled_spectrum_t = scale * spectrum_real_imag_t
                image = torch.irfft(scaled_spectrum_t, 2, normalized=True, signal_sizes=(h, w))
                image = image[:batch_size, :channels, :h, :w]
                # image += 1.0 # arbitrary additive from me for making image brighter. Works on init, but colors quickly decay from human to pale
                magic = 8.0  # Magic constant from Lucid library; increasing this seems to reduce saturation
                image = image / magic
                return _linear_decorrelate_color(image)
                # return image

            self.image = spectrum_real_imag_t
            self.image_to_bgr = inner
        else:
            intensity_offset = 0
            self.image = np.random.uniform(0 + intensity_offset, 255 - intensity_offset,
                                           (batch_size, self.size_0, self.size_1, 3))
            # self.image = np.random.normal(0, 0.1, (batch_size, self.size_0, self.size_1, 3)) * 256 + 128
            if self.use_my_model:
                # means = np.array([73.15835921, 82.90891754, 72.39239876])
                means = np.array([0.38302392, 0.42581415, 0.50640459]) * 255
                # means = np.array([0.50640459, 0.42581415, 0.38302392]) * 255
            else:
                means = np.array([0.485, 0.456, 0.406]) * 255
            for i in range(3):
                self.image[:, :, :, i] -= means[i]
            self.image /= 255
            self.image = torch.from_numpy(self.image).float()
            self.image = self.image.permute(0, 3, 1, 2)
            self.image = self.image.to(device).requires_grad_(True)
            self.image_to_bgr = lambda x: _linear_decorrelate_color(x)
            # self.image_to_rgb_fun = lambda x: x

        if os.path.exists('generated'):
            shutil.rmtree('generated', ignore_errors=True)
        os.makedirs('generated')

    def normalize(self, im_as_var):
        if self.use_my_model:
            means = torch.tensor([0.38302392, 0.42581415, 0.50640459]).to(device)
            std = torch.tensor([0.2903, 0.2909, 0.3114]).to(device)
        else:
            means = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])

        return (im_as_var-means[None, ...,None,None]) / std[None, ...,None,None]

    def recreate_image2_batch(self, img):
        img = self.image_to_bgr(img)
        img = img.cpu().data.numpy()
        recreated_ims = []
        for i in range(img.shape[0]):
            recreated_im = np.copy(img[i])

            if self.use_my_model:
                # means = np.array([73.15835921, 82.90891754, 72.39239876])
                means = np.array([0.38302392, 0.42581415, 0.50640459]) * 255
                std = [0.2903, 0.2909, 0.3114]
            else:
                means = np.array([0.485, 0.456, 0.406]) * 255
                std = [0.229, 0.224, 0.225]


            # CHW -> HWC
            recreated_im = recreated_im.transpose((1, 2, 0))

            for channel in range(3):
                recreated_im[:, :, channel] *= std[channel]

            recreated_im *= 255.0
            for i in range(3):
                recreated_im[:, :, i] += means[i]

            recreated_im[recreated_im > 255.0] = 255.0
            recreated_im[recreated_im < 0.0] = 0.0

            recreated_im = np.round(recreated_im)
            recreated_im = np.uint8(recreated_im)

            # BGR to RGB:
            recreated_im = recreated_im[:, :, [2, 1, 0]]

            recreated_ims.append(recreated_im)
        return recreated_ims

    def recreate_and_save(self, img, path):
        # img = torch.sigmoid(img)
        # img = self.normalize(img)
        recreated_img = self.recreate_image2_batch(img)
        if self.batch_size == 1:
            save_image(recreated_img[0], path)
        else:
            save_image_batch(recreated_img, path)

    class ModuleHook:
        def __init__(self, module):
            self.hook = module.register_forward_hook(self.hook_fn)
            self.module = None
            self.features = None

        def hook_fn(self, module, input, output):
            #TODO: DANGER! because output is not copied, if it's modified further in the forward pass, the values here will also be modified
            #(this happens with "+= shortcut")
            self.module = module
            self.features = output

        def close(self):
            self.hook.remove()

    @staticmethod
    def hook_model(model_rep):
        features = OrderedDict()

        # recursive hooking function
        def hook_layers(net, prefix):
            if hasattr(net, "_modules"):
                for name, layer in net._modules.items():
                    if layer is None:
                        # e.g. GoogLeNet's aux1 and aux2 layers
                        continue
                    features["_".join(prefix + [name])] = ClassSpecificImageGenerator.ModuleHook(layer)
                    hook_layers(layer, prefix=prefix + [name])

        hook_layers(model_rep, [])

        # def hook(layer):
        #     return features[layer].features

        return features

    def generate(self, n_steps, if_save_intermediate=True):
        coeff_due_to_small_input = 0.35
        # coeff_due_to_small_input /= coeff_due_to_small_input #comment this line when input is small
        initial_learning_rate = 0.01 * coeff_due_to_small_input  #0.03  # 0.05 is good for imagenet, but not for celeba
        # too high lr => checkerboard patterns & acid colors, too low lr => uniform grayness
        saved_iteration_probs = {}
        self.recreate_and_save(self.image, 'generated/init.jpg')
        self.image.register_hook(normalize_grad)
        optimizer = Adam([self.image], lr=initial_learning_rate)
        # torch.autograd.set_detect_anomaly(True)
        for i in range(n_steps + 1):
            # if (i + 1) % 30 == 0:
            #     lr_multiplier = 0.9
            #     for param_group in optimizer.param_groups:
            #         param_group['lr'] *= lr_multiplier
            #     if target_weight > 1.1:
            #         target_weight *= 0.95
            if self.use_my_model:
                self.feature_extractor.zero_grad()
                if self.target_layer_name == 'label':
                    self.classifier.zero_grad()
                    for icl in self.irrelevant_classifiers.values():
                        icl.zero_grad()
            else:
                self.model.zero_grad()
            optimizer.zero_grad()

            # temp = self.processed_image.clone().cpu().detach().numpy()[0, :, :, :]
            # temp = temp.permute((1, 2, 0))

            pixels_to_pad = 12 * 2
            pixels_to_jitter1 = 8 * 2
            pixels_to_jitter2 = 4 * 2
            img_width = self.size_0
            img_height = self.size_1
            ROTATE = 10
            SCALE = 1.1

            # pad is 'reflect' because that's what's in colab
            self.image_transformed = apply_transformations(self.image, [
                self.image_to_bgr,
                # lambda img: torch.sigmoid(img),
                # self.normalize,
                lambda img: nn.functional.pad(img, [pixels_to_pad] * 4,'constant', 0.0),
                # lambda img: random_crop(img, img_width, img_height),
                lambda img: jitter_lucid(img, pixels_to_jitter1),
                # random_scale([1] * 10 + [1.005, 0.995]),
                random_scale([1 + i / 25. for i in range(-5, 6)] + 5 * [1]),  # [SCALE ** (n/10.) for n in range(-10, 11)]),
                # random_rotate([-45]),
                random_rotate(list(range(-ROTATE, ROTATE + 1)) + 5 * [0]),  # range(-ROTATE, ROTATE+1)),#,
                # blur if i % 4 == 0 else identity,
                lambda img: jitter_lucid(img, pixels_to_jitter2),
                lambda img: torch.nn.functional.interpolate(img, (self.size_0, self.size_1), mode='bicubic', align_corners=False),
                lambda img: center_crop(img, img_width, img_height),
                lambda img: pad_to_correct_shape(img, img_width, img_height)
            ])
            # self.processed_image_transformed.data.clamp_(-1, 1)
            # self.image_transformed = torch.tanh(self.image_transformed)

            # if self.use_my_model:
            #     if False:
            #         features = self.feature_extractor(self.image_transformed, None)[0]
            #         output = features[self.target_class]
            #     else:
            #         features = self.feature_extractor(self.image_transformed)
            #         output = features[self.target_class]
            #     output = self.classifier.linear(output)#[0]
            #     # output = self.classifier(output, None)[0]
            #     softmax_prob = torch.nn.functional.softmax(output, dim=1)[:, 1].mean().item()
            #     # print(output)
            #     output = output[:, 1].mean() #- output[:, 0].mean()
            #     bad_softmaxes_sum = 0.
            #     bad_output = 0.
            #     for idx, icl in self.irrelevant_classifiers.items():
            #         bad_output_cur = self.classifier(features[idx], None)[0]
            #         bad_softmax_prob_cur = torch.nn.functional.softmax(bad_output_cur, dim=1)[:, 1].mean().item()
            #         bad_softmaxes_sum += bad_softmax_prob_cur
            #         bad_output += bad_output_cur[:, 1].mean()
            #     bad_softmaxes_sum /= 38. # 40 in total - current - blurry
            # else:
            #     output = self.model(self.image_transformed)[0, self.target_class]

            features = self.execute_trunk(self.image_transformed)
            class_loss = self.calculate_loss_f(self.image_transformed, features)

            # to_print = f"Iteration {i}:\t Loss {class_loss.item():.4f} \t" \
            #            f"{softmax_prob if self.use_my_model else '':.4f} {bad_softmaxes_sum if self.use_my_model else '':.2f}".expandtabs(20)

            class_loss.backward()
            optimizer.step()

            if i % 60 == 0:  # or i < 5:
                to_print = f"Iteration {i}:\t Loss {class_loss.item():.4f} \t".expandtabs(20)
                print(to_print)
                if if_save_intermediate or i == n_steps:
                    path = 'generated/c_specific_iteration_' + str(i) + '.jpg'
                    self.recreate_and_save(self.image, path)
                # if i % 20 == 0:
                #     saved_iteration_probs[i] = softmax_prob if self.use_my_model else ''

        print(saved_iteration_probs)
        return self.image


if __name__ == '__main__':
    use_my_model = True

    if use_my_model:
        # save_model_path = r'/mnt/raid/data/chebykin/saved_models/12_21_on_November_27/optimizer=Adam|batch_size=256|lr=0.0005|dataset=celeba|normalization_type=none|algorithm=no_smart_gradient_stuff|use_approximation=True|scales={_0___0.025|__1___0.025|__2___0.025|__3___0.025|__4___0._1_model.pkl'
        # save_model_path = r'/mnt/raid/data/chebykin/saved_models/16_53_on_February_11/optimizer=Adam|batch_size=256|lr=0.005|lambda_reg=0.005|chunks=[4|_4|_4]|dataset=celeba|normalization_type=none|algorithm=no_smart_gradient_stuff|use_approximation=True|scales={_0___0.025|__1___0.025_42_model.pkl'
        # save_model_path = r'/mnt/raid/data/chebykin/saved_models/12_37_on_February_19/optimizer=Adam|batch_size=256|lr=0.0005|lambda_reg=0.001|dataset=celeba|normalization_type=none|algorithm=no_smart_gradient_stuff|use_approximation=True_6_model.pkl'
        # param_file = 'old_params/sample_all.json'
        # save_model_path = r'/mnt/raid/data/chebykin/saved_models/22_47_on_February_12/optimizer=Adam|batch_size=256|lr=0.0005|lambda_reg=0.005|chunks=[4|_4|_4]|dataset=celeba|normalization_type=none|algorithm=no_smart_gradient_stuff|use_approximation=True|scales={_0___0.025|__1___0.02_10_model.pkl'
        # param_file = 'params/bigger_reg_4_4_4.json'

        # save_model_path = r'/mnt/raid/data/chebykin/saved_models/23_05_on_April_25/optimizer=SGD_Adam|batch_size=96|lr=0.01|connectivities_lr=0.0005|chunks=[8|_8|_8]|architecture=binmatr2_resnet18|width_mul=1|weight_decay=0.0|connectivities_l1=0.0001|connectivities_l1_all=False|if__16_model.pkl'
        # param_file = 'params/binmatr2_8_8_8_sgdadam001_pretrain_condecaytask1e-4_bigimg.json'
        # save_model_path = r'/mnt/raid/data/chebykin/saved_models/23_06_on_April_26/optimizer=SGD_Adam|batch_size=52|lr=0.002|connectivities_lr=0.0005|chunks=[16|_16|_4]|architecture=binmatr2_resnet18|width_mul=1|weight_decay=0.0|connectivities_l1=0.0001|connectivities_l1_all=False|_27_model.pkl'
        # param_file = 'params/binmatr2_16_16_4_sgdadam0002_pretrain_condecaytask1e-4_biggerimg.json'
        # save_model_path = r'/mnt/raid/data/chebykin/saved_models/12_25_on_April_30/optimizer=SGD_Adam|batch_size=96|lr=0.004|connectivities_lr=0.0005|chunks=[16|_16|_4]|architecture=binmatr2_resnet18|width_mul=1|weight_decay=0.0|connectivities_l1=0.0|connectivities_l1_all=False|if__23_model.pkl'
        # param_file = 'params/binmatr2_16_16_4_sgdadam0004_pretrain_fc_bigimg.json'
        # save_model_path = r'/mnt/raid/data/chebykin/saved_models/00_22_on_June_04/optimizer=SGD_Adam|batch_size=256|lr=0.01|connectivities_lr=0.0005|chunks=[8|_8|_8|_8|_8|_8|_8|_8|_8|_8|_8|_8|_8|_8|_8]|architecture=binmatr2_resnet18|width_mul=1|weight_decay=0.0|connectivities_l1=0_45_model.pkl'
        # param_file = 'params/binmatr2_15_8s_sgdadam001+0005_pretrain_nocondecay_comeback.json'
        # save_model_path = r'/mnt/raid/data/chebykin/saved_models/14_53_on_May_26/optimizer=SGD_Adam|batch_size=96|lr=0.004|connectivities_lr=0.0005|chunks=[64|_64|_64|_128|_128|_128|_128|_256|_256|_256|_256|_512|_512|_512|_512]|architecture=binmatr2_resnet18|width_mul=1|weight_de_49_model.pkl'
        # param_file = 'params/binmatr2_filterwise_sgdadam0004_pretrain_condecayall2e-6_bigimg.json'
        # save_model_path = r'/mnt/raid/data/chebykin/saved_models/07_52_on_June_04/optimizer=SGD_Adam|batch_size=256|lr=0.01|connectivities_lr=0.0005|chunks=[8|_8|_8|_8|_8|_8|_8|_8|_8|_8|_8|_8|_8|_8|_8]|architecture=binmatr2_resnet18|width_mul=1|weight_decay=0.0|connectivities_l1=1_40_model.pkl'
        # param_file = 'params/binmatr2_15_8s_sgdadam001+0005_pretrain_condecayall1e-5_comeback.json'
        # save_model_path = r'/mnt/raid/data/chebykin/saved_models/17_35_on_May_20/optimizer=SGD_Adam|batch_size=256|lr=0.01|connectivities_lr=0.0005|chunks=[64|_64|_64|_128|_128|_128|_128|_256|_256|_256|_256|_512|_512|_512|_512]|architecture=binmatr2_resnet18|width_mul=1' \
        #                   r'|weight_de_58_model.pkl'
        # param_file = 'params/binmatr2_filterwise_sgdadam001_pretrain_condecayall2e-6.json'
        # save_model_path = r'/mnt/raid/data/chebykin/saved_models/02_16_on_May_23/optimizer=SGD_Adam|batch_size=256|lr=0.01|connectivities_lr=0.0005|chunks=[64|_64|_64|_128|_128|_128|_128|_256|_256|_256|_256|_512|_512|_512|_512]|architecture=binmatr2_resnet18|width_mul=1|weight_de_180_model.pkl'
        # param_file = 'params/binmatr2_filterwise_sgdadam001_pretrain_fc_consontop.json'
        # save_model_path = r'/mnt/raid/data/chebykin/saved_models/20_46_on_June_08/optimizer=SGD_Adam|batch_size=96|lr=0.004|connectivities_lr=0.0|chunks=[64|_64|_64|_128|_128|_128|_128|_256|_256|_256|_256|_512|_512|_512|_512]|architecture=binmatr2_resnet18|width_mul=1|weight_decay_60_model.pkl'
        # param_file = 'params/binmatr2_filterwise_sgdadam0004+0005_pretrain_bias_fc_bigimg.json'
        # save_model_path = r'/mnt/raid/data/chebykin/saved_models/20_50_on_June_17/optimizer=SGD_Adam|batch_size=96|lr=0.004|connectivities_lr=0.0005|chunks=[64|_64|_64|_128|_128|_128|_128|_256|_256|_256|_256|_512|_512|_512|_512]|architecture=binmatr2_resnet18|width_mul=1|weight_de_46_model.pkl'
        # param_file = 'params/binmatr2_filterwise_sgdadam0004+0005_pretrain_bias_fc_bigimg_consontop_condecayall1e-5.json'
        # save_model_path = r'/mnt/raid/data/chebykin/saved_models/23_30_on_June_18/optimizer=SGD_Adam|batch_size=96|lr=0.004|connectivities_lr=0.0005|chunks=[64|_64|_64|_128|_128|_128|_128|_256|_256|_256|_256|_512|_512|_512|_512]|architecture=binmatr2_resnet18|width_mul=1|weight_de_60_model.pkl'
        # param_file = 'params/binmatr2_filterwise_sgdadam0004+0005_pretrain_bias_fc_bigimg_consontop_condecayall3e-5.json'
        # save_model_path = r'/mnt/raid/data/chebykin/saved_models/22_57_on_June_19/optimizer=SGD_Adam|batch_size=96|lr=0.004|connectivities_lr=0.0005|chunks=[64|_64|_64|_128|_128|_128|_128|_256|_256|_256|_256|_512|_512|_512|_512]|architecture=binmatr2_resnet18|width_mul=1|weight_de_90_model.pkl'
        # param_file = 'params/binmatr2_filterwise_sgdadam0004+0005_pretrain_bias_condecayall3e-6_comeback_bigimg.json'
        # save_model_path = r'/mnt/raid/data/chebykin/saved_models/21_41_on_June_21/optimizer=SGD_Adam|batch_size=96|lr=0.004|connectivities_lr=0.0005|chunks=[64|_64|_64|_128|_128|_128|_128|_256|_256|_256|_256|_512|_512|_512|_512]|architecture=binmatr2_resnet18|width_mul=1|weight_de_58_model.pkl'
        # param_file = 'params/binmatr2_filterwise_sgdadam0004+0005_pretrain_bias_condecayall3e-6_comeback_bigimg.json'
        # save_model_path = r'/mnt/raid/data/chebykin/saved_models/22_07_on_June_22/optimizer=SGD_Adam|batch_size=256|lr=0.01|connectivities_lr=0.0005|chunks=[64|_64|_64|_128|_128|_128|_128|_256|_256|_256|_256|_512|_512|_512|_512]|architecture=binmatr2_resnet18|width_mul=1|weight_de_90_model.pkl'
        # param_file = 'params/binmatr2_filterwise_sgdadam001+0005_pretrain_bias_condecayall2e-6_comeback_rescaled.json'
        save_model_path = r'/mnt/raid/data/chebykin/saved_models/12_18_on_June_24/optimizer=SGD_Adam|batch_size=256|lr=0.01|connectivities_lr=0.0005|chunks=[64|_64|_64|_128|_128|_128|_128|_256|_256|_256|_256|_512|_512|_512|_512]|architecture=binmatr2_resnet18|width_mul=1|weight_de_46_model.pkl'
        param_file = 'params/binmatr2_filterwise_sgdadam001+0005_pretrain_bias_condecayall3e-6_comeback_rescaled2.json'

        trained_model = load_trained_model(param_file, save_model_path)

        with open(param_file) as json_params:
            params = json.load(json_params)
        if 'input_size' in params:
            if params['input_size'] == 'default':
                im_size = (64, 64)
            elif params['input_size'] == 'bigimg':
                im_size = (128, 128)
            elif params['input_size'] == 'biggerimg':
                im_size = (192, 168)
        else:
            if 'bigimg' in param_file:  # I remember naming my parameter files consistently
                im_size = (128, 128)
            elif 'biggerimg' in param_file:
                im_size = (192, 168)
            else:
                im_size = (64, 64)

    else:
        target_class = 130  # Flamingo
        pretrained_model = models.alexnet(pretrained=True).to(device)
        im_size = (224, 224)

    if True:
        used_neurons = np.load('actually_good_nodes.npy', allow_pickle=True).item()

    batch_size = 1
    n_steps = 1200
    if use_my_model:
        model_name_short = save_model_path[37:53] + '...' + save_model_path[-12:-10]
        for j, layer_name in list(enumerate(layers)) + [(None, 'label')]:
            if layer_name != 'label':
                cur_neurons = used_neurons[j]
                cur_neurons = [int(x[x.find('_') + 1:]) for x in cur_neurons]
            else:
                cur_neurons = range(40)
            for i in cur_neurons:
                print('Current: ', layer_name, i)
                # layer_name = 'layer3_1_conv1'#'layer1_1'#'layer1_1_conv1'#'label'#'layer1_0' #
                csig = ClassSpecificImageGenerator(trained_model, (layer_name, i), im_size, batch_size, True)
                generated = csig.generate(n_steps, False)

                imshow_path = f"generated_imshow_{model_name_short}"
                Path(imshow_path).mkdir(parents=True, exist_ok=True)
                imshow_path = f"generated_imshow_{model_name_short}/{layer_name}"
                Path(imshow_path).mkdir(parents=True, exist_ok=True)

                separate_path = f"generated_separate_{model_name_short}"
                Path(separate_path).mkdir(parents=True, exist_ok=True)
                separate_path = f"generated_separate_{model_name_short}/{layer_name}"
                Path(separate_path).mkdir(parents=True, exist_ok=True)

                copyfile(f'generated/c_specific_iteration_{n_steps}.jpg', imshow_path + f'/{i}.jpg')
                recreated = csig.recreate_image2_batch(generated)
                for j in range(batch_size):
                    save_image(recreated[j], separate_path + f'/{i}_{j}.jpg')
    else:
        csig = ClassSpecificImageGenerator(pretrained_model, target_class, im_size, batch_size, False)
        csig.generate()
