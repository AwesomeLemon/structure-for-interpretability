"""
Based on:
(1) https://github.com/utkuozbulak/pytorch-cnn-visualizations/
(2) https://github.com/tensorflow/lucid
(3) https://github.com/greentfrapp/lucent
"""
import json
from torch.nn.functional import softmax
from torch.optim import SGD, Adam, RMSprop
from torchvision import models
import kornia

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
    def __init__(self, model, target_class, im_size, batch_size=1, use_my_model=True):
        self.target_class = target_class

        self.model = model
        self.use_my_model = use_my_model

        self.size_0 = im_size[0]
        self.size_1 = im_size[1]

        self.batch_size = batch_size

        if self.use_my_model:
            self.feature_extractor = self.model['rep']
            self.feature_extractor.eval()
            for param in self.feature_extractor.parameters():
                param.requires_grad_(False)
            self.classifier = self.model[str(self.target_class)]
            self.classifier.eval()
            for param in self.classifier.parameters():
                param.requires_grad_(False)
            if False:
                self.irrelevant_idx = list(range(self.target_class)) + list(range(self.target_class + 1, 40))
                self.irrelevant_idx = list(filter(lambda x: x != 10, self.irrelevant_idx))
            else:
                self.irrelevant_idx = []
            print(self.irrelevant_idx)
            self.irrelevant_classifiers = {idx: self.model[str(idx)] for idx in self.irrelevant_idx}
            for c in self.irrelevant_classifiers.values():
                c.eval()

        else:
            model.eval()

        # with open('configs.json') as config_params:
        #     configs = json.load(config_params)
        # test_dst = CELEBA(root=configs['celeba']['path'], is_transform=False, split='val',
        #                   img_size=(configs['celeba']['img_rows'], configs['celeba']['img_cols']),
        #                   augmentations=None)
        # self.created_image = test_dst.__getitem__(0)[0]

        # self.created_image = np.ones((64, 64, 3)) * 50
        # self.created_image[5:55, 5:55, :] = np.random.uniform(0 + 70, 255 - 70, (50, 50, 3))

        color_correlation_svd_sqrt = np.asarray([[0.26, 0.09, 0.02],
                                                 [0.27, 0.00, -0.05],
                                                 [0.27, -0.09, 0.03]]).astype("float32")
        max_norm_svd_sqrt = np.max(np.linalg.norm(color_correlation_svd_sqrt, axis=0))
        color_correlation_normalized = color_correlation_svd_sqrt / max_norm_svd_sqrt
        def _linear_decorrelate_color(tensor):
            t_permute = tensor.permute(0, 2, 3, 1)
            t_permute = torch.matmul(t_permute, torch.tensor(color_correlation_normalized.T).to(device))
            tensor = t_permute.permute(0, 3, 1, 2)
            return tensor

        self.fourier = True
        if self.fourier:
            h, w = self.size_0, self.size_1
            freqs = rfft2d_freqs(h, w)
            channels = 3
            init_val_size = (batch_size, channels) + freqs.shape + (2,)  # 2 for imaginary and real components
            sd = 0.01
            decay_power = 1.0

            spectrum_real_imag_t = (torch.randn(*init_val_size) * sd).to(device).requires_grad_(True)

            scale = 1.0 / np.maximum(freqs, 1.0 / max(w, h)) ** decay_power
            scale = torch.tensor(scale).float()[None, None, ..., None].to(device)

            def inner(spectrum_real_imag_t):
                scaled_spectrum_t = scale * spectrum_real_imag_t
                image = torch.irfft(scaled_spectrum_t, 2, normalized=True, signal_sizes=(h, w))
                image = image[:batch_size, :channels, :h, :w]
                # image += 2.0 # arbitrary additive from me for making image brighter. Works on init, but colors quickly decay from human to pale
                magic = 4.0  # Magic constant from Lucid library; increasing this seems to reduce saturation
                image = image / magic
                return _linear_decorrelate_color(image)

            self.image = spectrum_real_imag_t
            self.image_to_rgb_fun = inner
        else:
            intensity_offset = 0
            self.image = np.random.uniform(0 + intensity_offset, 255 - intensity_offset,
                                           (batch_size, self.size_0, self.size_1, 3))
            # self.image = np.random.normal(0, 0.1, (batch_size, self.size_0, self.size_1, 3)) * 256 + 128
            if self.use_my_model:
                means = np.array([73.15835921, 82.90891754, 72.39239876])
            else:
                means = np.array([0.485, 0.456, 0.406]) * 255
            for i in range(3):
                self.image[:, :, :, i] -= means[i]
            self.image /= 255
            self.image = torch.from_numpy(self.image).float()
            self.image = self.image.permute(0, 3, 1, 2)
            self.image = self.image.to(device).requires_grad_(True)
            self.image_to_rgb_fun = lambda x: _linear_decorrelate_color(x)

        if os.path.exists('generated'):
            shutil.rmtree('generated', ignore_errors=True)
        os.makedirs('generated')

    # def transform_img_batch(self, img):
    #     """ [0, 255] -> [-128, 128] -> [-0.5, 0.5]
    #     Mean substraction, remap to [0,1], channel order transpose to make Torch happy
    #     """
    #     # img = img[:, :, ::-1]
    #     img = img.astype(np.float64)
    #     if self.use_my_model:
    #         means = np.array([73.15835921, 82.90891754, 72.39239876])
    #     else:
    #         means = np.array([0.485, 0.456, 0.406]) * 255
    #     for i in range(3):
    #         img[:, :, :, i] -= means[i]
    #
    #     img = torch.from_numpy(img).float()
    #     with torch.no_grad():
    #         # HWC -> CWH
    #         img = img.permute(0, 3, 1, 2)
    #         img = nn.functional.interpolate(img, size=(self.size_0, self.size_1), mode='bilinear', align_corners=True)
    #
    #         # Resize scales images from 0 to 255, thus we need
    #         # to divide by 255.0
    #         img /= 255.0
    #
    #         if not self.use_my_model:
    #             std = [0.229, 0.224, 0.225]  # if not self.use_my_model else [0.329, 0.324, 0.325]
    #             for channel in range(3):
    #                 img[:, channel] /= std[channel]
    #
    #     return img.to(device).requires_grad_(True)

    def recreate_image2_batch(self, im_as_var):
        im_as_var = im_as_var.cpu().data.numpy()
        recreated_ims = []
        for i in range(im_as_var.shape[0]):
            recreated_im = np.copy(im_as_var[i])

            if not self.use_my_model:
                std = [0.229, 0.224, 0.225]  # if not self.use_my_model else [0.329, 0.324, 0.325]
                for channel in range(3):
                    recreated_im[channel] *= std[channel]

            recreated_im = recreated_im.transpose((1, 2, 0))

            recreated_im *= 255.0
            if self.use_my_model:
                means = np.array([73.15835921, 82.90891754, 72.39239876])
            else:
                means = np.array([0.485, 0.456, 0.406]) * 255
            for i in range(3):
                recreated_im[:, :, i] += means[i]

            recreated_im[recreated_im > 255.0] = 255.0
            recreated_im[recreated_im < 0.0] = 0.0

            recreated_im = np.round(recreated_im)
            recreated_im = np.uint8(recreated_im)

            recreated_ims.append(recreated_im)
        return recreated_ims

    def recreate_and_save(self, img, path):
        img = self.image_to_rgb_fun(img)
        recreated_img = self.recreate_image2_batch(img)
        if self.batch_size == 1:
            save_image(recreated_img[0], path)
        else:
            save_image_batch(recreated_img, path)

    def generate(self):
        coeff_due_to_small_input = 0.35
        coeff_due_to_small_input /= coeff_due_to_small_input #comment this line when input is small
        initial_learning_rate = 0.01 * coeff_due_to_small_input  #0.03  # 0.05 is good for imagenet, but not for celeba
        target_weight = 1.0
        # too high lr => checkerboard patterns & acid colors, too low lr => uniform grayness
        saved_iteration_probs = {}
        self.recreate_and_save(self.image, 'generated/init.jpg')
        self.image.register_hook(normalize_grad)
        optimizer = Adam([self.image], lr=initial_learning_rate)
        for i in range(600 + 1):
            # if (i + 1) % 30 == 0:
            #     lr_multiplier = 0.9
            #     for param_group in optimizer.param_groups:
            #         param_group['lr'] *= lr_multiplier
            #     if target_weight > 1.1:
            #         target_weight *= 0.95
            if self.use_my_model:
                self.feature_extractor.zero_grad()
                self.classifier.zero_grad()
                for icl in self.irrelevant_classifiers.values():
                    icl.zero_grad()
            else:
                self.model.zero_grad()
            optimizer.zero_grad()

            # temp = self.processed_image.clone().cpu().detach().numpy()[0, :, :, :]
            # temp = temp.permute((1, 2, 0))

            pixels_to_pad = 12
            pixels_to_jitter1 = 8
            pixels_to_jitter2 = 4
            img_width = self.size_0
            img_height = self.size_1
            ROTATE = 5
            SCALE = 1.1

            # pad is 'reflect' because that's what's in colab
            self.image_transformed = apply_transformations(self.image, [
                self.image_to_rgb_fun,
                lambda img: nn.functional.pad(img, [pixels_to_pad] * 4,
                                              'constant', 0.5),
                lambda img: random_crop(img, img_width, img_height),
                lambda img: jitter_lucid(img, pixels_to_jitter1),
                # random_scale([1] * 10 + [1.005, 0.995]),
                random_scale([1 + i / 25. for i in range(-5, 6)] + 5 * [1]),  # [SCALE ** (n/10.) for n in range(-10, 11)]),
                # random_rotate([-45]),
                random_rotate(list(range(-ROTATE, ROTATE + 1)) + 5 * [0]),  # range(-ROTATE, ROTATE+1)),#,
                # blur if i % 4 == 0 else identity,
                lambda img: jitter_lucid(img, pixels_to_jitter2),
                lambda img: center_crop(img, img_width, img_height),
                lambda img: pad_to_correct_shape(img, img_width, img_height)
            ])
            # self.processed_image_transformed.data.clamp_(-1, 1)
            # self.image_transformed = torch.tanh(self.image_transformed)

            if self.use_my_model:
                if False:
                    features = self.feature_extractor(self.image_transformed, None)[0]
                    output = features[self.target_class]
                else:
                    features = self.feature_extractor(self.image_transformed)
                    output = features[self.target_class]
                output = self.classifier(output, None)[0]
                softmax_prob = torch.nn.functional.softmax(output, dim=1)[:, 1].mean().item()
                output = output[:, 1].mean()
                bad_softmaxes_sum = 0.
                bad_output = 0.
                for idx, icl in self.irrelevant_classifiers.items():
                    bad_output_cur = self.classifier(features[idx], None)[0]
                    bad_softmax_prob_cur = torch.nn.functional.softmax(bad_output_cur, dim=1)[:, 1].mean().item()
                    bad_softmaxes_sum += bad_softmax_prob_cur
                    bad_output += bad_output_cur[:, 1].mean()
                bad_softmaxes_sum /= 38. # 40 in total - current - blurry
            else:
                output = self.model(self.image_transformed)[0, self.target_class]

            # 0.00005 was a good value for total variation
            class_loss = -output * target_weight \
                         + 0.00001 * total_variation_loss(self.image_transformed, 2) \
                         # + 0.0000005 * torch.norm(self.image_transformed, 1) \
                # + 0.7 * bad_output / 38.

            # + 0.000001 * torch.norm(self.processed_image + 1, 1) #+1 only because I clamp_
            # + 0.1 * torch.norm(self.processed_image + (torch.tensor([73.15835921, 82.90891754, 72.39239876]) / 255.0)[None, :, None, None].cuda(), 1)
            # + 0.0001 * torch.norm(self.processed_image, 1) \
            # + 0.00001 * torch.norm(self.processed_image, 2) \
            # + 0.000001 * torch.norm(self.processed_image, 6) \
            to_print = f"Iteration {i}:\t Loss {class_loss.item():.4f} \t" \
                       f"{softmax_prob if self.use_my_model else '':.4f} {bad_softmaxes_sum if self.use_my_model else '':.2f}".expandtabs(20)
            print(to_print)

            class_loss.backward()
            optimizer.step()

            if i % 60 == 0:  # or i < 5:
                path = 'generated/c_specific_iteration_' + str(i) + '.jpg'
                self.recreate_and_save(self.image, path)
                if i % 20 == 0:
                    saved_iteration_probs[i] = softmax_prob if self.use_my_model else ''

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
        save_model_path = r'/mnt/raid/data/chebykin/saved_models/23_06_on_April_26/optimizer=SGD_Adam|batch_size=52|lr=0.002|connectivities_lr=0.0005|chunks=[16|_16|_4]|architecture=binmatr2_resnet18|width_mul=1|weight_decay=0.0|connectivities_l1=0.0001|connectivities_l1_all=False|_27_model.pkl'
        param_file = 'params/binmatr2_16_16_4_sgdadam0002_pretrain_condecaytask1e-4_biggerimg.json'
        # save_model_path = r'/mnt/raid/data/chebykin/saved_models/12_25_on_April_30/optimizer=SGD_Adam|batch_size=96|lr=0.004|connectivities_lr=0.0005|chunks=[16|_16|_4]|architecture=binmatr2_resnet18|width_mul=1|weight_decay=0.0|connectivities_l1=0.0|connectivities_l1_all=False|if__23_model.pkl'
        # param_file = 'params/binmatr2_16_16_4_sgdadam0004_pretrain_fc_bigimg.json'
        # save_model_path = r'/mnt/raid/data/chebykin/saved_models/00_22_on_June_04/optimizer=SGD_Adam|batch_size=256|lr=0.01|connectivities_lr=0.0005|chunks=[8|_8|_8|_8|_8|_8|_8|_8|_8|_8|_8|_8|_8|_8|_8]|architecture=binmatr2_resnet18|width_mul=1|weight_decay=0.0|connectivities_l1=0_45_model.pkl'
        # param_file = 'params/binmatr2_15_8s_sgdadam001+0005_pretrain_nocondecay_comeback.json'
        # save_model_path = r'/mnt/raid/data/chebykin/saved_models/14_53_on_May_26/optimizer=SGD_Adam|batch_size=96|lr=0.004|connectivities_lr=0.0005|chunks=[64|_64|_64|_128|_128|_128|_128|_256|_256|_256|_256|_512|_512|_512|_512]|architecture=binmatr2_resnet18|width_mul=1|weight_de_40_model.pkl'
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

    batch_size = 4
    if use_my_model:
        for i in [31]:
            csig = ClassSpecificImageGenerator(trained_model, i, im_size, batch_size, True)
            generated = csig.generate()
            # copyfile('generated/c_specific_iteration_600.jpg', f'generated_imshow_binmatr2/{i}.jpg')
            # for j in range(batch_size):
            #     recreated = csig.recreate_image2(generated[j].unsqueeze(0))
            #     save_image(recreated, f'generated_10_binmatr2/{i}_{j}.jpg')
    else:
        csig = ClassSpecificImageGenerator(pretrained_model, target_class, im_size, batch_size, False)
        csig.generate()
