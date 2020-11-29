import os

imagenet_train_path = '/mnt/raid/data/ni/dnn/imagenet2012/train'
for filename in os.listdir(imagenet_train_path):
    class_name = filename[:filename.find('_')]
    dir_path = os.path.join(imagenet_train_path, class_name)
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)
        # print(dir_path)
    file_path = os.path.join(dir_path, filename)
    # print(file_path)
    os.rename(os.path.join(imagenet_train_path, filename), file_path)