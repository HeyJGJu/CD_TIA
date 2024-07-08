import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms, datasets
from utils import GradCAM, show_cam_on_image, center_crop_img
from resnet50 import ResModel
import cv2

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = models.mobilenet_v3_large(pretrained=True)

    pretrained_file = "/disk/sdc/OnlyChangModel/data_model_13.pth"  # 2-class
    model = ResModel()
    model.load_state_dict(torch.load(pretrained_file, map_location=device), strict=False)

    # target_layers = [model.features[-1]]
    # target_layers = [model.layer4, model.layer3, model.layer2, model.layer1]
    target_layers = [model.stage5[-1], model.stage4[-1], model.stage3[-1], model.stage2[-1], model.stage1[-1]]
    # target_layers = target_layers[0]
    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    # load image
    # img_path = "21_32.jpg"
    image_path = r"/disk/sdc/image"
    save_path = r"/disk/sdc/label"
    for image in os.listdir(image_path):
        print(image)
        img_path = os.path.join(image_path, image)
        save = os.path.join(save_path, image)
        assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
        img = Image.open(img_path).convert('RGB')
        img = np.array(img, dtype=np.uint8)
        img = center_crop_img(img,224)

        # [C, H, W]
        img_tensor = data_transform(img)
        # expand batch dimension
        # [C, H, W] -> [N, C, H, W]
        input_tensor = torch.unsqueeze(img_tensor, dim=0)

        cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)
        target_category = 0  # tabby, tabby cat
        # target_category = 254  # pug, pug-dog

        grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)

        grayscale_cam = grayscale_cam[0, :]
        visualization = show_cam_on_image(img.astype(dtype=np.float32) / 255.,
                                          grayscale_cam,
                                          use_rgb=True)

        plt.imshow(visualization)
        # plt.show()
        cv2.imwrite(save, visualization[:, :, (2, 1, 0)])
        #plt.savefig(save)

if __name__ == '__main__':
    main()

