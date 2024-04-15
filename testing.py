import torch
import torch.nn as nn
from models import build_classification_model
from dataloader import ChestXray14Dataset, build_transform_classification
from trainer import test_classification
from tqdm import tqdm
import timm
import collections

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the arguments used for training
checkpoint = torch.load("./Models/Classification/ChestXray14/swin_base_imagenet_21kSwinV1-21K-Base_ChestX-ray14/swin_base_imagenet_21kSwinV1-21K-Base_ChestX-ray14_run_1.pth.tar", map_location=device)
# print(type(checkpoint['state_dict']))
# args = checkpoint["args"]

temp = collections.OrderedDict()
for x in checkpoint['state_dict'].keys():
    temp[x[7:]] = checkpoint['state_dict'][x]

checkpoint['state_dict'] = temp

# Build the model
# model = build_classification_model(args)
model = timm.create_model('swin_base_patch4_window7_224_in22k', num_classes=14, pretrained=True)
model = model.to(device)

# Load the model weights
model.load_state_dict(checkpoint["state_dict"])

# Enable gradient checkpointing
# model.apply(lambda m: set_gradient_checkpointing(m, value=True))

# Enable model parallelism if available
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

# Set the model to evaluation mode
model.eval()

# Load the test dataset
# if args.model_name != "swinv2_base_192" and args.model_name != "swinv2_base_256" else args.img_size
image_size = 224
test_transform = build_transform_classification(normalize="chestx-ray", crop_size=image_size, mode="test")
test_dataset = ChestXray14Dataset(images_path="/scratch/hmudigon/datasets/ssl/ChestX-ray14/images", file_path="dataset/Xray14_test_official.txt", augment=test_transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False,
                                          num_workers=16, pin_memory=True)

# Test the model
# y_test, p_test = test_classification(None, test_loader, device, args)

if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)
model.to(device)

model.eval()

y_test = torch.FloatTensor().cuda()
p_test = torch.FloatTensor().cuda()

with torch.no_grad():
    for i, (samples, targets) in enumerate(tqdm(test_loader)):
        targets = targets.cuda()
        y_test = torch.cat((y_test, targets), 0)

        if len(samples.size()) == 4:
            bs, c, h, w = samples.size()
            n_crops = 1
        elif len(samples.size()) == 5:
            bs, n_crops, c, h, w = samples.size()

        varInput = torch.autograd.Variable(samples.view(-1, c, h, w).cuda())

        out = model(varInput)
        # if args.data_set == "RSNAPneumonia":
        #     out = torch.softmax(out,dim = 1)
        # else:
        out = torch.sigmoid(out)
        outMean = out.view(bs, n_crops, -1).mean(1)
        p_test = torch.cat((p_test, outMean.data), 0)

        # print(y_test, outMean.data)

# Compute and print the accuracy
accuracy = (y_test.cpu().numpy() == p_test.cpu().numpy().round()).mean()
print(f"Test Accuracy: {accuracy:.4f}")