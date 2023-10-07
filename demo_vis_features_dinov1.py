import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import os
import numpy as np
from PIL import Image
import torch.nn.functional as F
from extractors import ViTExtractor
from sd_dino_utils.utils_correspondence import resize

PCA = False
CO_PCA = True
PCA_DIMS = [256, 256, 256]
#change image size to 224 to match groupvit training size for images
SIZE =224
EDGE_PAD = False

FUSE_DINO = 1
ONLY_DINO = 1
DINOV2 = False
# changed dino model to small to maintain same embedding dim, number of layers and number of heads as groupvit
MODEL_SIZE = 'small'
DRAW_DENSE = 1
DRAW_SWAP = 1
TEXT_INPUT = False
SEED = 42
#changed to zero as ODISE
TIMESTEP = 100

DIST = 'l2' if FUSE_DINO and not ONLY_DINO else 'cos'
if ONLY_DINO:
    FUSE_DINO = True

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.benchmark = True

def compute_pair_feature(files, dist='cos'):
    #chang it to 224 from 244 for dinov1 ~ to match groupvit training size for images
    img_size = 224 #if DINOV2 else 224
    model_dict={'small':'dinov2_vits14',
                'base':'dinov2_vitb14',
                'large':'dinov2_vitl14',
                'giant':'dinov2_vitg14',
                'v1small':'dino_vits16',}
    
    model_type = model_dict[MODEL_SIZE] if DINOV2 else 'dino_vits16'
    layer = 11 if DINOV2 else 9
    if 'l' in model_type:
        layer = 23
    elif 'g' in model_type:
        layer = 39
    facet = 'token' if DINOV2 else 'key'
    #change to 16
    stride = 14 if DINOV2 else 16
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # indiactor = 'v2' if DINOV2 else 'v1'
    # model_size = model_type.split('vit')[-1]
    extractor = ViTExtractor(model_type, stride, device=device)
    patch_size = extractor.model.patch_embed.patch_size[0] if DINOV2 else extractor.model.patch_embed.patch_size
    print("patch_size", patch_size)
    num_patches = int(patch_size / stride * (img_size // patch_size - 1) + 1)
    print("num_patches", num_patches)
    

    N = len(files) // 1
    result = []
    for idx in range(N):

        # Load image 1
        img1 = Image.open(files[idx]).convert('RGB')
        img1 = resize(img1, img_size, resize=True, to_pil=True, edge=EDGE_PAD)

        with torch.no_grad():
            if not CO_PCA:
                img1_batch = extractor.preprocess_pil(img1)
                img1_desc_dino = extractor.extract_descriptors(img1_batch.to(device), layer, facet)
            else:
                img1_batch = extractor.preprocess_pil(img1)
                print("img1_batch", img1_batch.shape)
                img1_desc_dino = extractor.extract_descriptors(img1_batch.to(device), layer, facet)
                print("img1_desc_dino", img1_desc_dino.shape)
                
            if dist == 'l1' or dist == 'l2':
                # normalize the features
                img1_desc_dino = img1_desc_dino / img1_desc_dino.norm(dim=-1, keepdim=True)

           
        result.append(img1_desc_dino.cpu())

    return result

def process_images(src_img_path):
    files = [src_img_path]
    print("files",files)
    result = compute_pair_feature(files,dist=DIST)
    print("result",result)
    print("result",result[0].shape)
    return result

src_img_path = "/misc/student/sharmaa/groupvit/GroupViT/demo/examples/coco.jpg"
result = process_images(src_img_path)

