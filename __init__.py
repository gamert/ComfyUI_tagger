import torch, os, json, random, hashlib
from urllib.request import urlopen
import json

#from ComfyUI_tagger.modules import deepbooru
from .modules import deepbooru
from nodes import SaveImage, LoadImage
import torch
import numpy as np
import PIL
from PIL import Image

g_ClipTagger = "masterpiece best quality girl"

def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))


class ImageTaggerDD:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"images": ("IMAGE", ), },
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
                }

    RETURN_TYPES = ("TAGGER",)
    FUNCTION = "fetch_tagger"

    CATEGORY = "DD"
    # image: tensor
    def fetch_tagger(self, images, prompt, extra_pnginfo):
        pil = tensor2pil(images)
        tag = deepbooru.model.tag(pil)
        print(tag)
        g_ClipTagger = tag
        ##res[0] = tag
        return {0: tag}


#
class LoadImage_Tagger(LoadImage):
    def __init__(self):
        self.type = "temp"
        pass

    RETURN_TYPES = ("IMAGE", "TAGGER")
    # 重载=
    def load_image(self, image):
        ret = super().load_image(image)
        img = ret[0]
        mask= ret[1]

        pil = tensor2pil(img)
        tag = deepbooru.model.tag(pil)

        print(tag)

        image_path = os.path.join(self.input_dir, image)
        results = list()
        results.append({
            "filename": image_path,
            "subfolder": "subfolder",
            "type": self.type
        })
        #如果返回的是一个dic，才可能
        ret = {0: img, 1: tag, "ui": {"images": results}}
        return ret


#("STRING", {"default": '__', "multiline": False}),
class CLIPTextEncodeTaggerDD:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"text": ("STRING", {"default": g_ClipTagger, "multiline": True}),
                             "tag": ("TAGGER", ),
                             "clip": ("CLIP", )}}
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"

    CATEGORY = "conditioning"

    def encode(self, tag, clip, text):
        text = tag if len(tag) > 0 else text
        return ([[clip.encode(text), {}]], )



NODE_CLASS_MAPPINGS = {
    "ImageTaggerDD": ImageTaggerDD,
    "CLIPTextEncodeTaggerDD": CLIPTextEncodeTaggerDD,
    #"LoadImage_Tagger": LoadImage_Tagger
    #    "PromptDD":PromptDD
}

# if __name__ == "__main__":
#     import cv2
#     from PIL import Image
#
#     img = cv2.imread('d:/sd1.5_latent_upscale.png')
#     image = Image.fromarray(img)
#     prompt = deepbooru.model.tag(image)
#     print(prompt)
#     pass
