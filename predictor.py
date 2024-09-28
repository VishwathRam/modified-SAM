import json
import os
#import pandas as pd

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import copy

import torch
from torch import nn
import transformers
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from transformers import SamProcessor, SamModel

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = torch.load("./custom_sam_dev_revised_7.pt", map_location=DEVICE)
# model = SamModel.from_pretrained("facebook/sam-vit-base")
processor = SamProcessor.from_pretrained("facebook/sam-vit-large", do_normalize=True)
model.to(DEVICE)
model.eval()

annotations = []
selected_indices = []
rgb_mask = None
gray_mask = None


def hex_to_rgb(h):
    h = h[1:]
    return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))


def save_progress(pixel_id, class_id, class_label, color_map, mask, image, threshold) -> None:
    global annotations, rgb_mask, gray_mask
    color_map = hex_to_rgb(color_map)
    rgb = np.array(mask)
    mask = np.array(mask)[:, :, 0].astype(np.float32) / 255
    h, w = mask.shape
    mask = (mask > threshold).astype(np.uint8)
    
    cm_old = color_map
    color_map = np.concatenate([np.random.random(3), np.array([0.6])])
    image = np.array(image)
    image = nn.functional.interpolate(
        torch.tensor(image).permute(2, 0, 1).unsqueeze(0),
        size=(1024, 1024),
        mode='bilinear',
        align_corners=False
    ).squeeze(0).permute(1, 2, 0).cpu().numpy()
    mask_image = mask.reshape(h, w, 1) * color_map.reshape(1, 1, -1)
    print(f"Mask image shape: {mask_image.shape}")
    if rgb_mask is None:
        rgb_mask = mask_image
        rgb_mask[mask == 0, :] = np.concatenate([np.random.random(3), np.array([0.6])])
    rgb_mask[mask == 1, :] = color_map

    gray = np.array(mask)
    print(f"Sam seg: {mask.shape}")
    mask[mask == 1] = 255
    if gray_mask is None:
        gray_mask = np.zeros_like(gray)
        gray_mask[mask == 255] = pixel_id
    else:
        gray_mask[mask == 255] = pixel_id
    

    # print(np.unique(full_mask))
    annotations.append(
        {
            "class_id": class_id,
            "class_label": class_label,
            "pixel_id": pixel_id,
            "rgb_color": color_map,
        }
    )
    # selected_indices.clear()
    __import__("pprint").pprint(annotations)
    print(f"Gray mask: {gray_mask.shape}, RGB Mask: {rgb_mask.shape}")
    print(f"Data types: {gray_mask.dtype}, {image.dtype}")

    image = Image.fromarray(image)
    rgb_overlay = (rgb_mask * 255).astype(np.uint8)
    rgb_overlay = Image.fromarray(rgb_overlay)
    image.paste(rgb_overlay, mask=rgb_overlay)
    return Image.fromarray(gray_mask), image


def on_click_input_image(image, evt: gr.SelectData) -> Image:
    global selected_indices
    image = np.array(image)
    print(f"image shape: {image.shape}")
    selected_indices.append(evt.index)
    print(f"Selected indices: {selected_indices}")
    orig_size = image.shape[:2]
    sel = list()
    
    # input_points requires points in the form h, w (vertical, horizontal)
    # gradio.Image.select() gives w, h
    # orig_size is in format h, w
    for ind in selected_indices:
        # h = int (1024 * (ind[1] / orig_size[0]))
        # w = int (1024 * (ind[0] / orig_size[1]))
        h = int(1024 * (ind[0] / orig_size[1]))
        w = int(1024 * (ind[1] / orig_size[0]))
        sel.append([h, w])
    
    print(f"Selected indices: {sel}")
    
    image = nn.functional.interpolate(
        torch.tensor(image).permute(2, 0, 1).unsqueeze(0),
        size=(1024, 1024),
        mode='bilinear',
        align_corners=False
    ).squeeze(0)
    print(f"Image dtype: {image.dtype}")
    inputs = processor(image, input_points=[[sel]], return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs, multimask_output=False)
    std_size = (1024, 1024)
    pred_masks = nn.functional.interpolate(
        outputs["pred_masks"].squeeze(1),
        size=std_size,
        mode='bilinear',
        align_corners=False
    ).squeeze(0)

    pred_masks = pred_masks.squeeze(0)
    pred_masks = torch.sigmoid(pred_masks)
    pred_masks = pred_masks.cpu().numpy()
    print(f"Pred masks shape {pred_masks.shape}")

    return pred_masks


def save_meta(annotation_dir: str) -> None:
    if not os.path.exists('../annotations/'):
        os.makedirs('../annotations/')
    if not os.path.exists(f"../annotations/{annotation_dir}"):
        os.makedirs(f"../annotations/{annotation_dir}", exist_ok=True)
    with open(f"../annotations/{annotation_dir}/annot.json", "w") as fout:
        json.dump(annotations, fout)
    plt.imsave(f"../annotations/{annotation_dir}/rgb_mask.png", rgb_mask)
    plt.imsave(
        f"../annotations/{annotation_dir}/gray_mask.png", gray_mask, cmap="gray"
    )
    return None

def clear_prompt_points():
    global selected_indices
    selected_indices.clear()

with gr.Blocks() as demo:
    gr.Markdown(
        """# Image Annotation Tool Powered by SAM
## How to use?
- Click on a portion of the image which you want to segment.
- Add the required details in the Class ID and Class Label boxes.
    - Please ensure there are uniformity in what you input
- To save the current label, click on the save progress to save the label.
"""
    )
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Input Image", type="pil")
        with gr.Column():
            output_image = gr.Image(label="Masked Image", type="pil")
    with gr.Row():
        clear_pts = gr.Button(value="Clear prompt points")
        clear_pts.click(fn=clear_prompt_points)
    with gr.Row():
        with gr.Column():
            gray_image = gr.Image(label="Grayscale Masks", type="pil", height=1024, width=1024)
    with gr.Row():
        with gr.Column():
            color_image = gr.Image(label="RGB Masks", type="pil", height=1024, width=1024)
    with gr.Row():
        with gr.Column():
            pixel_id = gr.Number(value=255, label="Pixel ID")
        with gr.Column():
            class_name = gr.Textbox(label="Class Label")
        with gr.Column():
            class_id = gr.Number(label=" Class ID")
    with gr.Row():
        with gr.Column():
            color = gr.ColorPicker(label="Choose the color for the segmentation map")
        with gr.Column():
            annot_dir_name = gr.Textbox(label="Enter Annotation Folder Name")
        with gr.Column():
            threshold = gr.Slider(minimum=0.0, maximum=1.0, value=0.5, step=0.01, label="Threshold")

    input_image.select(
        fn=on_click_input_image,
        inputs=[input_image],
        outputs=[output_image],
    )
    with gr.Row():
        btn = gr.Button(value="Save Progress!")
        btn.click(
            fn=save_progress,
            inputs=[pixel_id, class_id, class_name, color, output_image, input_image, threshold],
            outputs=[gray_image, color_image],
        )

        save = gr.Button(value="Save Image/Color Maps")
        save.click(fn=save_meta, inputs=[annot_dir_name], outputs=None)


if __name__ == "__main__":
    demo.launch()