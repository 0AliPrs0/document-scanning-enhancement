import os
import sys
import cv2
import math
import torch
import numpy as np
import streamlit as st

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model_enhancement import DocumentEnhancementUNet
from model_corners import CornerHeatmapUNet, CornerDirectRegressor


def extract_coords(logits):
    B, C, H, W = logits.shape

    flat = logits.view(B, C, -1)
    max_idx = torch.argmax(flat, dim=2)

    y_coords = (max_idx // W).float() / (H - 1)
    x_coords = (max_idx % W).float() / (W - 1)

    return torch.stack([x_coords, y_coords], dim=2)


def load_weights(model, path, device):
    state = torch.load(path, map_location=device)
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    return model


def preprocess_image(img, target_size=512):
    h, w = img.shape[:2]

    scale = min(target_size / w, target_size / h)

    nw = max(1, int(w * scale))
    nh = max(1, int(h * scale))

    canvas = np.ones((target_size, target_size, 3), dtype=np.uint8) * 255

    x_off = (target_size - nw) // 2
    y_off = (target_size - nh) // 2

    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)

    canvas[y_off : y_off + nh, x_off : x_off + nw] = resized

    return canvas, scale, x_off, y_off


def validate_corners(corners, image_shape):
    h, w = image_shape[:2]

    corners = corners.copy()

    corners[:, 0] = np.clip(corners[:, 0], 0, w - 1)

    corners[:, 1] = np.clip(corners[:, 1], 0, h - 1)

    tl, tr, br, bl = corners

    valid = True

    if tl[0] >= tr[0]:
        valid = False

    if bl[0] >= br[0]:
        valid = False

    if tl[1] >= bl[1]:
        valid = False

    if tr[1] >= br[1]:
        valid = False

    return corners, valid


def draw_corners_on_image(img, corners):
    vis_img = img.copy()

    pts = corners.astype(np.int32).reshape((-1, 1, 2))

    cv2.polylines(vis_img, [pts], isClosed=True, color=(0, 255, 0), thickness=3)

    for i, corner in enumerate(corners):
        cx = int(corner[0])
        cy = int(corner[1])

        cv2.circle(vis_img, (cx, cy), 8, (255, 0, 0), -1)

        cv2.putText(
            vis_img,
            str(i),
            (cx + 10, cy - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return vis_img


@st.cache_resource
def load_pipeline_models(corner_method, use_dropout):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    drop_name = "dropout" if use_dropout else "nodrop"

    if corner_method == "Heatmap":
        corner_model = CornerHeatmapUNet(use_dropout=use_dropout).to(device)

        corner_path = os.path.join("output", f"best_heatmap_{drop_name}.pth")
    else:
        corner_model = CornerDirectRegressor(use_dropout=use_dropout).to(device)

        corner_path = os.path.join("output", f"best_reg_{drop_name}.pth")

    enhancement_model = DocumentEnhancementUNet(use_dropout=use_dropout).to(device)

    enhancement_path = os.path.join("output", f"best_enhancement_{drop_name}.pth")

    if not os.path.exists(corner_path):
        raise FileNotFoundError(f"Corner weights not found:\n{corner_path}")

    if not os.path.exists(enhancement_path):
        raise FileNotFoundError(f"Enhancement weights not found:\n{enhancement_path}")

    corner_model = load_weights(corner_model, corner_path, device)

    enhancement_model = load_weights(enhancement_model, enhancement_path, device)

    corner_model.eval()
    enhancement_model.eval()

    return (corner_model, enhancement_model, device, corner_path, enhancement_path)


st.set_page_config(
    page_title="Document Scanner", layout="wide", initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
    .main {
        background-color: #f4f4f9;
    }

    .stButton>button {
        width: 100%;
        border-radius: 4px;
        font-weight: 600;
        background-color: #0056b3;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
    }

    .stButton>button:hover {
        background-color: #004494;
        color: white;
    }

    h1, h2, h3 {
        color: #333;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


with st.sidebar:
    st.title("Scanner Configuration")

    st.subheader("Corner Detection")

    corner_method = st.radio("Method", ["Heatmap", "Regression"])

    use_dropout = st.checkbox("Use Dropout", value=False)

    st.subheader("Selected Models")

    drop_name = "dropout" if use_dropout else "nodrop"

    if corner_method == "Heatmap":
        selected_corner_path = f"output/best_heatmap_{drop_name}.pth"
    else:
        selected_corner_path = f"output/best_reg_{drop_name}.pth"

    selected_enh_path = f"output/best_enhancement_{drop_name}.pth"

    st.code(f"Corner:\n{selected_corner_path}\n\n" f"Enhancement:\n{selected_enh_path}")

    st.subheader("Input")

    uploaded_file = st.file_uploader("Upload Document", type=["jpg", "jpeg", "png"])


st.title("Document Scanner Pipeline")


if uploaded_file is None:
    st.info("Please upload an image using the sidebar.")
    st.stop()


file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)

original_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

if original_img is None:
    st.error("Could not read uploaded image.")
    st.stop()

original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)


col_img, col_empty = st.columns([1, 2])

with col_img:
    st.image(original_img, caption="Original Image", use_container_width=True)


if st.button("Process Image"):

    try:
        with st.spinner("Loading models..."):

            corner_model, enhancement_model, device, corner_path, enhancement_path = (
                load_pipeline_models(corner_method, use_dropout)
            )

        with st.spinner("Processing image..."):

            input_size = 512

            canvas, scale, x_off, y_off = preprocess_image(
                original_img, target_size=input_size
            )

            input_tensor = (
                torch.from_numpy(canvas.transpose(2, 0, 1))
                .float()
                .unsqueeze(0)
                .to(device)
                / 255.0
            )

            with torch.no_grad():

                corner_output = corner_model(input_tensor)

                if corner_method == "Regression":
                    norm_corners = corner_output[0].cpu().numpy().reshape(4, 2)
                else:
                    norm_corners = extract_coords(corner_output)[0].cpu().numpy()

            corners_canvas = norm_corners * (input_size - 1)

            corners_original = (
                corners_canvas - np.array([x_off, y_off], dtype=np.float32)
            ) / scale

            corners_original = corners_original.astype(np.float32)

            corners_original, is_valid = validate_corners(
                corners_original, original_img.shape
            )

            if not is_valid:
                st.warning("Invalid corner prediction detected.")

            vis_img = draw_corners_on_image(original_img, corners_original)

            tl, tr, br, bl = corners_original

            top_width = max(int(np.linalg.norm(tr - tl)), 1)

            bottom_width = max(int(np.linalg.norm(br - bl)), 1)

            left_height = max(int(np.linalg.norm(bl - tl)), 1)

            right_height = max(int(np.linalg.norm(br - tr)), 1)

            target_width = max(top_width, bottom_width)

            target_height = max(left_height, right_height)

            destination = np.array(
                [
                    [0, 0],
                    [target_width - 1, 0],
                    [target_width - 1, target_height - 1],
                    [0, target_height - 1],
                ],
                dtype=np.float32,
            )

            H = cv2.getPerspectiveTransform(corners_original, destination)

            rectified = cv2.warpPerspective(
                original_img, H, (target_width, target_height)
            )

            patch_size = 512

            pad_h = math.ceil(target_height / patch_size) * patch_size

            pad_w = math.ceil(target_width / patch_size) * patch_size

            padded_rect = np.ones((pad_h, pad_w, 3), dtype=np.uint8) * 255

            padded_rect[:target_height, :target_width] = rectified

            enhanced_canvas = np.zeros((pad_h, pad_w, 3), dtype=np.uint8)

            total_patches = (pad_h // patch_size) * (pad_w // patch_size)

            current_patch = 0

            progress_bar = st.progress(0)

            for y in range(0, pad_h, patch_size):

                for x in range(0, pad_w, patch_size):

                    patch = padded_rect[y : y + patch_size, x : x + patch_size]

                    patch_tensor = (
                        torch.from_numpy(patch.transpose(2, 0, 1))
                        .float()
                        .unsqueeze(0)
                        .to(device)
                        / 255.0
                    )

                    with torch.no_grad():
                        out_tensor = enhancement_model(patch_tensor)

                    out_patch = out_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)

                    out_patch = (out_patch * 255.0).clip(0, 255).astype(np.uint8)

                    enhanced_canvas[y : y + patch_size, x : x + patch_size] = out_patch

                    current_patch += 1

                    progress_bar.progress(current_patch / total_patches)

            final_enhanced = enhanced_canvas[:target_height, :target_width]

        st.success("Processing Complete")

        st.subheader("Detected Corners")

        st.image(vis_img, caption="Detected Corners", use_container_width=True)

        st.subheader("Pipeline Results")

        res_col1, res_col2 = st.columns(2)

        with res_col1:
            st.image(rectified, caption="Rectified", use_container_width=True)

        with res_col2:
            st.image(final_enhanced, caption="Enhanced", use_container_width=True)

    except Exception as e:
        st.error(f"Error during processing: {str(e)}")
