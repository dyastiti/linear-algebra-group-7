import streamlit as st
import numpy as np
from PIL import Image
import math
import os

st.set_page_config(page_title="Matrix & Convolution Image Lab", layout="wide")

# Utility image functions 
def load_image(uploaded_file):
    if uploaded_file is None:
        return None
    image = Image.open(uploaded_file).convert("RGB")
    return np.array(image)

def pil_from_array(arr):
    return Image.fromarray(np.clip(arr, 0, 255).astype("uint8"))

def show_side_by_side(orig_arr, out_arr):
    col1, col2 = st.columns(2)
    with col1:
        st.image(pil_from_array(orig_arr), caption="Original", use_container_width=True)
    with col2:
        st.image(pil_from_array(out_arr), caption="Transformed", use_container_width=True)

#Bilinear interpolation 
def bilinear_interpolate(img, x, y):
    """
    img: (H, W, C)
    x, y: float arrays with shape (H, W)
    returns interpolated colors for each (x,y)
    """
    h, w, c = img.shape
    x0 = np.floor(x).astype(np.int32)
    x1 = x0 + 1
    y0 = np.floor(y).astype(np.int32)
    y1 = y0 + 1
    x0 = np.clip(x0, 0, w - 1)
    x1 = np.clip(x1, 0, w - 1)
    y0 = np.clip(y0, 0, h - 1)
    y1 = np.clip(y1, 0, h - 1)
    Ia = img[y0, x0]
    Ib = img[y1, x0]
    Ic = img[y0, x1]
    Id = img[y1, x1]
    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)
    wa = wa[..., None]
    wb = wb[..., None]
    wc = wc[..., None]
    wd = wd[..., None]
    return wa * Ia + wb * Ib + wc * Ic + wd * Id

#Geometric transformations
def make_translation(tx, ty):
    return np.array([[1, 0, tx],
                     [0, 1, ty],
                     [0, 0, 1]], dtype=float)
def make_scaling(sx, sy):
    return np.array([[sx, 0, 0],
                     [0, sy, 0],
                     [0, 0, 1]], dtype=float)
def make_rotation(angle, center=None):
    theta = np.radians(angle)
    cos_a, sin_a = np.cos(theta), np.sin(theta)
    R = np.array([
        [cos_a, -sin_a, 0],
        [sin_a,  cos_a, 0],
        [0, 0, 1]
    ], dtype=np.float32)
    if center is not None:
        cx, cy = center
        T1 = np.array([[1,0,-cx],[0,1,-cy],[0,0,1]])
        T2 = np.array([[1,0,cx],[0,1,cy],[0,0,1]])
        R = T2 @ R @ T1
    return R
def make_shear(shx, shy):
    return np.array([[1, shx, 0],
                     [shy, 1, 0],
                     [0, 0, 1]], dtype=float)
def make_reflection(axis="x", center=None, angle_degrees=0):
    if axis == "x":
        M = np.array([[1, 0, 0],
                      [0, -1, 0],
                      [0, 0, 1]], dtype=float)
    elif axis == "y":
        M = np.array([[-1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1]], dtype=float)
    elif axis == "y=x":
        M = np.array([[0, 1, 0],
                      [1, 0, 0],
                      [0, 0, 1]], dtype=float)
    elif axis == "angle":
        theta = math.radians(angle_degrees)
        c = math.cos(2 * theta)
        s = math.sin(2 * theta)
        M = np.array([[c, s, 0],
                      [s, -c, 0],
                      [0, 0, 1]], dtype=float)
    else:
        raise ValueError("Invalid axis for reflection")

    if center is not None:
        cx, cy = center
        return make_translation(cx, cy) @ M @ make_translation(-cx, -cy)
    return M

def apply_affine_transform_manual(img, M, output_shape=None, fill_value=0):
    h, w = img.shape[:2]
    if output_shape is None:
        out_h, out_w = h, w
    else:
        out_h, out_w = output_shape
    inv = np.linalg.inv(M)
    out = np.zeros((out_h, out_w, img.shape[2]), dtype=img.dtype)
    for y in range(out_h):
        for x in range(out_w):
            src = inv @ np.array([x, y, 1])
            xs = src[0]
            ys = src[1]

            if 0 <= xs < w-1 and 0 <= ys < h-1:
                out[y, x] = bilinear_interpolate(img, xs, ys)
            else:
                out[y, x] = fill_value
    return out

# Convolution functions #
def convolve_image_manual(image, kernel):
    """
    Manual convolution (no cv2, no scipy).
    image: H x W x 3
    kernel: k x k
    """
    if kernel.ndim != 2:
        raise ValueError("Kernel must be 2D")

    img = image.astype(float)
    h, w, c = img.shape
    kh, kw = kernel.shape
    pad_h = kh // 2
    pad_w = kw // 2

    padded = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode="edge")
    out = np.zeros_like(img, dtype=float)

    k = np.flipud(np.fliplr(kernel))

    for y in range(h):
        for x in range(w):
            region = padded[y:y + kh, x:x + kw]
            for ch in range(c):
                out[y, x, ch] = np.sum(region[:, :, ch] * k)

    return np.clip(out, 0, 255).astype("uint8")

# Streamlit Pages#
def page_home():
    st.title("Matrix & Convolution Image Lab")

    st.markdown("""
This application demonstrates **matrix-based geometric transformations** and **convolution-based image filters**.

**Goals:**
- Practice 2D matrix transformations (translation, scaling, rotation, shear, reflection).
- Understand how convolution works using blur and sharpen kernels.
- Implement everything manually using NumPy, without relying on built-in warp/blur from image libraries.
""")

    st.header("Matrix transformation (example: rotation)")
    st.latex(r"""
\begin{bmatrix}
\cos\theta & -\sin\theta & 0 \\
\sin\theta &  \cos\theta & 0 \\
0          &  0          & 1
\end{bmatrix}
""")

    st.header("Convolution (example: 3×3 blur)")
    st.latex(r"""
\frac{1}{9}
\begin{bmatrix}
1 & 1 & 1 \\
1 & 1 & 1 \\
1 & 1 & 1
\end{bmatrix}
""")

    st.header("How transformations are applied")
    st.latex(r"""
\begin{bmatrix}
x \\
y \\
1
\end{bmatrix}
=
M^{-1}
\begin{bmatrix}
x' \\
y' \\
1
\end{bmatrix}
""")

def page_tools():
    st.title("Image Processing Tools")

    uploaded = st.sidebar.file_uploader("Upload an image (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"], key="main_upload")
    image = load_image(uploaded)

    if image is None:
        st.info("Upload an image to start using the tools.")
        return

    st.sidebar.markdown("## Tool selection")
    mode = st.sidebar.selectbox("Mode", ["Geometric Transformations", "Filters (Convolution)"], key="mode_select")

    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)

    if mode == "Geometric Transformations":
        st.header("Geometric Transformations (Matrix-based)")

        transform = st.selectbox(
            "Choose transformation",
            ["Translation", "Scaling", "Rotation", "Shearing", "Reflection"],
            key="transform_select"
        )

        M = np.eye(3)

        # transformation parameter widgets (unique keys)
        if transform == "Translation":
            tx = st.slider("Translate X (pixels)", -w, w, 0, key="tx")
            ty = st.slider("Translate Y (pixels)", -h, h, 0, key="ty")
            M = make_translation(tx, ty)

        elif transform == "Scaling":
            sx = st.slider("Scale X", 0.1, 3.0, 1.0, 0.01, key="sx")
            sy = st.slider("Scale Y", 0.1, 3.0, 1.0, 0.01, key="sy")
            about_center = st.checkbox("Scale about image center", value=True, key="scale_center")
            M = make_scaling(sx, sy)
            if about_center:
                M = make_translation(center[0], center[1]) @ M @ make_translation(-center[0], -center[1])

        elif transform == "Rotation":
            angle = st.slider("Rotation angle (degrees)", -180, 180, 0, key="rot_angle")
            about_center = st.checkbox("Rotate about image center", value=True, key="rot_center")
            if about_center:
                M = make_rotation(angle, center=center)
            else:
                M = make_rotation(angle)
            transformed = apply_affine_transform_manual(image, M)
            st.write("Matrix used:", M)
            st.write("Pixel[0,0] original:", image[0,0])
            st.write("Pixel[0,0] transformed:", transformed[0,0])
            st.image(transformed, caption="Transformed Image", use_container_width=True)

        elif transform == "Shearing":
            shx = st.slider("Shear X", -1.0, 1.0, 0.0, 0.01, key="shx")
            shy = st.slider("Shear Y", -1.0, 1.0, 0.0, 0.01, key="shy")
            about_center = st.checkbox("Shear about image center", value=False, key="shear_center")
            M = make_shear(shx, shy)
            if about_center:
                M = make_translation(center[0], center[1]) @ M @ make_translation(-center[0], -center[1])

        elif transform == "Reflection":
            refl_type = st.selectbox("Reflection axis", ["x", "y", "y=x", "angle"], key="refl_type")
            if refl_type == "angle":
                angle_ref = st.slider("Axis angle (degrees)", -180, 180, 0, key="refl_angle")
                about_center = st.checkbox("Reflect about axis through center", value=True, key="refl_center")
                if about_center:
                    M = make_reflection("angle", center=center, angle_degrees=angle_ref)
                else:
                    M = make_reflection("angle", center=None, angle_degrees=angle_ref)
            else:
                about_center = st.checkbox("Reflect about axis through center", value=True, key="refl_center2")
                if about_center:
                    M = make_reflection(refl_type, center=center)
                else:
                    M = make_reflection(refl_type, center=None)

        st.subheader("Transformation matrix (3×3)")
        st.write(M)

        # Expand canvas option (UI-level) with unique key
        expand_canvas = st.checkbox("Expand output canvas", value=False, key="expand_canvas_geo")

        # Apply transform (single button)
        if st.button("Apply Transform", key="apply_transform_btn"):
            if expand_canvas:
                # compute transformed corners to determine output size
                corners = np.array([
                    [0, 0, 1],
                    [w, 0, 1],
                    [0, h, 1],
                    [w, h, 1],
                ]).T  # 3x4

                mapped = (M @ corners).T
                xs = mapped[:, 0]
                ys = mapped[:, 1]
                minx, maxx = xs.min(), xs.max()
                miny, maxy = ys.min(), ys.max()

                out_w = int(np.ceil(maxx - minx))
                out_h = int(np.ceil(maxy - miny))

                # shift so top-left is at (0,0)
                shift = make_translation(-minx, -miny)
                M_shifted = shift @ M

                out = apply_affine_transform_manual(image, M_shifted, output_shape=(out_h, out_w), fill_value=0)
            else:
                out = apply_affine_transform_manual(image, M, output_shape=(h, w), fill_value=0)

            show_side_by_side(image, out)

    elif mode == "Filters (Convolution)":
        st.header("Image Filtering (Convolution-based)")

        filter_type = st.selectbox("Filter type", ["Blur (smoothing)", "Sharpen (high-pass)"], key="filter_type")

        if filter_type == "Blur (smoothing)":
            ksize = st.slider("Kernel size (odd)", 3, 15, 3, step=2, key="blur_ksize")
            kernel = np.ones((ksize, ksize), dtype=float) / (ksize * ksize)

            st.subheader("Blur kernel")
            st.write(kernel)

            if st.button("Apply Blur", key="apply_blur"):
                out = convolve_image_manual(image, kernel)
                show_side_by_side(image, out)

        elif filter_type == "Sharpen (high-pass)":
            choice = st.radio("Sharpen kernel preset", ["Standard 3×3", "Stronger"], key="sharpen_choice")
            if choice == "Standard 3×3":
                kernel = np.array([[0, -1, 0],
                                   [-1, 5, -1],
                                   [0, -1, 0]], dtype=float)
            else:
                kernel = np.array([[-1, -1, -1],
                                   [-1, 9, -1],
                                   [-1, -1, -1]], dtype=float)

            st.subheader("Sharpen kernel")
            st.write(kernel)

            if st.button("Apply Sharpen", key="apply_sharpen"):
                out = convolve_image_manual(image, kernel)
                show_side_by_side(image, out)
    
# Team page (2 methods) #
def page_team():
    st.title("Team Members")

    st.markdown("""
This page shows:
- Group members
- Each member's role
- Member photos

You can display photos in two different ways:
1. Upload photos using Streamlit file uploader.
2. Use local image files from an `images/` folder.
""")

    method = st.radio(
        "Choose photo method:",
        ["Upload photos via file uploader", "Use photos from local folder (`images/`)"],
        key="team_method"
    )

    num_members = st.number_input("Number of team members", min_value=1, max_value=10, value=3, step=1, key="num_members")

    if method == "Upload photos via file uploader":
        st.subheader("Method 1: Upload photos")
        members_data = []

        for i in range(num_members):
            st.markdown(f"### Member {i + 1}")
            col1, col2 = st.columns([1, 4])

            with col1:
                photo_file = st.file_uploader(
                    f"Upload photo for member {i + 1}",
                    type=["png", "jpg", "jpeg"],
                    key=f"photo_upload_{i}"
                )
                if photo_file is not None:
                    photo = Image.open(photo_file).convert("RGB")
                    st.image(photo, width=150)
                else:
                    st.image(Image.new("RGB", (150, 150), color=(200, 200, 200)), width=150)

            with col2:
                name = st.text_input(f"Name of member {i + 1}", key=f"name_input_{i}")
                role = st.text_area(f"Role / contribution of member {i + 1}", key=f"role_input_{i}")

            members_data.append({
                "name": name,
                "role": role,
                "photo_file": photo_file,
            })

    else:
        st.subheader("Method 2: Local images from `images/` folder")

        st.info("""
Put your member photos in a folder named **images** in the same directory as `app.py`.
Example filenames:
- `images/member1.jpg`
- `images/member2.png`
- etc.
""")

        # default file mapping; the user can replace the files in the folder with their own
        default_members = [
            {"name": f"Member {i+1}", "role": "Role / contribution", "photo_path": f"images/member{i+1}.jpg"}
            for i in range(num_members)
        ]

        for i, m in enumerate(default_members):
            st.markdown(f"### Member {i + 1}")
            col1, col2 = st.columns([1, 3])

            with col1:
                path = m["photo_path"]
                if os.path.exists(path):
                    img = Image.open(path).convert("RGB")
                else:
                    img = Image.new("RGB", (150, 150), color=(200, 200, 200))
                st.image(img, width=150)

            with col2:
                name = st.text_input(f"Name of member {i + 1}", value=m["name"], key=f"local_name_{i}")
                role = st.text_area(f"Role / contribution of member {i + 1}", value=m["role"], key=f"local_role_{i}")

    st.subheader("How the app works (short summary)")
    st.markdown("""
- **Geometric transformations**: For each transformation, a 3×3 matrix is constructed
  (translation, scaling, rotation, shear, reflection). The app uses inverse mapping to find
  the corresponding source pixel and bilinear interpolation to estimate color values.
- **Convolution filters**: Blur and sharpen are implemented by manually sliding a kernel over
  the image and computing weighted sums for each channel.
- **Interface**: Streamlit handles file uploads, sliders for parameters, and side-by-side
  display of original vs transformed images.
""")
    
# Page router#
PAGES = {
    "Home / Introduction": page_home,
    "Image Processing Tools": page_tools,
    "Team Members": page_team,
}

def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", list(PAGES.keys()), key="page_nav")
    st.sidebar.markdown("---")
    st.sidebar.markdown("Matrix & Convolution Demo App")
    PAGES[page]()

st.cache_data.clear()
if __name__ == "__main__":
    main()
