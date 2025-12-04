# app.py
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO, StringIO
import html

st.set_page_config(page_title="2D Transform + Composite Matrix (Preview SVG + Matplotlib)", layout="wide")

def translation(tx, ty):
    return np.array([[1, 0, tx],
                     [0, 1, ty],
                     [0, 0, 1]], dtype=float)
def scaling(sx, sy):
    return np.array([[sx, 0, 0],
                     [0, sy, 0],
                     [0, 0, 1]], dtype=float)
def rotation(angle_deg):
    a = np.radians(angle_deg)
    return np.array([[np.cos(a), -np.sin(a), 0],
                     [np.sin(a),  np.cos(a), 0],
                     [0, 0, 1]], dtype=float)
def shear(shx, shy):
    return np.array([[1, shx, 0],
                     [shy, 1, 0],
                     [0, 0, 1]], dtype=float)

def reflection(axis):
    if axis == "x":
        return np.array([[1, 0, 0],
                         [0, -1, 0],
                         [0, 0, 1]], dtype=float)
    elif axis == "y":
        return np.array([[-1, 0, 0],
                         [0, 1, 0],
                         [0, 0, 1]], dtype=float)
    else:
        return np.eye(3)

def apply_matrix_to_points(pointsNx2, M):
    # points: (N,2)
    homo = np.hstack([pointsNx2, np.ones((pointsNx2.shape[0], 1))])  # (N,3)
    transformed = (M @ homo.T).T  # (N,3)
    return transformed[:, :2]

# ---------------------
# SVG generation
# ---------------------
def make_svg_preview(points_orig, points_transformed, width=360, height=360, padding=10):
    """
    returns SVG string showing original (blue) and transformed (orange) polygons/points.
    """
    # compute bounding box for both sets
    all_pts = np.vstack([points_orig, points_transformed])
    xmin, ymin = all_pts.min(axis=0)
    xmax, ymax = all_pts.max(axis=0)
    if xmax == xmin:
        xmax += 1
        xmin -= 1
    if ymax == ymin:
        ymax += 1
        ymin -= 1
    span_x = xmax - xmin
    span_y = ymax - ymin
    xmin -= 0.1 * span_x + 1e-6
    ymin -= 0.1 * span_y + 1e-6
    xmax += 0.1 * span_x + 1e-6
    ymax += 0.1 * span_y + 1e-6

    def map_pt(pt):
        x, y = pt
        sx = padding + (x - xmin) / (xmax - xmin) * (width - 2*padding)
        sy = padding + (1 - (y - ymin) / (ymax - ymin)) * (height - 2*padding)
        return sx, sy
    def points_to_str(pts):
        return " ".join(f"{map_pt(p)[0]:.2f},{map_pt(p)[1]:.2f}" for p in pts)
    svg_parts = []
    svg_parts.append(f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">')
    svg_parts.append(f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff" stroke="none"/>')
    poly_orig = points_to_str(points_orig)
    svg_parts.append(f'<polyline points="{html.escape(poly_orig)}" fill="none" stroke="#3b82f6" stroke-width="2" stroke-opacity="0.8" />')
    
    # original points
    for i, p in enumerate(points_orig):
        sx, sy = map_pt(p)
        svg_parts.append(f'<circle cx="{sx:.2f}" cy="{sy:.2f}" r="4" fill="#1e40af"/>')
        svg_parts.append(f'<text x="{sx+6:.2f}" y="{sy-6:.2f}" font-size="10" fill="#1e40af">P{i}</text>')

    # transformed polygon/line (orange)
    poly_tr = points_to_str(points_transformed)
    svg_parts.append(f'<polyline points="{html.escape(poly_tr)}" fill="none" stroke="#f97316" stroke-width="2" stroke-opacity="0.9" />')
    # transformed points
    for i, p in enumerate(points_transformed):
        sx, sy = map_pt(p)
        svg_parts.append(f'<circle cx="{sx:.2f}" cy="{sy:.2f}" r="4" fill="#c2410c"/>')
        svg_parts.append(f'<text x="{sx+6:.2f}" y="{sy-6:.2f}" font-size="10" fill="#c2410c">P{i}\'</text>')

    svg_parts.append('</svg>')
    return "\n".join(svg_parts)

# ---------------------
# Streamlit UI
# ---------------------
st.title("2D Transformations (Homogeneous 3×3) — Preview SVG + Matplotlib")

st.markdown("This application demonstrates translation, scaling, rotation, shearing, and reflection using 3×3 homogeneous matrices. A quick preview is shown with SVG, while full visualization is provided through Matplotlib with grids and labels.")

# Sidebar - points and transform
st.sidebar.header("Points (enter as x,y ; x,y ; ...)")
default_points = "0,0; 2,1; 1,3; 0,0"  # closed loop example
raw = st.sidebar.text_area("Points", value=default_points, height=120)
try:
    pts = np.array([ [float(s.split(",")[0]), float(s.split(",")[1])] for s in raw.split(";") if s.strip() != "" ])
    if pts.shape[0] < 1:
        raise ValueError("Need at least one point")
except Exception as e:
    st.sidebar.error("Invalid points format. Use: x,y; x,y; ...")
    st.stop()

st.sidebar.markdown("---")
st.sidebar.header("Transformation to build (single)")
choice = st.sidebar.selectbox("Transformation", ["Translation", "Scaling", "Rotation", "Shearing", "Reflection"])

# Build the single transform matrix
if choice == "Translation":
    tx = st.sidebar.number_input("tx", value=1.0, key="tx")
    ty = st.sidebar.number_input("ty", value=0.0, key="ty")
    T = translation(tx, ty)
elif choice == "Scaling":
    sx = st.sidebar.number_input("sx", value=1.0, key="sx")
    sy = st.sidebar.number_input("sy", value=1.0, key="sy")
    T = scaling(sx, sy)
elif choice == "Rotation":
    ang = st.sidebar.slider("Angle (deg)", -180.0, 180.0, 0.0, key="rot")
    T = rotation(ang)
elif choice == "Shearing":
    shx = st.sidebar.number_input("shx", value=0.0, key="shx")
    shy = st.sidebar.number_input("shy", value=0.0, key="shy")
    T = shear(shx, shy)
else:
    axis = st.sidebar.selectbox("Axis", ["x", "y"], key="reflaxis")
    T = reflection(axis)

# Composite matrix controls
st.sidebar.markdown("---")
st.sidebar.header("Composite Matrix")
if "composite" not in st.session_state:
    st.session_state.composite = np.eye(3)
st.sidebar.write("Current composite (3×3):")
st.sidebar.write(pd.DataFrame(st.session_state.composite))

if st.sidebar.button("Add single transform to composite"):
    # Note: left-multiply to apply single then previous composite when using column vectors
    st.session_state.composite = T @ st.session_state.composite

if st.sidebar.button("Reset composite"):
    st.session_state.composite = np.eye(3)

# Apply transforms
single_out = apply_matrix_to_points(pts, T)
composite_out = apply_matrix_to_points(pts, st.session_state.composite)

# Results table + download
df = pd.DataFrame({
    "x_original": pts[:,0],
    "y_original": pts[:,1],
    "x_single": single_out[:,0],
    "y_single": single_out[:,1],
    "x_composite": composite_out[:,0],
    "y_composite": composite_out[:,1],
})
st.subheader("Numeric Results")
st.dataframe(df)

csv_bytes = df.to_csv(index=False).encode()
st.download_button("Download coordinates (CSV)", csv_bytes, file_name="transform_results.csv")

# Layout previews
col_svg, col_plot = st.columns([1, 1])

with col_svg:
    st.subheader("Quick SVG Preview")
    svg = make_svg_preview(pts, composite_out, width=420, height=420)
    # st.markdown supports raw svg if allow_html; better to use components.html to avoid escaping
    st.components.v1.html(svg, height=440)

with col_plot:
    st.subheader("Matplotlib visualization (grid & labels)")
    fig, ax = plt.subplots(figsize=(6,6))
    # draw grid lines
    # choose bounds based on combined points
    allpts = np.vstack([pts, composite_out])
    xmin, ymin = allpts.min(axis=0) - 1
    xmax, ymax = allpts.max(axis=0) + 1
    # nice symmetric bounds
    xpad = max(1, (xmax - xmin) * 0.1)
    ypad = max(1, (ymax - ymin) * 0.1)
    ax.set_xlim(xmin - xpad, xmax + xpad)
    ax.set_ylim(ymin - ypad, ymax + ypad)
    ax.set_aspect("equal", adjustable="box")

    # grid
    ax.grid(True, which='major', linestyle='--', alpha=0.5)

    # plot original polygon/points
    ax.plot(pts[:,0], pts[:,1], marker='o', color='tab:blue', label='Original')
    for i,p in enumerate(pts):
        ax.text(p[0]+0.05, p[1]+0.05, f"P{i}", color='tab:blue', fontsize=9)

    # plot single transform (optional faint)
    show_single = st.checkbox("Show single transform result (faint)", value=False)
    if show_single:
        ax.plot(single_out[:,0], single_out[:,1], marker='s', linestyle=':', color='tab:green', label='Single transform (dotted)')
        for i,p in enumerate(single_out):
            ax.text(p[0]+0.05, p[1]+0.05, f"P{i}¹", color='tab:green', fontsize=9)

    # plot composite transform
    ax.plot(composite_out[:,0], composite_out[:,1], marker='o', linestyle='-', color='tab:orange', label='Composite')
    for i,p in enumerate(composite_out):
        ax.text(p[0]+0.05, p[1]+0.05, f"P{i}\'", color='tab:orange', fontsize=9)

    ax.legend()
    st.pyplot(fig)

st.caption("SVG preview is intentionally lightweight (instant). Matplotlib provides a richer plot with grid and labels for analysis.")
