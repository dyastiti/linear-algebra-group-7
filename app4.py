import streamlit as st
import numpy as np
import pandas as pd

st.title("2D Matrix Transformation using Homogeneous Coordinates")

# initialize composite matrix
if "composite" not in st.session_state:
    st.session_state.composite = np.eye(3)

st.sidebar.header("Input Points")
num_points = st.sidebar.number_input("Number of points", 1, 10, 3)

points = []
for i in range(num_points):
    x = st.sidebar.number_input(f"X{i+1}", value=float(i))
    y = st.sidebar.number_input(f"Y{i+1}", value=float(i))
    points.append([x, y, 1])

points = np.array(points).T

st.subheader("Define Transformation")

choice = st.selectbox("Choose Transformation:",
                      ["Translation", "Scaling", "Rotation", "Shearing", "Reflection"])

# Build transformation matrix
if choice == "Translation":
    tx = st.number_input("tx", value=1.0)
    ty = st.number_input("ty", value=1.0)
    T = np.array([
        [1, 0, tx],
        [0, 1, ty],
        [0, 0, 1]
    ])

elif choice == "Scaling":
    sx = st.number_input("sx", value=2.0)
    sy = st.number_input("sy", value=2.0)
    T = np.array([
        [sx, 0, 0],
        [0, sy, 0],
        [0, 0, 1]
    ])

elif choice == "Rotation":
    angle = np.radians(st.number_input("Angle (degrees)", value=45.0))
    T = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1]
    ])

elif choice == "Shearing":
    shx = st.number_input("Shear X", value=0.2)
    shy = st.number_input("Shear Y", value=0.2)
    T = np.array([
        [1, shx, 0],
        [shy, 1, 0],
        [0, 0, 1]
    ])

else:  # Reflection
    plane = st.selectbox("Reflect over:", ["X-axis", "Y-axis"])
    if plane == "X-axis":
        T = np.array([
            [1, 0, 0],
            [0,-1, 0],
            [0, 0, 1]
        ])
    else:
        T = np.array([
            [-1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])

st.write("### Single Transformation Matrix")
st.write(pd.DataFrame(T))

# Apply transformation directly
single_output = T @ points

# Update composite transformation
if st.button("Add to Composite"):
    st.session_state.composite = T @ st.session_state.composite

st.write("### Composite Matrix (Accumulated Transformations)")
st.write(pd.DataFrame(st.session_state.composite))

# Apply composite transform
composite_output = st.session_state.composite @ points

# Display result
df = pd.DataFrame({
    "x_original": points[0],
    "y_original": points[1],
    "x_single": single_output[0],
    "y_single": single_output[1],
    "x_composite": composite_output[0],
    "y_composite": composite_output[1]
})

st.subheader("Transformation Result Table")
st.dataframe(df)

# Export output
csv = df.to_csv(index=False).encode()
st.download_button("Download Output Coordinates", csv, "output.csv", "text/csv")

if st.button("Reset Composite Matrix"):
    st.session_state.composite = np.eye(3)
    st.experimental_rerun()
