import streamlit as st
import numpy as np
import pandas as pd
st.title("Matrix Transformation App (Lite Version)")

# Input coordinates
st.sidebar.header("Input Points")
num_points = st.sidebar.number_input("Number of points", 1, 10, 3)
points = []
for i in range(num_points):
    x = st.sidebar.number_input(f"X{i+1}", value=float(i))
    y = st.sidebar.number_input(f"Y{i+1}", value=float(i))
    points.append([x, y, 1])
points = np.array(points).T
choice = st.selectbox("Choose Transformation",
                      ["Translation", "Scaling", "Rotation", "Shearing", "Reflection"])
if choice == "Translation":
    tx = st.number_input("tx", value=1.0)
    ty = st.number_input("ty", value=1.0)
    T = np.array([[1, 0, tx],
                  [0, 1, ty],
                  [0, 0, 1]])
elif choice == "Scaling":
    sx = st.number_input("sx", value=2.0)
    sy = st.number_input("sy", value=2.0)
    T = np.array([[sx, 0, 0],
                  [0, sy, 0],
                  [0, 0, 1]])
elif choice == "Rotation":
    angle = np.radians(st.number_input("Angle (deg)", value=45.0))
    T = np.array([[np.cos(angle), -np.sin(angle), 0],
                  [np.sin(angle),  np.cos(angle), 0],
                  [0, 0, 1]])
elif choice == "Shearing":
    shx = st.number_input("shx", value=0.2)
    shy = st.number_input("shy", value=0.2)
    T = np.array([[1, shx, 0],
                  [shy, 1, 0],
                  [0, 0, 1]])
else:  # Reflection
    ref = st.selectbox("Reflect over:", ["X-axis", "Y-axis"])
    if ref == "X-axis":
        T = np.array([[1, 0, 0],
                      [0, -1, 0],
                      [0, 0, 1]])
    else:
        T = np.array([[-1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1]])
# Transform points
transformed = T @ points
df = pd.DataFrame({
    "x_original": points[0],
    "y_original": points[1],
    "x_transformed": transformed[0],
    "y_transformed": transformed[1]
})
st.subheader("Table of Results")
st.dataframe(df)

# minimal visualization
st.subheader("Original vs Transformed X coordinates")
st.line_chart(df[["x_original", "x_transformed"]])
st.subheader("Original vs Transformed Y coordinates")
st.line_chart(df[["y_original", "y_transformed"]])

# export
csv = df.to_csv(index=False).encode()
st.download_button(
    "Download Result CSV",
    csv,
    "coordinates.csv",
    "text/csv"
)
