import streamlit as st
import numpy as np
import pandas as pd

st.set_page_config(layout="wide")
st.title("🧪 Polynomial Coefficient Generator")
st.write("Enter your experimental data to calculate the coefficients for your optimizer.")

# --- Data Input Section ---
st.subheader("1. Input Experimental Data")
col1, col2 = st.columns(2)

with col1:
    st.info("Input your X1, X2, and Real Result (R) values below:")
    
    # Default data with floats
    default_data = {
        "X1": [-1.0, -1.0, -1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0],
        "X2": [-1.0, 0.0, 1.0, 0.0, -1.0, 1.0, 0.0, -1.0, 1.0],
        "R_real": [203627.0, 94126.0, 64203.0, 111601.0, 134818.0, 67093.0, 95949.0, 156349.0, 66080.0]
    }
    
    # Use column_config to explicitly set columns as floats (NumberColumn)
    df_input = st.data_editor(
        pd.DataFrame(default_data), 
        num_rows="dynamic",
        column_config={
            "X1": st.column_config.NumberColumn(format="%.4f"),
            "X2": st.column_config.NumberColumn(format="%.4f"),
            "R_real": st.column_config.NumberColumn(format="%.4f") # This allows any float input
        }
    )

# --- Calculation Logic ---
if st.button("Generate Coefficients"):
    # Ensure values are cast to float64 for precision
    X1 = df_input["X1"].values.astype(float)
    X2 = df_input["X2"].values.astype(float)
    R_real = df_input["R_real"].values.astype(float)
    
    # Create the high-order design matrix
    X_matrix = np.column_stack([
        np.ones(len(X1)),      # a0
        X1,                    # a1
        X2,                    # a2
        X1*X2,                 # a12
        X1**2,                 # a11
        X2**2,                 # a22
        X1**2 * X2,            # a112
        X1 * X2**2,            # a122
        X1**2 * X2**2,         # a1122
        X1**3,                 # a111
        X2**3                  # a222
    ])

    # Solve for coefficients using Least Squares
    coeffs, residuals, rank, s = np.linalg.lstsq(X_matrix, R_real, rcond=None)
    
    # Predictions and Accuracy Check
    R_pred = X_matrix.dot(coeffs)
    
    # Avoid division by zero if R_real contains 0
    with np.errstate(divide='ignore', invalid='ignore'):
        percent_diff = np.abs((R_real - R_pred) / R_real) * 100
        percent_diff = np.nan_to_num(percent_diff) 

    # --- Results Section ---
    st.divider()
    res_col1, res_col2 = st.columns([1, 2])

    with res_col1:
        st.subheader("2. Coefficients")
        st.write("Copy these into your Optimizer App:")
        labels = ['a0', 'a1', 'a2', 'a12', 'a11', 'a22', 'a112', 'a122', 'a1122', 'a111', 'a222']
        coeff_df = pd.DataFrame({"Coefficient": labels, "Value": coeffs})
        st.dataframe(coeff_df, use_container_width=True)
        
        # Formatted string for easy copying
        coeff_list = ", ".join([f"{c:.6f}" for c in coeffs])
        st.text_area("Horizontal View (for quick copy)", coeff_list)

    with res_col2:
        st.subheader("3. Model Accuracy Check")
        check_df = pd.DataFrame({
            "Real Value": R_real,
            "Predicted": R_pred,
            "Error (%)": percent_diff
        })
        st.dataframe(check_df.style.format("{:.4f}"), use_container_width=True)
        
        avg_error = np.mean(percent_diff)
        if avg_error < 1.0:
            st.success(f"Excellent Fit! Average Error: {avg_error:.6f}%")
        else:
            st.warning(f"Average Error: {avg_error:.4f}%")
