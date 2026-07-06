# app.py
import streamlit as st
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from torch_geometric.datasets import JODIEDataset
from model import TGNLinkPredictor
import time

# Premium dark/clean layout configuration
st.set_page_config(page_title="TGN Research Implementation Dashboard", page_icon="🔬", layout="wide")

# --- HEADER SECTION ---
st.title("🔬 Temporal Graph Networks (TGN) Interactive Framework")
st.markdown("""
This production-grade dashboard demonstrates the complete implementation of the state-of-the-art **TGN architecture** (Rossi et al., Twitter Research) on continuous-time dynamic graphs. 
""")

# --- LOAD DATA IN BACKGROUND FOR COLD METRICS ---
@st.cache_resource
def load_base_data():
    dataset = JODIEDataset(root='data/', name='wikipedia')
    return dataset[0]

full_data = load_base_data()
num_nodes_total = max(int(full_data.src.max()), int(full_data.dst.max())) + 1

# --- TAB CONTROL LAYOUT ---
tab1, tab2, tab3 = st.tabs(["📘 Framework Architecture & Theory", "📊 Dynamic Dataset Explorer", "🚀 Live Training Engine"])

# ==========================================
# TAB 1: RESEARCH THEORY & MATHEMATICS
# ==========================================
with tab1:
    st.subheader("💡 Core Architecture & Mathematical Design")
    st.write("How this implementation translates the raw mathematical equations from the PDF into working code:")
    
    t_col1, t_col2 = st.columns(2)
    with t_col1:
        with st.container(border=True):
            st.markdown("### 1. Dynamic Node Memory Vector ($s_i$)")
            st.write("""
            **The Concept:** Traditional GNNs treat graphs as static. TGN maintains a persistent, updated 'diary' vector 
            for every single node that summarizes its unique historical timeline.
            """)
            st.info("💡 **Code Equivalent:** Initialized as a static parameter array `TGNMemory(num_nodes, memory_dim=172)`.")
            
        with st.container(border=True):
            st.markdown("### 2. Eliminating Memory Staleness ($\Delta t$)")
            st.write("""
            **The Concept:** If Node A hasn't interacted in 3 hours, its memory state is stale. The model fixes this 
            by calculating the exact continuous time delta ($\Delta t = t - \text{last\_update}$) and encoding it into vector space.
            """)
            st.info("🔢 **Mathematical Operator:** Processed live via our custom `TimeEncoder` module.")

    with t_col2:
        with st.container(border=True):
            st.markdown("### 3. Anti-Information Leakage Loop")
            st.warning("**Crucial Presentation Point:** Why our model evaluation is 100% scientifically honest.")
            st.write("""
            Section 3.2 of the paper highlights a dangerous pitfall: if you update a node's memory *before* predicting its 
            current link, your model looks into the future and leaks data.
            
            Our execution logic strictly reads the existing memory state, calculates the loss, and **only then** pushes the 
            current features back to update the memory module for subsequent chronological batches.
            """)

# ==========================================
# TAB 2: DATASET EXPLORER
# ==========================================
with tab2:
    st.subheader("📋 Dataset Inspection: JODIE Wikipedia Dynamic Graph")
    st.write("This tab allows your seniors to inspect the real-world sequence data streaming through the network.")
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Chronological Interactions", f"{full_data.num_events:,}")
    m2.metric("Unique Graph Entities (Nodes)", f"{num_nodes_total:,}")
    m3.metric("Edge Embedding Dimension", f"{full_data.msg.size(1)} Features")

    st.write("#### 🔎 Sample Interaction Sequence (First 5 Rows)")
    sample_df = pd.DataFrame({
        "Source User ID (src)": full_data.src[:5].numpy(),
        "Destination Page ID (dst)": full_data.dst[:5].numpy(),
        "Continuous Timestamp (t)": full_data.t[:5].numpy(),
    })
    st.dataframe(sample_df, use_container_width=True)

# ==========================================
# TAB 3: LIVE TRAINING ENGINE
# ==========================================
with tab3:
    st.subheader("🏎️ Execution Controls & Live Metrics Convergence")
    
    # Dashboard configuration panels
    c_col1, c_col2, c_col3 = st.columns([1, 1, 2])
    with c_col1:
        epochs = st.slider("Training Epochs", 1, 10, 5, help="Controls complete iterations over the sequence.")
        batch_size = st.select_slider("Batch Size Window", options=[50, 100, 200, 500], value=200)
    with c_col2:
        learning_rate = st.selectbox("Adam Learning Rate", [0.001, 0.0005, 0.0001], index=2)
        speed_mode = st.checkbox("Enable Presentation Fast-Track", value=True, 
                                 help="Slices the stream data down to 10k interactions for a rapid 10-second live presentation.")
    with c_col3:
        st.write("#### 🚀 Deployment Recommendation")
        st.markdown("""
        Keep **Presentation Fast-Track enabled** during live slide reviews to demonstrate convergence behaviors instantly on CPU architectures. 
        Uncheck it to train on all 157k chronological interactions.
        """)
        run_btn = st.button("▶️ Execute Live Optimization Loop", type="primary", use_container_width=True)

    if run_btn:
        st.divider()
        
        # Hardware Targeting
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Apply Data Slicing Optimization
        max_events = 10000 if speed_mode else full_data.num_events
        
        class DataSlice: pass
        data = DataSlice()
        data.src = full_data.src[:max_events].to(device)
        data.dst = full_data.dst[:max_events].to(device)
        data.t = full_data.t[:max_events].to(device)
        data.msg = full_data.msg[:max_events].to(device)
        data.num_events = max_events

        num_nodes = max(int(data.src.max()), int(data.dst.max())) + 1
        edge_feat_dim = data.msg.size(1)

        train_end = int(data.num_events * 0.70)
        val_end = int(data.num_events * 0.85)

        # Initialize TGN Model Pipeline
        model = TGNLinkPredictor(num_nodes=num_nodes, edge_feat_dim=edge_feat_dim).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = torch.nn.BCEWithLogitsLoss()

        # Visual placeholders for live telemetry updates
        status_box = st.info("Initializing neural weights and clearing sequence registers...")
        p_bar = st.progress(0)
        
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.write("**Validation Average Precision (AP %)**")
            ap_chart = st.line_chart()
        with col_c2:
            st.write("**Training Loss Optimization Curve**")
            loss_chart = st.line_chart()

        # Chronological metrics storage dictionaries
        epoch_idx = []
        ap_history = []
        loss_history = []

        # Start Execution
        for epoch in range(1, epochs + 1):
            status_box.text(f"Processing Chronological Steps | Epoch {epoch}/{epochs}...")
            model.train()
            model.memory.reset_state() 
            total_loss = 0
            batch_count = 0
            
            for i in range(0, train_end, batch_size):
                optimizer.zero_grad()
                b_src = data.src[i:i+batch_size]
                b_dst = data.dst[i:i+batch_size]
                b_t = data.t[i:i+batch_size]
                b_msg = data.msg[i:i+batch_size]

                # Compute predictions on current historical memories
                pred = model(b_src, b_dst, b_t)
                
                # Strict Uniform Random Negative Sampling
                b_neg_dst = torch.randint(0, num_nodes, (len(b_src),), device=device)
                neg_pred = model(b_src, b_neg_dst, b_t)

                pos_label = torch.ones_like(pred)
                neg_label = torch.zeros_like(neg_pred)
                loss = criterion(pred, pos_label) + criterion(neg_pred, neg_label)
                
                loss.backward()
                optimizer.step()
                
                # Update persistent node registers AFTER computing loss objectives
                model.update_memory(b_src, b_dst, b_t, b_msg)
                model.memory.detach()
                total_loss += loss.item()
                batch_count += 1

            # --- Validation Evaluation Loop ---
            model.eval()
            val_preds, val_labels = [], []
            with torch.no_grad():
                for i in range(train_end, val_end, batch_size):
                    b_src = data.src[i:i+batch_size]
                    b_dst = data.dst[i:i+batch_size]
                    b_t = data.t[i:i+batch_size]
                    b_msg = msg_slice = data.msg[i:i+batch_size]

                    pos_out = torch.sigmoid(model(b_src, b_dst, b_t))
                    val_preds.extend(pos_out.cpu().numpy())
                    val_labels.extend(np.ones(len(pos_out)))
                    
                    b_neg_dst = torch.randint(0, num_nodes, (len(b_src),), device=device)
                    neg_out = torch.sigmoid(model(b_src, b_neg_dst, b_t))
                    val_preds.extend(neg_out.cpu().numpy())
                    val_labels.extend(np.zeros(len(neg_out)))

                    model.update_memory(b_src, b_dst, b_t, b_msg)

            val_ap = average_precision_score(val_labels, val_preds) * 100
            avg_loss = total_loss / batch_count
            
            # Record metrics
            epoch_idx.append(f"Epoch {epoch:02d}")
            ap_history.append(val_ap)
            loss_history.append(avg_loss)
            
            # Draw points in real-time onto UI graph components
            ap_chart.line_chart(pd.DataFrame({"Average Precision (%)": ap_history}, index=epoch_idx))
            loss_chart.line_chart(pd.DataFrame({"Loss Value": loss_history}, index=epoch_idx))
            
            p_bar.progress(int((epoch / epochs) * 100))

        status_box.success(f"🎯 Optimization Framework Complete! Peak Validation Accuracy achieved: **{max(ap_history):.2f}% AP**")
        st.balloons()