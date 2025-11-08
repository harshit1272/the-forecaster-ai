import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import json  # <--- CHANGE 1: Import the json library

# -----------------------------------------------------------
# PAGE CONFIG & CSS
# -----------------------------------------------------------
st.set_page_config(page_title="The Forecaster | Ratings", page_icon="⭐", layout="wide")

st.markdown("""
<style>
.block-container {padding-top:2rem; padding-bottom:2rem; max-width:1200px;}
h1,h2,h3,h4,h5 {font-family:'Segoe UI',sans-serif;}
hr {margin:1rem 0 1.5rem 0;}

/* --- Metric Cards (equal heights) --- */
.metric-card{
  border-radius:12px;
  padding:14px 16px;
  border:1px solid rgba(255,255,255,0.14);
  background:rgba(255,255,255,0.05);
  height:230px;             
  overflow:hidden;
  display:flex;
  flex-direction:column;
  justify-content:space-between;
}
.metric-card *{overflow-wrap:anywhere; word-break:break-word;}

/* --- Dotted “equation” boxes --- */
.eq{
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
  background:rgba(255,255,255,0.04);
  padding:6px 8px; border-radius:6px;
  border:1px dashed rgba(255,255,255,0.12);
  font-size:0.92rem; color:#EDEDED; margin-top:4px;
}
.eq b{color:#ffffff;}
.num1{color:#2ECC71;}
.num2{color:#00BFFF;}
.num3{color:#F39C12;}

.final-formula{
  border-radius:12px; border:1px solid rgba(255,255,255,0.16);
  background:rgba(255,255,255,0.045); padding:14px 16px; color:#EEE;
}
.section-title{margin-top:1.6rem; margin-bottom:0.6rem;}
.caption-note{color:#bdbdbd; font-size:0.92rem;}
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------
# LOAD MODELS & DATA
# -----------------------------------------------------------
try:
    weights = joblib.load("data_driven_weights.pkl")
    WEIGHT_ML, WEIGHT_AI = float(weights["WEIGHT_ML"]), float(weights["WEIGHT_AI"])
except Exception:
    WEIGHT_ML, WEIGHT_AI = 0.5, 0.5

@st.cache_resource
def load_assets():
    return {
        "ml_model": joblib.load('ml_specialist_model.pkl'),
        "ai_model": joblib.load('ai_specialist_model.pkl'),
        "preprocessor": joblib.load('preprocessor.pkl'),
        "category_list": joblib.load('category_list.pkl'),
        "category_stats": pd.read_pickle('category_stats.pkl'),
        "category_archetypes": joblib.load('category_archetypes.pkl'),
        "category_review_archetypes": joblib.load('category_review_archetypes.pkl'),
        "sentiment_analyzer": SentimentIntensityAnalyzer(),
        "sbert": SentenceTransformer('all-MiniLM-L6-v2')
    }

assets = load_assets()

# -----------------------------------------------------------
# FILTER CATEGORY LIST (≥ 4 entries)
# -----------------------------------------------------------
try:
    stats = assets["category_stats"]
    valid_categories = (
        stats.groupby("main_category")
        .filter(lambda x: len(x) >= 4)["main_category"]
        .unique()
        .tolist()
    )
    assets["category_list"] = [cat for cat in assets["category_list"] if cat in valid_categories]
except Exception as e:
    st.warning(f"⚠️ Could not filter categories: {e}")

# -----------------------------------------------------------
# HEADER
# -----------------------------------------------------------
st.markdown("<h1 style='font-weight:600;'>🔮 The Forecaster</h1>", unsafe_allow_html=True)
st.caption("Explainable AI combining **Market**, **Description**, and **Sentiment** signals to estimate fair product ratings.")
st.write("---")

# -----------------------------------------------------------
# INPUTS (two columns)
# -----------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📦 Category & Snapshot")
    main_category = st.selectbox("Select Product Category", assets["category_list"], index=0)
    stats = assets["category_stats"]
    cat_stats = stats[stats["main_category"] == main_category]
    if not cat_stats.empty:
        avg_price = float(cat_stats["mean_price"].iloc[0])
        avg_rating = float(cat_stats["rating"].mean())
    else:
        avg_price, avg_rating = 0.0, 0.0

    c1, c2, c3 = st.columns(3)
    c1.metric("Average Price", f"₹{avg_price:,.0f}")
    c2.metric("Average Rating", f"{avg_rating:.2f} ⭐")
    predicted_top_slot = c3.container()   # reserve a slot in 3rd column for Predicted Rating (with star)

with col2:
    st.markdown("### 💰 Pricing Details")
    actual_price = st.number_input("Actual Price (₹)", min_value=0.0, format="%.2f", value=1099.0)
    discounted_price = st.number_input("Discounted Price (₹)", min_value=0.0, format="%.2f", value=399.0)
    discount_percentage = ((actual_price - discounted_price)/actual_price*100) if actual_price>0 else 0.0
    st.metric("Discount Percentage", f"{discount_percentage:.0f}%")

st.markdown("---")

st.markdown("### 🧠 Text Inputs")
dcol, rcol = st.columns(2)
with dcol:
    product_description = st.text_area("Product Description", height=160, placeholder="Describe product features, materials, or specifications...")
with rcol:
    review_content = st.text_area("User Review (optional)", height=160, placeholder="Paste a customer review (optional).")

show_chart = st.checkbox("📊 Show component comparison chart", value=False)

# -----------------------------------------------------------
# PREDICTION
# -----------------------------------------------------------
if st.button("✨ Predict Rating", use_container_width=True):
    if discounted_price > actual_price:
        st.error("❌ Discounted price cannot exceed actual price.")
    elif not product_description.strip():
        st.error("⚠️ Please enter a valid product description.")
    else:
        with st.spinner("🔎 Running specialists..."):
            sbert = assets["sbert"]

            # Market (ML)
            cat_mean = float(stats.loc[stats['main_category']==main_category, 'mean_price'].iloc[0])
            cat_std  = float(stats.loc[stats['main_category']==main_category, 'std_price'].iloc[0] or 0.0)
            z = (discounted_price - cat_mean) / cat_std if cat_std > 0 else 0.0
            price_factor = float(np.exp(-abs(z)))

            ml_df = pd.DataFrame({
                'actual_price':[actual_price],
                'discounted_price':[discounted_price],
                'discount_percentage':[discount_percentage],
                'price_zscore':[z],
                'main_category':[main_category]
            })
            ml_raw = float(assets["ml_model"].predict(ml_df)[0])
            ml_adj = ml_raw * price_factor

            # AI (Description)
            desc_vec = sbert.encode([product_description])
            desc_arch = assets["category_archetypes"][main_category].astype(np.float32)
            desc_rel = float(util.cos_sim(desc_vec, desc_arch).item())
            ai_feats = getattr(assets["ai_model"], 'feature_names_in_', [f'f{i}' for i in range(desc_vec.shape[1])])
            ai_raw = float(np.clip(assets["ai_model"].predict(pd.DataFrame(desc_vec, columns=ai_feats))[0], 1, 5))
            ai_adj = ai_raw * desc_rel

            # Sentiment (optional)
            if review_content.strip():
                sent_used = True
                rev_vec = sbert.encode([review_content])
                rev_arch = assets["category_review_archetypes"][main_category].astype(np.float32)
                rev_rel = float(util.cos_sim(rev_vec, rev_arch).item())
                s_comp = float(assets["sentiment_analyzer"].polarity_scores(str(review_content))['compound'])
                s_scaled = float(np.clip(3 + 2*s_comp, 1, 5))
                s_adj = s_scaled * rev_rel
                text_score = (ai_adj + s_adj)/2.0
            else:
                sent_used = False
                s_comp = s_scaled = rev_rel = s_adj = 0.0
                text_score = ai_adj

            final_rating = float(np.clip((WEIGHT_ML*ml_adj)+(WEIGHT_AI*text_score), 1, 5))
        
        if show_chart:
            st.markdown("### 📊 Rating Composition Chart")

            # Compute contributions
            total_contrib = (WEIGHT_ML * ml_adj) + (WEIGHT_AI * text_score)
            price_contrib_pct = (WEIGHT_ML * ml_adj / total_contrib) * 100
            text_contrib_pct  = (WEIGHT_AI * text_score / total_contrib) * 100

            if sent_used:
                ai_sub_pct   = (ai_adj / (ai_adj + s_adj)) * text_contrib_pct
                sent_sub_pct = (s_adj / (ai_adj + s_adj)) * text_contrib_pct
            else:
                ai_sub_pct   = text_contrib_pct
                sent_sub_pct = 0

            # Colors
            price_color = "#2EFFA2"
            text_color  = "#00CFFF"
            ai_color    = "#00CFFF"
            sent_color  = "#FFB347"

            # --- Figure setup ---
            fig, ax = plt.subplots(figsize=(8, 0.9), facecolor="none")
            plt.subplots_adjust(bottom=0.20)
            fig.patch.set_alpha(0.0)
            ax.set_facecolor("none")

            # --- Bars ---
            ax.barh(0, price_contrib_pct, color="none", edgecolor=price_color, lw=2.0, height=0.22, zorder=2)
            ax.barh(0, text_contrib_pct, left=price_contrib_pct, color="none", edgecolor=text_color, lw=2.0, height=0.22, zorder=2)
            if sent_used:
                ax.barh(0, ai_sub_pct-0.8, left=price_contrib_pct+0.4, color="none", edgecolor=ai_color, lw=1.5, height=0.10, zorder=3)
                ax.barh(0, sent_sub_pct-0.8, left=price_contrib_pct+ai_sub_pct+0.8, color="none", edgecolor=sent_color, lw=1.5, height=0.10, zorder=3)

            # --- Axes cleanup ---
            ax.set_xlim(0, 100)
            ax.set_ylim(-0.5, 0.5)
            ax.set_yticks([])
            ax.set_xticks([0, 20, 40, 60, 80, 100])
            ax.tick_params(colors="white", labelsize=8, pad=4)
            ax.set_xlabel("Contribution to Final Rating (%)", color="white", fontsize=8.5, labelpad=8)
            ax.set_title("Final Rating Composition", color="white", fontsize=11, pad=10, weight="bold")

            for spine in ax.spines.values():
                spine.set_visible(False)

            # --- Top labels ---
            ax.text(price_contrib_pct / 2, 0.28,
                    f"Price contribution to rating ({price_contrib_pct:.1f}%)",
                    ha="center", va="bottom", fontsize=9.2, color=price_color, weight="semibold")
            ax.text(price_contrib_pct + text_contrib_pct / 2, 0.28,
                    f"Text contribution to rating ({text_contrib_pct:.1f}%)",
                    ha="center", va="bottom", fontsize=9.2, color=text_color, weight="semibold")

            # --- Bottom labels (auto-collision avoidance) ---
            if sent_used:
                y_bottom = -0.22
                desc_x = price_contrib_pct + ai_sub_pct / 2
                rev_x = price_contrib_pct + ai_sub_pct + sent_sub_pct / 2

                # If too close, offset review label
                min_gap = 6  # percent gap between centers
                if rev_x - desc_x < min_gap:
                    rev_x += (min_gap - (rev_x - desc_x)) / 2
                    desc_x -= (min_gap - (rev_x - desc_x)) / 2

                ax.text(desc_x, y_bottom, f"Description ({ai_sub_pct:.1f}%)",
                        ha="center", va="top", fontsize=7.8, color=ai_color)
                ax.text(rev_x, y_bottom, f"Review ({sent_sub_pct:.1f}%)",
                        ha="center", va="top", fontsize=7.8, color=sent_color)
            else:
                ax.text(price_contrib_pct + text_contrib_pct / 2, -0.22,
                        f"Description ({ai_sub_pct:.1f}%)",
                        ha="center", va="top", fontsize=7.8, color=ai_color)

            st.pyplot(fig, transparent=True)


        # -----------------------------------------------------------
        # TOP PREDICTED RENDER (with ⭐ in value)
        # -----------------------------------------------------------
        predicted_top_slot.metric("Predicted Rating", f"{final_rating:.2f} ⭐")

        # -----------------------------------------------------------
        # SUMMARY METRICS (with ⭐)
        # -----------------------------------------------------------
        st.markdown("### ✅ Prediction Summary")
        m1, m2, m3 = st.columns(3)
        m1.metric("⭐ Predicted Rating", f"{final_rating:.2f} ⭐")
        m2.metric("💰 Market Adjusted", f"{ml_adj:.2f} ⭐")
        m3.metric("🧠 Textual Adjusted", f"{text_score:.2f} ⭐")

        # -----------------------------------------------------------
        # DETAILED BREAKDOWN (equal height boxes)
        # -----------------------------------------------------------
        st.markdown("### 🧩 Detailed Breakdown")
        b1, b2, b3 = st.columns(3)

        with b1:
            st.markdown("**💰 Market (ML)**")
            st.markdown(
                f"<div class='metric-card'>"
                f"<div>Z-score: {z:.2f}<br>"
                f"Price Relevance Factor: {price_factor:.3f}<br>"
                f"ML Raw Rating: {ml_raw:.2f}<br>"
                f"ML Adjusted Rating: {ml_adj:.2f}</div>"
                f"<div>"
                f"<div class='eq'>Price Relevance = exp(−|<span class='num1'>{z:.2f}</span>|) = <span class='num2'>{price_factor:.3f}</span></div>"
                f"<div class='eq'>ML adjusted = <span class='num1'>{ml_raw:.3f}</span> × <span class='num2'>{price_factor:.3f}</span> = <span class='num3'>{ml_adj:.3f}</span></div>"
                f"</div></div>", unsafe_allow_html=True
            )

        with b2:
            st.markdown("**🧠 Description (AI)**")
            st.markdown(
                f"<div class='metric-card'>"
                f"<div>AI Raw Rating: {ai_raw:.2f}<br>"
                f"Description Archetype Relevance: {desc_rel:.3f}<br>"
                f"AI Adjusted Rating: {ai_adj:.2f}</div>"
                f"<div>"
                f"<div class='eq'>AI adjusted = <span class='num1'>{ai_raw:.3f}</span> × <span class='num2'>{desc_rel:.3f}</span> = <span class='num3'>{ai_adj:.3f}</span></div>"
                f"</div></div>", unsafe_allow_html=True
            )

        with b3:
            st.markdown("**💬 User-Review (Optional)**")
            if sent_used:
                st.markdown(
                    f"<div class='metric-card'>"
                    f"<div>Compound Score: {s_comp:.2f}<br>"
                    f"Rescaled Sentiment: {s_scaled:.2f}<br>"
                    f"Review Archetype Relevance: {rev_rel:.3f}<br>"
                    f"Sentiment Adjusted: {s_adj:.2f}</div>"
                    f"<div>"
                    f"<div class='eq'>Sentiment adjusted = <span class='num1'>{s_scaled:.3f}</span> × <span class='num2'>{rev_rel:.3f}</span> = <span class='num3'>{s_adj:.3f}</span></div>"
                    f"</div></div>", unsafe_allow_html=True
                )
            else:
                st.info("No review provided — sentiment skipped.")

        # -----------------------------------------------------------
        # FINAL FORMULA (colored)
        # -----------------------------------------------------------
        st.markdown("<div class='section-title'><h3>🧮 Final Rating Formula</h3></div>", unsafe_allow_html=True)
        if sent_used:
            textual_avg = (ai_adj + s_adj)/2.0
            st.markdown(
                f"<div class='final-formula'>"
                f"<div class='eq'>Textual average = (<span class='num1'>{ai_adj:.3f}</span> + <span class='num2'>{s_adj:.3f}</span>) ÷ 2 = <span class='num3'>{textual_avg:.3f}</span></div>"
                f"<div class='eq'>Final rating = (<span class='num1'>{ml_adj:.3f}</span> × {WEIGHT_ML:.3f}) + (<span class='num2'>{textual_avg:.3f}</span> × {WEIGHT_AI:.3f}) = <b class='num3'>{final_rating:.3f} ⭐</b></div>"
                f"</div>", unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"<div class='final-formula'>"
                f"<div class='eq'>Final rating = (<span class='num1'>{ml_adj:.3f}</span> × {WEIGHT_ML:.3f}) + (<span class='num2'>{ai_adj:.3f}</span> × {WEIGHT_AI:.3f}) = <b class='num3'>{final_rating:.3f} ⭐</b></div>"
                f"</div>", unsafe_allow_html=True
            )

        # -----------------------------------------------------------
        
        
            
        st.caption(
                "💡 Sentiment is optional — if no review is provided, only the AI (description) component is used. "
            )


        st.caption(
                "💡 Price Relevance Factor shows how representative the product’s price is for its category "
                "(based on Z-score). AI and Sentiment each use separate archetype relevances for robustness."
            )


        st.caption(
                "💡 Sentiment is calculated independently of data and has its *own* review archetype relevance — "
                "so irrelevant reviews (e.g., about other products) does not distort the prediction."
            )

# -----------------------------------------------------------
# QUICK TEST TEMPLATES (High-Relevance Final Version)
# -----------------------------------------------------------
st.markdown("---")
st.markdown("### 🧪 Quick Test Templates")

test_templates = {
    "USBCables": {
        "Good Description": "Premium 1 m braided USB-A to Type-C fast charging cable supporting up to 3 A current and 480 Mbps data transfer. Features double-layer nylon jacket, reinforced aluminum connectors, and smart chipset protection against over-voltage, over-heat, and short circuit. Compatible with smartphones, tablets, and power banks. 10 000 + bend lifespan with 1 year warranty for everyday reliable use.",
        "Bad Description": "Ordinary charging cable. Works for most phones but build quality feels basic and tangles easily.",
        "Positive Review": "I’ve been using this cable for more than a month now and it still performs like new. Charges my phone from 0-100 % in under an hour and data transfer is fast. Connectors fit snugly and the braided sleeve feels durable. Delivery was quick and packaging neat. For this price range, definitely one of the best Type-C cables.",
        "Negative Review": "Worked fine for the first couple of weeks but then charging speed dropped. Connector feels loose and sometimes disconnects on its own. Build looks nice but durability is disappointing for a daily-use cable."
    },

    "SmartWatches": {
        "Good Description": "1.85-inch AMOLED full-touch smartwatch with Bluetooth calling, AI voice assistant, and complete health tracking — SpO₂, heart-rate, stress, and sleep monitor. 120 + sports modes, metallic frame, IP68 water resistance, 7-day battery, and 150 + watch faces. Includes camera/music control, notifications for calls/messages, and fast magnetic charging. Compatible with Android and iOS.",
        "Bad Description": "Basic smartwatch with few sports modes. Battery is fine but sensors seem inconsistent.",
        "Positive Review": "Received it two weeks ago and I’m genuinely impressed. The display is crisp, touch response is smooth, and Bluetooth calling works flawlessly. Health metrics are close to accurate and battery lasts almost a week per charge. Strap quality is good too. For the features offered, it’s excellent value for money.",
        "Negative Review": "After using for a few days, the watch keeps disconnecting and step counter gives random readings. Battery drains faster than claimed and speaker volume is low during calls. Not happy considering the price."
    },

    "Smartphones": {
        "Good Description": "5G dual-SIM smartphone with 6.7-inch FHD+ AMOLED 120 Hz display, 50 MP AI triple camera with OIS, and 16 MP front camera. Powered by octa-core processor, 8 GB RAM / 128 GB storage, and 5000 mAh battery with 33 W Type-C fast charging. Features side fingerprint, face unlock, stereo speakers, and Android 13 with clean UI. Includes 1 year manufacturer warranty.",
        "Bad Description": "Regular phone with decent camera and display. Battery okay for normal use.",
        "Positive Review": "Been using this phone daily for over three weeks. Display is vibrant and smooth, apps load instantly, and camera performs well in daylight. Battery easily lasts a full day and 33 W fast charging tops it up quickly. No heating issues so far. Overall a solid performer in this segment.",
        "Negative Review": "Phone looks good but performance isn’t stable. Slight lag while multitasking and night photos lack detail. Battery drains quicker than expected and sound output is average. Expected better quality for the price."
    },

    "SmartTelevisions": {
        "Good Description": "43-inch 4K Ultra HD LED Smart TV with Android / Google TV 11 and built-in Chromecast. Dolby Audio + DTS-HD sound, bezel-less design, quad-core processor, dual-band Wi-Fi, 2 HDMI + 2 USB ports, 2 GB RAM / 16 GB storage. Supports Netflix, Prime Video, Disney+ Hotstar, and YouTube with voice remote and Google Assistant. Includes 1 year comprehensive warranty.",
        "Bad Description": "Smart TV with average picture and sound. Apps work but sometimes lag.",
        "Positive Review": "Installed last week and setup was hassle-free. Picture quality is crisp, colors are vibrant, and sound output fills my living room nicely. Wi-Fi and streaming apps run smoothly with no noticeable lag. Voice search through the remote works great. For this price bracket, it’s a perfect family TV.",
        "Negative Review": "After a few days of use the TV started lagging while switching apps. Sound feels flat and the remote response is inconsistent. Service support took time to respond. Not the smooth Android TV experience I expected."
    },

    "In-Ear": {
        "Good Description": "Bluetooth 5.3 true-wireless in-ear earbuds with 13 mm drivers delivering deep bass and clear vocals. Offer up to 40 hours total playback with Type-C fast charging, low-latency gaming mode, and ENC for calls. IPX5 sweat-resistant ergonomic design with instant pairing and voice assistant support — ideal for music, workouts, and calls.",
        "Bad Description": "Simple earphones with okay sound and average battery. Sometimes disconnect while using.",
        "Positive Review": "Been using these earbuds for about three weeks. Sound is rich with punchy bass and vocals are clear. Connection is stable even outdoors and the case charges quickly via Type-C. Fit is comfortable for long listening sessions. For this budget, they outperform most rivals.",
        "Negative Review": "Audio starts cracking at high volume and one earbud lost connection after ten days. Mic quality during calls is below average. Looks stylish but not durable enough for daily commute."
    },

    "RemoteControls": {
        "Good Description": "Universal IR remote control compatible with most Smart TVs, LED/LCD models, and DTH set-top boxes. Ready-to-use — no programming required for supported devices. Offers dedicated hot keys for Netflix, Prime Video, and YouTube. Long battery life with sturdy ABS body. Note: this is a compatible remote, not the original manufacturer product.",
        "Bad Description": "TV remote that works with some models. Buttons and range feel basic.",
        "Positive Review": "Ordered for my Samsung Smart TV and it worked instantly without setup. All keys including hot keys function properly and range is good. Build quality is solid for this price. After two weeks of use, completely satisfied with performance.",
        "Negative Review": "Remote arrived on time but doesn’t pair with my set-top box. Some buttons are unresponsive and range is limited. Works partially, so had to request a replacement."
    }
}

# --- Styling for code blocks ---
st.markdown("""
<style>
[data-testid="stCodeBlock"] {
    border: 1px dashed rgba(255,255,255,0.25);
    background: rgba(255,255,255,0.04);
    border-radius: 8px;
    margin-bottom: 0.8rem;
    padding: 10px 14px;
}
[data-testid="stCodeBlock"] pre {
    font-family: "Segoe UI", sans-serif;
    color: #EDEDED;
    font-size: 0.93rem;
    white-space: pre-wrap;
    word-wrap: break-word;
}
</style>
""", unsafe_allow_html=True)

selected_templ = st.selectbox("Select product to view example texts:", list(test_templates.keys()))
examples = test_templates[selected_templ]

st.markdown(f"#### 📄 {selected_templ} – Example Texts")

def display_example(title, text):
    st.markdown(f"**{title}**")
    st.code(text, language=None)

display_example("Good Product Description", examples["Good Description"])
display_example("Bad Product Description", examples["Bad Description"])
display_example("Positive Review", examples["Positive Review"])
display_example("Negative Review", examples["Negative Review"])
