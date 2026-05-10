# Libraries

import streamlit as st
import pandas as pd
import numpy as np
import pickle as pck
from pathlib import Path

st.set_page_config(page_title="Startup Screening Tool", layout="wide")


@st.cache_resource
def load_models():
    repo_root = Path(__file__).parent.parent
    with open(repo_root / "Models" / "start_ups_success_predictor.pkl", "rb") as f:
        success_model = pck.load(f)
    with open(repo_root / "Models" / "market_category_encoder.pkl", "rb") as f:
        mc_encoder = pck.load(f)
    with open(repo_root / "Models" / "start_ups_investment_predictor.pkl", "rb") as f:
        invest_model = pck.load(f)
    return success_model, mc_encoder, invest_model


success_model, mc_encoder, invest_model = load_models()

MARKET_CATEGORIES = [
    "Advertising & Marketing", "Analytics & AI", "Biotechnology",
    "Clean Tech & Energy", "E-Commerce & Retail", "Education",
    "Energy", "Entertainment & Media", "Finance & FinTech",
    "Hardware & Semiconductors", "Healthcare",
    "Manufacturing & Logistics", "Mobile", "Other", "Real Estate",
    "Security", "Social & Communication", "Software & SaaS",
    "Travel & Hospitality", "Unknown",
]

FUNDING_TYPES = [
    "seed", "venture", "angel", "equity_crowdfunding", "convertible_note",
    "debt_financing", "grant", "private_equity", "undisclosed",
    "post_ipo_equity", "post_ipo_debt", "secondary_market", "product_crowdfunding",
]

ROUNDS = ["round_A", "round_B", "round_C", "round_D", "round_E", "round_F", "round_G", "round_H"]

INVEST_CLASSES = {0: "Seed (< $1M)", 1: "Early ($1M - $20M)", 2: "Growth (> $20M)"}


st.title("Startup screening tool")
st.markdown(
    "This webapp screens startups based on their funding profile. "
    "Select a prediction type to get started."
)
st.markdown(
    "For more information about the model creation and training, please refer to the GitHub repository: "
    "[EADA_DAM25_Final_Project](https://github.com/albemela-creator/EADA_DAM25_Final_Project)"
)
st.write("")

prediction_type = st.selectbox(
    "Select prediction type",
    options=["Success prediction", "Investment size"],
    index=None,
    placeholder="Choose an option...",
)

# ---------- Success prediction ----------

if prediction_type == "Success prediction":
    st.markdown(
        "Predicts whether a startup will be **acquired** or **closed**, "
        "based on its funding profile. Model: Gradient Boosting Classifier."
    )
    st.divider()

    with st.form("success_form"):
        st.subheader("General information")
        col1, col2 = st.columns(2)
        with col1:
            market_category = st.selectbox(
                "Market category",
                options=MARKET_CATEGORIES,
                index=None,
                placeholder="Select a category...",
            )
            funding_total_usd = st.number_input(
                "Total funding raised (USD)",
                min_value=0.0, value=None, step=10000.0, format="%.0f",
                placeholder="e.g. 500000",
            )
        with col2:
            funding_rounds = st.number_input(
                "Number of funding rounds",
                min_value=1, value=None, step=1,
                placeholder="e.g. 3",
            )
            funding_cycle_months = st.number_input(
                "Funding cycle duration (months)",
                min_value=0.1, value=None, step=1.0, format="%.1f",
                placeholder="e.g. 24",
            )

        st.subheader("Funding breakdown by type (USD - enter 0 if not applicable)")
        ft_cols = st.columns(3)
        amounts = {}
        for i, ft in enumerate(FUNDING_TYPES):
            with ft_cols[i % 3]:
                amounts[ft] = st.number_input(
                    ft.replace("_", " ").title(),
                    min_value=0.0, value=None, step=10000.0, format="%.0f",
                    placeholder="0", key=f"s_{ft}",
                )

        st.subheader("Funding rounds breakdown (USD - enter 0 if not applicable)")
        r_cols = st.columns(4)
        round_vals = {}
        for i, r in enumerate(ROUNDS):
            with r_cols[i % 4]:
                round_vals[r] = st.number_input(
                    r.replace("_", " ").title(),
                    min_value=0.0, value=None, step=10000.0, format="%.0f",
                    placeholder="0", key=f"s_{r}",
                )

        submitted_s = st.form_submit_button("Predict success", width='stretch')

    if submitted_s:
        if any(v is None for v in [market_category, funding_total_usd, funding_rounds, funding_cycle_months]):
            st.warning("Please fill in all general information fields before predicting.")
        else:
            amounts = {k: (v if v is not None else 0.0) for k, v in amounts.items()}
            round_vals = {k: (v if v is not None else 0.0) for k, v in round_vals.items()}

            mc_enc = mc_encoder.transform(pd.DataFrame({"market_category": [market_category]})).ravel()[0]
            avg_per_round = funding_total_usd / funding_rounds if funding_rounds > 0 else 0
            intensity = funding_total_usd / funding_cycle_months if funding_cycle_months > 0 else 0
            rounds_per_year = funding_rounds / (funding_cycle_months / 12) if funding_cycle_months > 0 else 0
            has_early = int(amounts["seed"] > 0 or amounts["angel"] > 0)
            has_late = int(any(round_vals.get(r, 0) > 0 for r in ["round_D", "round_E", "round_F", "round_G", "round_H"]))
            institutional = amounts["venture"] + amounts["private_equity"]
            inst_share = institutional / funding_total_usd if funding_total_usd > 0 else 0
            debt = amounts["debt_financing"] + amounts["convertible_note"]
            equity = amounts["venture"] + amounts["angel"] + amounts["private_equity"]
            debt_eq_ratio = debt / (equity + 1e-6)
            post_ipo = int(amounts["post_ipo_equity"] > 0 or amounts["post_ipo_debt"] > 0)
            diversity = int(pd.Series([amounts[ft] for ft in FUNDING_TYPES]).nunique())

            inp = {
                "market_category": mc_enc,
                "funding_total_usd": funding_total_usd,
                "funding_rounds": funding_rounds,
                **{ft: amounts[ft] for ft in FUNDING_TYPES},
                **{r: round_vals[r] for r in ROUNDS},
                "funding_cycle_months": funding_cycle_months,
                "funding_avg_per_round": avg_per_round,
                "funding_diversity": diversity,
                "funding_intensity": intensity,
                "rounds_per_year": rounds_per_year,
                "has_early_stage": has_early,
                "has_late_stage": has_late,
                "institutional_share": inst_share,
                "debt_equity_ratio": debt_eq_ratio,
                "post_ipo_presence": post_ipo,
            }

            X = pd.DataFrame([inp])[list(success_model.feature_names_in_)]
            pred = success_model.predict(X)[0]
            proba = success_model.predict_proba(X)[0]

            st.divider()
            st.subheader("Prediction result")
            if pred:
                st.success(f"**Acquired** - Probability: {proba[1]*100:.1f}%")
            else:
                st.error(f"**Closed** - Probability: {proba[0]*100:.1f}%")
            col1, col2 = st.columns(2)
            col1.metric("P(Acquired)", f"{proba[1]*100:.1f}%")
            col2.metric("P(Closed)", f"{proba[0]*100:.1f}%")

# ---------- Investment size prediction ----------

elif prediction_type == "Investment size":
    st.markdown(
        "Predicts the likely funding bracket: Seed (<$1M), Early ($1M-$20M), or Growth (>$20M). "
        "Model: Random Forest Classifier."
    )
    st.divider()

    with st.form("invest_form"):
        st.subheader("General information")
        col1, col2 = st.columns(2)
        with col1:
            market_category = st.selectbox(
                "Market category",
                options=MARKET_CATEGORIES,
                index=None,
                placeholder="Select a category...",
            )
            status = st.selectbox(
                "Current startup status",
                options=["operating", "acquired", "closed"],
                index=None,
                placeholder="Select a status...",
            )
        with col2:
            funding_rounds = st.number_input(
                "Number of funding rounds",
                min_value=1, value=None, step=1,
                placeholder="e.g. 3",
            )
            funding_cycle_months = st.number_input(
                "Funding cycle duration (months)",
                min_value=0.1, value=None, step=1.0, format="%.1f",
                placeholder="e.g. 24",
            )

        st.subheader("Funding types present (check all that apply)")
        ft_cols = st.columns(3)
        ft_vals = {}
        for i, ft in enumerate(FUNDING_TYPES):
            with ft_cols[i % 3]:
                ft_vals[ft] = st.checkbox(ft.replace("_", " ").title(), key=f"i_{ft}")

        st.subheader("Funding rounds reached (check all that apply)")
        r_cols = st.columns(4)
        r_vals = {}
        for i, r in enumerate(ROUNDS):
            with r_cols[i % 4]:
                r_vals[r] = st.checkbox(r.replace("_", " ").title(), key=f"i_{r}")

        submitted_i = st.form_submit_button("Predict investment size", width='stretch')

    if submitted_i:
        if any(v is None for v in [market_category, status, funding_rounds, funding_cycle_months]):
            st.warning("Please fill in all general information fields before predicting.")
        else:
            rounds_per_year = funding_rounds / (funding_cycle_months / 12) if funding_cycle_months > 0 else 0
            ft_bool = [int(ft_vals[ft]) for ft in FUNDING_TYPES]
            diversity = int(pd.Series(ft_bool).nunique())
            debt_b = int(ft_vals["debt_financing"]) + int(ft_vals["convertible_note"])
            eq_b = int(ft_vals["venture"]) + int(ft_vals["angel"]) + int(ft_vals["private_equity"])
            debt_eq_ratio = debt_b / (eq_b + 1e-6)

            inp = {
                "funding_rounds": funding_rounds,
                **{ft: int(ft_vals[ft]) for ft in FUNDING_TYPES},
                **{r: int(r_vals[r]) for r in ROUNDS},
                "funding_cycle_months": funding_cycle_months,
                "funding_diversity": diversity,
                "rounds_per_year": rounds_per_year,
                "debt_equity_ratio": debt_eq_ratio,
            }

            for cat in MARKET_CATEGORIES[1:]:
                inp[f"market_category_{cat}"] = int(market_category == cat)

            inp["status_closed"] = int(status == "closed")
            inp["status_operating"] = int(status == "operating")

            X = pd.DataFrame([inp])[list(invest_model.feature_names_in_)]
            pred = invest_model.predict(X)[0]
            proba = invest_model.predict_proba(X)[0]

            st.divider()
            st.subheader("Prediction result")
            st.success(f"**{INVEST_CLASSES[pred]}** - Confidence: {proba[pred]*100:.1f}%")
            st.subheader("Probability breakdown")
            prob_df = pd.DataFrame({
                "Funding bracket": [INVEST_CLASSES[i] for i in range(3)],
                "Probability": [f"{p*100:.1f}%" for p in proba],
            })
            st.dataframe(prob_df, hide_index=True, width='stretch')
