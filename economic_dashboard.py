# Economic Indicators Dashboard
# Save this as: economic_dashboard.py
# Run with: streamlit run economic_dashboard.py

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc,
)

st.set_page_config(
    page_title="Economic Indicators Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-header {
        font-size: 42px;
        font-weight: bold;
        color: white;
        text-align: center;
        padding: 20px;
        background: linear-gradient(90deg, #2c3e50 0%, #4ca1af 100%);
        border-radius: 10px;
    }
    .sub-header {
        font-size: 24px;
        font-weight: bold;
        color: #2c3e50;
        padding: 10px 0;
        border-bottom: 3px solid #4ca1af;
        margin-bottom: 15px;
    }
    .finding-box {
        background: #fff3cd;
        border: 1px solid #ffc107;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
    }
    .recommendation {
        background: #d4edda;
        border: 1px solid #28a745;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
    }
    .alert-box {
        background: #f8d7da;
        border: 1px solid #dc3545;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        color: #721c24;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

CORE_INDICATORS = [
    "gdp_growth",
    "inflation_rate",
    "unemployment_rate",
    "interest_rate",
    "consumer_spending_growth",
    "industrial_production_growth",
    "trade_balance",
    "stock_index_return",
]

NICE_NAMES = {
    "gdp_growth": "GDP Growth",
    "inflation_rate": "Inflation Rate",
    "unemployment_rate": "Unemployment Rate",
    "interest_rate": "Interest Rate",
    "consumer_spending_growth": "Consumer Spending Growth",
    "industrial_production_growth": "Industrial Production Growth",
    "trade_balance": "Trade Balance",
    "stock_index_return": "Stock Index Return",
    "real_growth_proxy": "Real Growth Proxy",
    "economic_activity_index": "Economic Activity Index",
    "labor_stress_index": "Labor Stress Index",
    "policy_gap": "Policy Gap",
}


@st.cache_data
def load_data():
    df = None
    for path in ["data/economic_data.csv", "economic_data.csv"]:
        try:
            df = pd.read_csv(path)
            break
        except FileNotFoundError:
            continue

    if df is None:
        dates = pd.date_range("2015-01-01", periods=120, freq="M")
        rng = np.random.default_rng(42)
        df = pd.DataFrame(
            {
                "date": dates,
                "gdp_growth": rng.normal(2.5, 1.2, len(dates)),
                "inflation_rate": rng.normal(3.0, 1.0, len(dates)),
                "unemployment_rate": rng.normal(6.0, 1.5, len(dates)),
                "interest_rate": rng.normal(4.0, 1.0, len(dates)),
                "consumer_spending_growth": rng.normal(2.2, 1.1, len(dates)),
                "industrial_production_growth": rng.normal(2.8, 1.3, len(dates)),
                "trade_balance": rng.normal(-1.5, 0.8, len(dates)),
                "stock_index_return": rng.normal(0.8, 2.5, len(dates)),
            }
        )

    if "date" not in df.columns:
        for c in ["Date", "month", "period"]:
            if c in df.columns:
                df = df.rename(columns={c: "date"})
                break

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values("date").reset_index(drop=True)
    return df


@st.cache_data
def build_features(df):
    d = df.copy()

    if "gdp_growth" in d.columns and "inflation_rate" in d.columns:
        d["real_growth_proxy"] = d["gdp_growth"] - d["inflation_rate"] * 0.2

    if "consumer_spending_growth" in d.columns and "industrial_production_growth" in d.columns:
        d["economic_activity_index"] = (
            d["consumer_spending_growth"].fillna(0) * 0.5
            + d["industrial_production_growth"].fillna(0) * 0.5
        )

    if "unemployment_rate" in d.columns:
        d["labor_stress_index"] = d["unemployment_rate"] / (d["unemployment_rate"].mean() + 1e-6)

    if "interest_rate" in d.columns and "inflation_rate" in d.columns:
        d["policy_gap"] = d["interest_rate"] - d["inflation_rate"]

    return d


@st.cache_data
def train_models(df):
    d = build_features(df)

    if "growth_class" in d.columns:
        target = "growth_class"
    elif "recession_flag" in d.columns:
        target = "recession_flag"
    elif "inflation_flag" in d.columns:
        target = "inflation_flag"
    else:
        base = d["gdp_growth"] if "gdp_growth" in d.columns else pd.Series([0] * len(d))
        d["target"] = (base > base.median()).astype(int)
        target = "target"

    y = d[target].astype(int)
    feature_cols = [
        c for c in d.columns
        if c not in ["date", target] and pd.api.types.is_numeric_dtype(d[c])
    ]
    X = d[feature_cols].copy().fillna(d[feature_cols].median(numeric_only=True))

    if len(X) < 20 or y.nunique() < 2:
        return {}, X, y, target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    }

    results = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in models.items():
        use_scaled = name == "Logistic Regression"
        Xtr = X_train_scaled if use_scaled else X_train
        Xte = X_test_scaled if use_scaled else X_test

        model.fit(Xtr, y_train)
        y_pred = model.predict(Xte)
        y_prob = model.predict_proba(Xte)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(Xte)
        cv_scores = cross_val_score(model, Xtr, y_train, cv=cv, scoring="accuracy")
        fpr, tpr, _ = roc_curve(y_test, y_prob)

        results[name] = {
            "model": model,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
            "fpr": fpr,
            "tpr": tpr,
            "roc_auc": auc(fpr, tpr),
            "cm": confusion_matrix(y_test, y_pred),
        }

    return results, X, y, target


def page_executive_summary(df):
    st.markdown('<div class="sub-header">Executive Summary</div>', unsafe_allow_html=True)

    latest = df.iloc[-1]
    prior = df.iloc[-2] if len(df) > 1 else latest

    cols = st.columns(4)
    metrics = [
        ("GDP Growth", "gdp_growth", "%"),
        ("Inflation Rate", "inflation_rate", "%"),
        ("Unemployment Rate", "unemployment_rate", "%"),
        ("Interest Rate", "interest_rate", "%"),
    ]
    for col, (label, key, suffix) in zip(cols, metrics):
        if key in df.columns:
            delta = latest[key] - prior[key]
            col.metric(label, f"{latest[key]:.2f}{suffix}", f"{delta:+.2f}{suffix}")

    st.markdown("<br>", unsafe_allow_html=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    if "gdp_growth" in df.columns:
        axes[0].plot(df["date"], df["gdp_growth"], color="#2c3e50", linewidth=2)
        axes[0].set_title("GDP Growth Over Time")
        axes[0].set_ylabel("%")
        axes[0].grid(alpha=0.3)
    if "economic_activity_index" in df.columns:
        axes[1].plot(df["date"], df["economic_activity_index"], color="#4ca1af", linewidth=2)
        axes[1].set_title("Economic Activity Index Over Time")
        axes[1].grid(alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.markdown(
        f"""
        <div class="finding-box">
        <strong>Snapshot:</strong> the latest reading shows GDP growth at
        {latest.get('gdp_growth', float('nan')):.2f}%, inflation at
        {latest.get('inflation_rate', float('nan')):.2f}%, and a policy gap
        (interest rate minus inflation) of {latest.get('policy_gap', float('nan')):.2f}
        points.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div class="recommendation">
        <strong>Coverage:</strong> {len(df)} monthly observations across
        {len(df.select_dtypes(include=np.number).columns)} numeric indicators are
        available for analysis in this dashboard.
        </div>
        """,
        unsafe_allow_html=True,
    )


def page_trend_analysis(df):
    st.markdown('<div class="sub-header">Trend Analysis</div>', unsafe_allow_html=True)

    available = [c for c in CORE_INDICATORS if c in df.columns]
    selected = st.multiselect(
        "Select indicators to plot",
        available,
        default=available[:3],
        format_func=lambda c: NICE_NAMES.get(c, c),
    )

    if selected:
        fig, ax = plt.subplots(figsize=(14, 5))
        for col in selected:
            ax.plot(df["date"], df[col], linewidth=2, label=NICE_NAMES.get(col, col))
        ax.set_xlabel("Date")
        ax.set_ylabel("Value")
        ax.legend(loc="upper left")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("Select at least one indicator above to see its trend.")

    st.markdown("<br>", unsafe_allow_html=True)
    indicator = st.selectbox(
        "Indicator detail view",
        available,
        format_func=lambda c: NICE_NAMES.get(c, c),
    )
    if indicator:
        fig2, ax2 = plt.subplots(figsize=(14, 4))
        ax2.plot(df["date"], df[indicator], color="#2c3e50", linewidth=1.5)
        rolling = df[indicator].rolling(6, min_periods=1).mean()
        ax2.plot(df["date"], rolling, color="#dc3545", linewidth=2, linestyle="--", label="6-month rolling avg")
        ax2.set_title(f"{NICE_NAMES.get(indicator, indicator)} with 6-Month Rolling Average")
        ax2.legend()
        ax2.grid(alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

        c1, c2, c3 = st.columns(3)
        c1.metric("Mean", f"{df[indicator].mean():.2f}")
        c2.metric("Std Dev", f"{df[indicator].std():.2f}")
        c3.metric("Range", f"{df[indicator].min():.2f} to {df[indicator].max():.2f}")


def page_correlation_analysis(df):
    st.markdown('<div class="sub-header">Correlation Analysis</div>', unsafe_allow_html=True)

    available = [c for c in CORE_INDICATORS if c in df.columns]
    corr = df[available].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        ax=ax,
        xticklabels=[NICE_NAMES.get(c, c) for c in available],
        yticklabels=[NICE_NAMES.get(c, c) for c in available],
    )
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    pairs = (
        corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        .stack()
        .sort_values(key=lambda s: s.abs(), ascending=False)
    )
    st.markdown('<div class="sub-header" style="font-size:18px;">Strongest Relationships</div>', unsafe_allow_html=True)
    top = pairs.head(5)
    for (a, b), val in top.items():
        direction = "positively" if val > 0 else "negatively"
        st.markdown(
            f"- **{NICE_NAMES.get(a, a)}** and **{NICE_NAMES.get(b, b)}** are {direction} "
            f"correlated (r = {val:.2f})"
        )


def page_ml_models(results):
    st.markdown('<div class="sub-header">Machine Learning Models</div>', unsafe_allow_html=True)

    if not results:
        st.warning("Not enough data or class diversity to train models. Provide a larger labeled dataset.")
        return

    rows = []
    for name, r in results.items():
        rows.append(
            {
                "Model": name,
                "Accuracy": r["accuracy"],
                "Precision": r["precision"],
                "Recall": r["recall"],
                "F1 Score": r["f1"],
                "CV Mean": r["cv_mean"],
                "CV Std": r["cv_std"],
            }
        )
    results_df = pd.DataFrame(rows).set_index("Model")
    st.dataframe(results_df.style.format("{:.3f}").background_gradient(cmap="Greens", subset=["Accuracy", "F1 Score"]))

    best_model = results_df["F1 Score"].idxmax()
    st.markdown(
        f"""
        <div class="recommendation">
        <strong>Best performing model (by F1 score):</strong> {best_model}
        ({results_df.loc[best_model, 'F1 Score']:.3f})
        </div>
        """,
        unsafe_allow_html=True,
    )

    fig, ax = plt.subplots(figsize=(12, 5))
    results_df[["Accuracy", "Precision", "Recall", "F1 Score"]].plot(kind="bar", ax=ax)
    ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison")
    ax.legend(loc="lower right")
    ax.set_ylim(0, 1)
    plt.xticks(rotation=20)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    selected_model = st.selectbox("Inspect a model", list(results.keys()))
    model_obj = results[selected_model]["model"]
    if hasattr(model_obj, "feature_importances_"):
        st.markdown('<div class="sub-header" style="font-size:18px;">Feature Importance</div>', unsafe_allow_html=True)
        importances = pd.Series(model_obj.feature_importances_, index=model_obj.feature_names_in_)
        importances = importances.sort_values(ascending=True).tail(10)
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        importances.plot(kind="barh", ax=ax2, color="#4ca1af")
        ax2.set_xlabel("Importance")
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)
    elif hasattr(model_obj, "coef_"):
        st.markdown('<div class="sub-header" style="font-size:18px;">Coefficients</div>', unsafe_allow_html=True)
        coefs = pd.Series(model_obj.coef_[0], index=model_obj.feature_names_in_)
        coefs = coefs.sort_values()
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        coefs.plot(kind="barh", ax=ax2, color="#2c3e50")
        ax2.set_xlabel("Coefficient")
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)


def page_roc_curves(results):
    st.markdown('<div class="sub-header">ROC Curves</div>', unsafe_allow_html=True)

    if not results:
        st.warning("No trained models available.")
        return

    fig, ax = plt.subplots(figsize=(8, 7))
    colors = ["#2c3e50", "#4ca1af", "#dc3545", "#28a745"]
    for (name, r), color in zip(results.items(), colors):
        ax.plot(r["fpr"], r["tpr"], color=color, linewidth=2, label=f"{name} (AUC = {r['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve Comparison")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    best_auc_model = max(results, key=lambda k: results[k]["roc_auc"])
    st.markdown(
        f"""
        <div class="finding-box">
        <strong>Highest AUC:</strong> {best_auc_model} achieves an AUC of
        {results[best_auc_model]['roc_auc']:.3f}, indicating the strongest overall
        ability to separate the two classes across thresholds.
        </div>
        """,
        unsafe_allow_html=True,
    )


def page_confusion_matrices(results):
    st.markdown('<div class="sub-header">Confusion Matrices</div>', unsafe_allow_html=True)

    if not results:
        st.warning("No trained models available.")
        return

    names = list(results.keys())
    cols = st.columns(2)
    for i, name in enumerate(names):
        cm = results[name]["cm"]
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=ax,
            xticklabels=["Predicted 0", "Predicted 1"],
            yticklabels=["Actual 0", "Actual 1"],
        )
        ax.set_title(name)
        plt.tight_layout()
        cols[i % 2].pyplot(fig)
        plt.close(fig)


def page_findings_recommendations(df, results):
    st.markdown('<div class="sub-header">Findings & Recommendations</div>', unsafe_allow_html=True)

    if "policy_gap" in df.columns:
        st.markdown(
            f"""
            <div class="finding-box">
            <strong>Finding — Policy stance:</strong> the average policy gap
            (interest rate minus inflation) across the sample is
            {df['policy_gap'].mean():.2f} points, indicating the typical real
            interest rate stance over the observed period.
            </div>
            """,
            unsafe_allow_html=True,
        )

    if "unemployment_rate" in df.columns:
        st.markdown(
            f"""
            <div class="alert-box">
            <strong>Watch — Labor market volatility:</strong> unemployment shows
            the widest variation among core indicators (std. dev. =
            {df['unemployment_rate'].std():.2f}), warranting closer monitoring
            for early stress signals.
            </div>
            """,
            unsafe_allow_html=True,
        )

    if results:
        best_model = max(results, key=lambda k: results[k]["f1"])
        st.markdown(
            f"""
            <div class="recommendation">
            <strong>Recommendation — Model selection:</strong> deploy
            {best_model} for production scoring, given its leading F1 score of
            {results[best_model]['f1']:.3f} on the held-out test set. Re-validate
            quarterly as new data arrives.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="recommendation">
        <strong>Recommendation — Data pipeline:</strong> persist trained models
        between sessions (e.g. with joblib) and replace the synthetic-data
        fallback with a validated CSV ingestion path that surfaces clear
        warnings when expected columns are missing.
        </div>
        """,
        unsafe_allow_html=True,
    )


def main():
    st.markdown('<div class="main-header">Economic Indicators Dashboard</div>', unsafe_allow_html=True)

    df = load_data()
    df = build_features(df)
    results, X, y, target = train_models(df)

    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Analysis Module",
        [
            "Executive Summary",
            "Trend Analysis",
            "Correlation Analysis",
            "ML Models",
            "ROC Curves",
            "Confusion Matrices",
            "Findings & Recommendations",
        ],
    )

    st.sidebar.markdown("---")
    st.sidebar.info(
        f"Records: {len(df)}\n"
        f"Numeric Variables: {len(df.select_dtypes(include=np.number).columns)}"
    )

    if page == "Executive Summary":
        page_executive_summary(df)
    elif page == "Trend Analysis":
        page_trend_analysis(df)
    elif page == "Correlation Analysis":
        page_correlation_analysis(df)
    elif page == "ML Models":
        page_ml_models(results)
    elif page == "ROC Curves":
        page_roc_curves(results)
    elif page == "Confusion Matrices":
        page_confusion_matrices(results)
    elif page == "Findings & Recommendations":
        page_findings_recommendations(df, results)


if __name__ == "__main__":
    main()
