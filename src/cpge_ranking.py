import pandas as pd
import streamlit as st
import plotly.express as px
import numpy as np
import uuid
import math

# Predefined CSV file path
CSV_FILE_PATH = "./data/joined_file_mp_pc_ecg.csv"

# Subject names and default notes
DEFAULT_NOTES = {
    "Maths": 16.5,
    "Physique-chimie": 17.5,
    "LVA": 18.0,
    "LVB": 12.0,
    "Histoire-geo": 16.5,
    "Philosophie": 14.5,
    "DNL": 15.0,
    "EPS": 19.5,
    "Enseignement Scientifique": 17.0,
    "Maths expertes": 15.0,
    "SVT": -1,
    "Sciences économiques": -1,
}

# Map French acceptance probability classes to numerical weights
ACCEPTANCE_CLASSES = {
    "Rarement": 1,
    "Occasionnellement": 2,
    "Régulièrement": 3,
    "Plus de 50%": 4,
    "Plus de 80%": 5,
}

# Assign acceptance probability to classes
APW_TO_PROB = {
    1: 0.05,  # <5%
    2: 0.15,  # >15%
    3: 0.33,  # >33%
    4: 0.50,  # >50%
    5: 0.80   # >80%
}


def extract_coordinates(dataframe, gps_column):
    """
    Extract latitude and longitude from the GPS column.

    Args:
        dataframe (pd.DataFrame): Input DataFrame with a GPS coordinates column.
        gps_column (str): Name of the column containing GPS coordinates.

    Returns:
        pd.DataFrame: DataFrame with added latitude and longitude columns.
    """
    if gps_column not in dataframe.columns:
        raise ValueError(f"Column '{gps_column}' does not exist in the DataFrame.")

    # Ensure the GPS column contains strings
    dataframe[gps_column] = dataframe[gps_column].astype(str)

    # Split the GPS column into latitude and longitude
    coords = dataframe[gps_column].str.split(",", expand=True)

    # Add latitude and longitude columns
    dataframe["latitude"] = pd.to_numeric(coords[0], errors="coerce")
    dataframe["longitude"] = pd.to_numeric(coords[1], errors="coerce")

    # Check for missing values
    if dataframe[["latitude", "longitude"]].isnull().any().any():
        st.warning("Some GPS coordinates could not be processed. They have been set to NaN.")

    return dataframe




def calculate_weighted_average(notes, weights):
    """Calculate the weighted average of the student's notes, excluding subjects with a note of -1."""
    total_weighted_score = sum(note * weights[subject] for subject, note in notes.items() if note != -1)
    total_weights = sum(weights[subject] for subject, note in notes.items() if note != -1)
    return total_weighted_score / total_weights if total_weights > 0 else 0


def get_acceptance_probability(df, average_note):
    """Get acceptance probabilities for each university based on the average note."""
    column = (
        "moy_gen_18_value" if average_note >= 18 else
        "moy_gen_17_value" if average_note >= 17 else
        "moy_gen_16_value" if average_note >= 16 else
        "moy_gen_15_value"
    )
    df["APW"] = df[column].map(ACCEPTANCE_CLASSES)
    return df


def rank_universities(df, notes, csv_file, top_n, univ_type, subject_weights, sss_weights, filter_internat, filter_public, selected_regions):
    # Load the university data
    # df = pd.read_csv(csv_file, delimiter=";")

    # Normalize column names to avoid case issues
    df.columns = df.columns.str.strip()

    # Filter by CPGE type and region
    df = df[df["Filière de formation détaillée bis"].str.lower() == univ_type.lower()]
    if selected_regions:
        df = df[df["Région de l’établissement"].isin(selected_regions)]

    # Drop rows with missing key data
    target_columns = ["Taux", "Taux d’accès", "moy_gen_15_value", "moy_gen_16_value", "moy_gen_17_value", "moy_gen_18_value"]
    df = df.dropna(subset=target_columns)

    # Convert percentages to numeric
    for col in ["Taux", "Taux d’accès"]:
        df[col] = df[col].astype(str).str.replace('%', '', regex=False).str.replace(',', '.', regex=False)
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Calculate weighted average for student notes
    average_note = calculate_weighted_average(notes, subject_weights)

    # Map acceptance probabilities to APW
    df = get_acceptance_probability(df, average_note)

    # Calculate normalized scores and SSS
    df["Normalized_AR"] = df["Taux d’accès"] / 100
    df["Normalized_QR"] = df["Taux"] / 100
    # df["APW_norm"] = (df["APW"] - 1) / 4.0
    df["Proba d'admission"] = df["APW"].map(APW_TO_PROB)

    # Now APW_prob is in [0..1], similar to Normalized_AR and Normalized_QR
    df["SSS"] = (
            df["Proba d'admission"] * sss_weights["apw"] +
            df["Normalized_AR"] * sss_weights["access_rate"] +
            df["Normalized_QR"] * sss_weights["quality_rate"]
    )

    # Debug all rows for SSS calculation
    # debug_sss_df = df[["Établissement", "Commune de l’établissement", "Taux d’accès", "Taux", "APW", "SSS", "latitude", "longitude"]]

    # Apply additional filters
    if filter_internat:
        df = df[df["Effectif des candidats classés par l’établissement en internat (CPGE)"] > 0]
    if filter_public:
        df = df[df["Statut de l’établissement de la filière de formation (public, privé…)"].str.lower() == "public"]

    # Rank and select top N
    ranked_df = df.sort_values(by="SSS", ascending=False).head(top_n)
    ranked_df.rename(columns={"Taux": "Taux de réussite"}, inplace=True)
    ranked_df.rename(columns={"Commune de l’établissement": "Commune"}, inplace=True)

    # return ranked_df, average_note, debug_sss_df
    return df, ranked_df, average_note



def plot_universities_map(dataframe, selected_regions):
    """
    Plot a map of universities in France based on their geographic locations.

    Args:
        dataframe (pd.DataFrame): DataFrame with university data, including latitude and longitude.
        selected_regions (list): List of selected regions for filtering universities.

    Returns:
        None
    """
    # Filter by selected regions if provided
    if selected_regions:
        dataframe = dataframe[dataframe["Région de l’établissement"].isin(selected_regions)]

    # Drop rows without geographic data
    dataframe = dataframe.dropna(subset=["latitude", "longitude", "SSS"])

    # Ensure latitude and longitude are numeric
    dataframe["latitude"] = pd.to_numeric(dataframe["latitude"], errors="coerce")
    dataframe["longitude"] = pd.to_numeric(dataframe["longitude"], errors="coerce")

    # # Debug: Display DataFrame to check structure
    # st.write("Data for Map Plotting:")
    # st.dataframe(dataframe[["Établissement", "latitude", "longitude", "SSS"]].head())

    # Create a map plot
    fig = px.scatter_mapbox(
        dataframe,
        lat="latitude",
        lon="longitude",
        hover_name="Établissement",
        hover_data=["Commune", "Taux de réussite", "Taux d’accès"],
        color="SSS",  # Color by SSS score
        size="SSS",   # Size by SSS score
        size_max=10,  # Reduces maximum bubble size
        center={"lat": 46.5, "lon": 2.0},  # Center roughly over France
        zoom=4,  # Set initial zoom level
        # title="Carte des CPGE sélectionnées",
        mapbox_style="carto-positron"
    )

    st.plotly_chart(fig)


def plot_universities_graph(file_path, debug_sss_df, univ_type, top_n):
    """
    Plot a graph of all universities with 1 - Taux d'accès on the x-axis and Taux on the y-axis,
    using Plotly for interactivity and hover labels. The Top-N universities are colorized with a red gradient
    based on their enhanced SSS score, while all other universities are in a grey-to-black gradient.

    Args:
        file_path (str): Path to the CSV file with university data.
        debug_sss_df (pd.DataFrame): Debug DataFrame containing SSS scores for all universities.
        univ_type (str): The selected CPGE Type (e.g., "mpsi", "pcsi", "ecg").
        top_n (int): Number of top universities to highlight in red.
    """
    # Use the debug data for plotting
    df = debug_sss_df.copy()

    # Add enhanced SSS score for all universities
    df["Enhanced_SSS"] = np.exp(df["SSS"].fillna(0)) - 1  # Exponential scaling for SSS

    # Compute x-axis as 1 - Taux d’accès
    df["1-Taux d’accès"] = 1 - (df["Taux d’accès"] / 100)

    # Identify Top-N universities
    top_n_universities = df.nlargest(top_n, "Enhanced_SSS")
    top_n_indices = top_n_universities.index

    # Separate Top-N and Other universities
    df["Group"] = "Other"
    df.loc[top_n_indices, "Group"] = "Top-N"

    # Apply transformations for color scaling
    df["Scaled_Color"] = np.where(
        df["Group"] == "Top-N",
        np.log1p(df["Enhanced_SSS"]),  # Log-scaled for Top-N
        -np.log1p(df["Enhanced_SSS"])  # Negative log-scaled for Others
    )

    # Apply exponential scaling for better contrast
    df["Scaled_Color"] = np.exp(df["Scaled_Color"]) - 1

    # Create scatter plot
    fig = px.scatter(
        df,
        x="1-Taux d’accès",
        y="Taux",
        color="Scaled_Color",
        hover_name="Établissement",
        title="Positionnement de la sélection (en rouge) au sein de l'ensemble des CPGE",
        labels={
            "1-Taux d’accès": "Difficulté d'accès",
            "Taux": "Taux de réussite aux concours",
            "Scaled_Color": "Enhanced SSS"
        },
        color_continuous_scale=[
            (0.0, "lightgray"),  # Light grey for lowest scores (Other)
            (0.3, "black"),      # Dark grey for higher scores (Other)
            (0.8, "pink"),       # Light pink for lower scores (Top-N)
            (1.0, "red")         # Bright red for highest scores (Top-N)
        ],
        range_color=[df["Scaled_Color"].min(), df["Scaled_Color"].max()]
    )

    # Update marker size and layout for better readability
    fig.update_traces(marker=dict(size=10, opacity=0.8), selector=dict(mode="markers"))
    fig.update_layout(coloraxis_colorbar=dict(title="SSS Gradient"))

    # Display the Plotly chart in Streamlit
    st.plotly_chart(fig)


def main():
    st.title("CPGE Ranking App")
    st.write(f"""
Découvrez quelles CPGE correspondent le mieux à votre profil en fonction de vos notes, de la sélectivité et des taux de réussite aux concours. 
Comparez les établissements, estimez vos chances d’admission et optimisez votre stratégie de classement Parcoursup.

*Données exploitées : Parcoursup 2023 et 2024; l'Etudiant 2024.*""")

    # Initialize session_state defaults if not set
    if "apw_weight" not in st.session_state:
        st.session_state.apw_weight = 0.2
    if "access_rate_weight" not in st.session_state:
        st.session_state.access_rate_weight = 0.3
    if "quality_rate_weight" not in st.session_state:
        st.session_state.quality_rate_weight = 0.5

    # Input fields for notes and weights
    st.header("Entrez vos notes et choisissez les coefficients")
    st.markdown("**Note:** Entrez `-1` pour les matières non suivies ou que vous souhaitez exclure du calcul.")
    student_notes = {}
    subject_weights = {}

    for subject, default_note in DEFAULT_NOTES.items():
        col1, col2 = st.columns([2, 1])
        with col1:
            student_notes[subject] = st.number_input(
                f"{subject} Note:",
                min_value=-1.0,
                max_value=20.0,
                step=0.1,
                value=float(default_note),  # Ensure the default value is a float
                key=f"note_{subject}"
            )
        with col2:
            subject_weights[subject] = st.number_input(
                f"{subject} Weight:",
                min_value=0.0,
                max_value=10.0,
                step=0.5,
                value=1.0,  # Default weight as a float
                key=f"weight_{subject}"
            )

    # Input fields for other parameters
    univ_type = st.selectbox("CPGE Type:", ["mpsi", "pcsi", "ecg"])
    top_n = st.number_input("Top N CPGE:", min_value=1, step=1, value=20)

    # Sliders for SSS weights
    st.header("Ajustez les poids pour le calcul des scores")

    st.subheader("Ou choisissez un profil :")
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Prudent"):
            st.session_state.apw_weight = 0.6
            st.session_state.access_rate_weight = 0.15
            st.session_state.quality_rate_weight = 0.25
            st.query_params = {"force": str(uuid.uuid4())}

    with col2:
        if st.button("Equilibré"):
            st.session_state.apw_weight = 0.35
            st.session_state.access_rate_weight = 0.15
            st.session_state.quality_rate_weight = 0.5
            st.query_params = {"force": str(uuid.uuid4())}

    with col3:
        if st.button("Audacieux"):
            st.session_state.apw_weight = 0.05
            st.session_state.access_rate_weight = 0.15
            st.session_state.quality_rate_weight = 0.8
            st.query_params = {"force": str(uuid.uuid4())}

    apw_weight = st.slider("APW : Probabilité d'avoir une proposition d'admission selon sa moyenne générale (données 2024)",
                           min_value=0.0, max_value=1.0, step=0.05, key="apw_weight")
    access_rate_weight = st.slider("Normalized Access Rate Weight : Sélectivité à l'accès (données 2023)",
                                   min_value=0.0, max_value=1.0, step=0.05, key="access_rate_weight")
    quality_rate_weight = st.slider("Normalized Quality Rate Weight : Taux de réussite aux concours (données 2024)",
                                    min_value=0.0, max_value=1.0, step=0.05, key="quality_rate_weight")


    # Ensure the sum of weights equals 1
    total_weight = apw_weight + access_rate_weight + quality_rate_weight
    if not math.isclose(total_weight, 1.0, abs_tol=1e-9):
        st.warning(
            f"The total weight is {total_weight:.2f}. "
            "Adjust the sliders to ensure the total equals 1.0."
        )
    # Add checkboxes for filters
    filter_internat = st.checkbox("Internat")
    filter_public = st.checkbox("Public")

    # Load data and extract coordinates
    df = pd.read_csv(CSV_FILE_PATH, delimiter=";")
    # if "Coordonnées GPS de la formation" not in df.columns:
    #     st.error("Column 'Coordonnées GPS de la formation' is missing from the dataset.")
    # else:
    #     st.write("GPS Data Sample:")
    #     st.write(df["Coordonnées GPS de la formation"].head())
    df = extract_coordinates(df, "Coordonnées GPS de la formation")

    # st.write("Extracted Coordinates:")
    # if "latitude" in df.columns and "longitude" in df.columns:
    #     st.dataframe(df[["latitude", "longitude"]].head())
    # else:
    #     st.error("Latitude and Longitude columns were not created.")

    # Add region filter
    all_regions = sorted(df["Région de l’établissement"].dropna().unique())
    selected_regions = st.multiselect("Filtrez par Région:", all_regions)

    # Run analysis
    if st.button("Calculer les scores") and total_weight == 1.0:
        try:
            df_with_sss, ranked_universities, avg_note = rank_universities(
                df, student_notes, CSV_FILE_PATH, top_n, univ_type, subject_weights,
                {"apw": apw_weight, "access_rate": access_rate_weight, "quality_rate": quality_rate_weight},
                filter_internat, filter_public, selected_regions
            )
            # Display the student's average note
            st.write(f"Moyenne pondérée: {avg_note:.2f}")

            st.subheader("CPGE sélectionnées")
            columns_to_show = [
                "Établissement",
                "Commune",
                "Proba d'admission",
                "Taux d’accès",
                "Taux de réussite",  # This is the renamed "Taux"
                "SSS"
            ]

            st.dataframe(ranked_universities[columns_to_show], hide_index=True)

            # st.write("Data for Map Plotting:")
            # st.dataframe(df_with_sss[["Établissement", "latitude", "longitude", "SSS"]].head())

            # st.subheader("Map of Selected CPGE")
            plot_universities_map(ranked_universities, selected_regions)

            # Add a graph to represent the universities
            # st.subheader("University Rankings Visualization")
            plot_universities_graph(CSV_FILE_PATH, df_with_sss, univ_type, top_n)

            # Add explanations
            st.write("### Explications :")
            st.write(f"""
                            **SSS (Student-Specific Score):** Combine les éléments suivants pondérés par les poids renseignés par l'utilisateur :
                            - APW (Acceptance Probability Weight): {apw_weight:.2f}
                            - Normalized Access Rate: {access_rate_weight:.2f}
                            - Normalized Quality Rate: {quality_rate_weight:.2f}
                            - Formule : SSS = (APW * {apw_weight:.2f}) + (Normalized Access Rate * {access_rate_weight:.2f}) + (Normalized Quality Rate * {quality_rate_weight:.2f})
                            
                            *NB: le taux de réussite aux concours se base sur les résultats des écoles du haut du panier (X, ENS, Centrales, Mines, Top CCINP); un taux de réussite de 20% ne signifie donc pas que seuls 20% des élèves ont été admis dans une école d'ingénieur, il reflète davantage la qualité de formation et le niveaux des élèves de la CPGE...* 
                        """)

        except Exception as e:
            st.error(f"Error: {e}")


if __name__ == "__main__":
    main()
