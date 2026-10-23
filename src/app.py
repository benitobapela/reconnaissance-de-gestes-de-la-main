from pathlib import Path
import tempfile

import streamlit as st

from inference import predire


# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="HandGesture",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# DESIGN
# ============================================================

st.markdown("""
<style>

/* IMPORT FONT */
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');


/* GLOBAL */

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background:
        radial-gradient(
            circle at 75% 5%,
            rgba(79, 70, 229, 0.15),
            transparent 27%
        ),
        radial-gradient(
            circle at 15% 40%,
            rgba(37, 99, 235, 0.08),
            transparent 30%
        ),
        #070b14;

    color: #f8fafc;
}

.block-container {
    max-width: 1180px;
    padding-top: 2rem;
    padding-bottom: 5rem;
}


/* TITLES */

h1, h2, h3 {
    font-family: 'Manrope', sans-serif !important;
    color: #f8fafc !important;
}

h1 {
    font-size: 3.7rem !important;
    font-weight: 800 !important;
    letter-spacing: -3px !important;
    line-height: 1.05 !important;
}

h2 {
    font-weight: 700 !important;
    letter-spacing: -1px !important;
}

h3 {
    font-weight: 650 !important;
}


/* TEXT */

p {
    color: #94a3b8;
    font-size: 15px;
    line-height: 1.7;
}

[data-testid="stCaptionContainer"] {
    color: #64748b;
}


/* HEADER BRAND */

.brand-text {
    font-family: 'Manrope', sans-serif;
    font-weight: 800;
    font-size: 18px;
    color: #ffffff;
    letter-spacing: -0.6px;
}

.brand-accent {
    color: #818cf8;
}


/* BADGE */

.badge {
    display: inline-block;

    background: rgba(99, 102, 241, 0.10);
    border: 1px solid rgba(129, 140, 248, 0.25);

    color: #a5b4fc;

    padding: 7px 12px;
    border-radius: 100px;

    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1px;

    margin-top: 55px;
    margin-bottom: 15px;
}


/* HERO */

.hero-title {
    font-family: 'Manrope', sans-serif;

    color: #f8fafc;

    font-size: clamp(48px, 6vw, 76px);
    font-weight: 800;

    line-height: 1.02;
    letter-spacing: -4px;

    max-width: 900px;

    margin-bottom: 25px;
}

.gradient-text {
    background: linear-gradient(
        90deg,
        #818cf8,
        #60a5fa
    );

    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-description {
    color: #94a3b8;

    max-width: 650px;

    font-size: 17px;
    line-height: 1.8;

    margin-bottom: 50px;
}


/* SEPARATORS */

hr {
    border: none !important;
    border-top: 1px solid rgba(148, 163, 184, 0.12) !important;

    margin-top: 40px !important;
    margin-bottom: 40px !important;
}


/* FILE UPLOADER */

[data-testid="stFileUploader"] {
    background: rgba(15, 23, 42, 0.68);

    border: 1px solid rgba(148, 163, 184, 0.13);
    border-radius: 20px;

    padding: 10px;

    backdrop-filter: blur(15px);

    box-shadow:
        0 20px 60px rgba(0, 0, 0, 0.18);
}

[data-testid="stFileUploaderDropzone"] {
    background: rgba(15, 23, 42, 0.40);

    border: 1px dashed rgba(129, 140, 248, 0.35);

    border-radius: 15px;

    padding: 30px;
}

[data-testid="stFileUploaderDropzone"]:hover {
    border-color: #818cf8;

    background:
        rgba(79, 70, 229, 0.06);
}


/* FILE UPLOADER BUTTON */

[data-testid="stFileUploader"] button {
    background: rgba(99, 102, 241, 0.12);

    border: 1px solid rgba(129, 140, 248, 0.25);

    color: #c7d2fe;

    border-radius: 9px;
}


/* MAIN BUTTON */

.stButton > button {
    width: 100%;

    min-height: 52px;

    border: 0;
    border-radius: 12px;

    background: linear-gradient(
        135deg,
        #4f46e5,
        #2563eb
    );

    color: #ffffff;

    font-family: 'DM Sans', sans-serif;

    font-size: 14px;
    font-weight: 700;

    box-shadow:
        0 10px 30px rgba(79, 70, 229, 0.22);

    transition: all 0.2s ease;
}

.stButton > button:hover {
    color: white;

    transform: translateY(-2px);

    box-shadow:
        0 15px 40px rgba(79, 70, 229, 0.32);
}

.stButton > button:active {
    transform: translateY(0);
}


/* METRIC */

[data-testid="stMetric"] {
    background:
        linear-gradient(
            145deg,
            rgba(79, 70, 229, 0.16),
            rgba(15, 23, 42, 0.85)
        );

    border: 1px solid rgba(129, 140, 248, 0.22);

    padding: 25px;

    border-radius: 18px;

    box-shadow:
        0 15px 50px rgba(0, 0, 0, 0.18);
}

[data-testid="stMetricLabel"] {
    color: #94a3b8;
}

[data-testid="stMetricValue"] {
    font-family: 'Manrope', sans-serif;

    color: #ffffff;

    font-size: 30px;

    font-weight: 800;

    letter-spacing: -1px;
}


/* IMAGE */

[data-testid="stImage"] img {
    border-radius: 16px;

    border: 1px solid rgba(148, 163, 184, 0.12);

    box-shadow:
        0 20px 50px rgba(0, 0, 0, 0.20);
}


/* INFO / ALERTS */

[data-testid="stAlert"] {
    background: rgba(15, 23, 42, 0.75);

    border: 1px solid rgba(148, 163, 184, 0.15);

    border-radius: 13px;

    color: #cbd5e1;
}


/* SPINNER */

[data-testid="stSpinner"] {
    color: #a5b4fc;
}


/* HIDE STREAMLIT DEFAULT */

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    background: transparent !important;
}


/* MOBILE */

@media (max-width: 768px) {

    .block-container {
        padding-left: 20px;
        padding-right: 20px;
    }

    .hero-title {
        font-size: 45px;
        letter-spacing: -2.5px;
    }

    .hero-description {
        font-size: 15px;
    }
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# BRAND
# ============================================================

st.markdown(
    """
    <div class="brand-text">
        <span class="brand-accent">HandGesture</span>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HERO
# ============================================================

st.markdown(
    """
    <div class="">
        COMPUTER VISION / MACHINE LEARNING
    </div>

    <div class="hero-title">
        Comprendre un geste.<br>
        <span class="gradient-text">À partir d'une image.</span>
    </div>

    <div class="hero-description">
        Un système de reconnaissance de gestes basé sur la segmentation
        HSV, l'analyse géométrique par moments de Hu et un modèle
        K-Nearest Neighbors.
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# PIPELINE
# ============================================================

st.caption("PIPELINE D'INFÉRENCE")

p1, p2, p3, p4 = st.columns(4)

with p1:
    st.markdown("### 01")
    st.markdown("**Prétraitement**")
    st.caption("Lecture, redimensionnement et conversion HSV")

with p2:
    st.markdown("### 02")
    st.markdown("**Segmentation**")
    st.caption("Isolation de la main et contour principal")

with p3:
    st.markdown("### 03")
    st.markdown("**Extraction**")
    st.caption("Calcul des sept moments invariants de Hu")

with p4:
    st.markdown("### 04")
    st.markdown("**Prédiction**")
    st.caption("StandardScaler et classification K-NN")


st.divider()


# ============================================================
# UPLOAD
# ============================================================

st.caption("NOUVELLE ANALYSE")

st.header("Tester le modèle")

st.write(
    "Importez une photographie contenant un geste de la main. "
    "Le modèle analysera automatiquement sa forme et retournera "
    "la classe prédite."
)

st.write("")

fichier = st.file_uploader(
    "Importer une image",
    type=["jpg", "jpeg", "png", "heic"],
)


# ============================================================
# IMAGE DISPONIBLE
# ============================================================

if fichier is not None:

    st.write("")

    preview, details = st.columns(
        [1.35, 0.65],
        gap="large"
    )

    with preview:

        st.caption("APERÇU")

        st.image(
            fichier,
            use_container_width=True
        )

    with details:

        st.caption("FICHIER")

        st.subheader(fichier.name)

        st.write(
            f"{fichier.size / 1024:.0f} Ko"
        )

        if fichier.type:
            st.caption(fichier.type)

        st.write("")
        st.write("")

        analyser = st.button(
            "Analyser le geste",
            type="primary",
            use_container_width=True
        )


    # ========================================================
    # INFERENCE
    # ========================================================

    if analyser:

        extension = Path(fichier.name).suffix

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=extension
        ) as temp:

            temp.write(
                fichier.getbuffer()
            )

            chemin_temp = temp.name

        try:

            with st.spinner(
                "Traitement et classification en cours..."
            ):

                geste, masque, contour = predire(
                    chemin_temp
                )


            # =================================================
            # RESULT
            # =================================================

            st.divider()

            st.caption("RÉSULTAT DU MODÈLE")

            st.header("Prédiction")

            result_col, description_col = st.columns(
                [0.8, 1.2],
                gap="large"
            )

            with result_col:

                st.metric(
                    label="Geste reconnu",
                    value=geste
                )

            with description_col:

                st.subheader("Classification terminée")

                st.write(
                    "Le pipeline a extrait la forme dominante "
                    "de la main, calculé les sept moments de Hu "
                    "puis transmis ces caractéristiques au modèle K-NN."
                )


            # =================================================
            # VISUAL ANALYSIS
            # =================================================

            st.divider()

            st.caption("VISION PAR ORDINATEUR")

            st.header("Ce que le modèle analyse")

            st.write(
                "La segmentation permet de visualiser la région "
                "à partir de laquelle les caractéristiques "
                "géométriques ont été calculées."
            )

            st.write("")

            image_col, mask_col = st.columns(
                2,
                gap="large"
            )

            with image_col:

                st.caption("IMAGE ORIGINALE")

                st.image(
                    fichier,
                    use_container_width=True
                )

            with mask_col:

                st.caption("MASQUE SEGMENTÉ")

                st.image(
                    masque,
                    use_container_width=True,
                    clamp=True
                )


            # =================================================
            # TECHNICAL SECTION
            # =================================================

            st.divider()

            st.caption("SOUS LE CAPOT")

            st.header("Architecture de reconnaissance")

            t1, t2, t3 = st.columns(
                3,
                gap="large"
            )

            with t1:

                st.subheader("HSV")

                st.write(
                    "L'espace colorimétrique HSV facilite "
                    "l'isolation des pixels correspondant "
                    "à la région de la main."
                )

            with t2:

                st.subheader("Hu Moments")

                st.write(
                    "Sept descripteurs numériques représentent "
                    "la géométrie de la forme détectée."
                )

            with t3:

                st.subheader("K-NN")

                st.write(
                    "Le modèle recherche les exemples appris "
                    "les plus proches pour déterminer "
                    "la classe du geste."
                )


        except ValueError as erreur:

            st.error(
                f"Impossible d'analyser cette image : {erreur}"
            )

        except FileNotFoundError as erreur:

            st.error(
                f"Ressource introuvable : {erreur}"
            )

        except Exception as erreur:

            st.error(
                f"Erreur pendant l'analyse : {erreur}"
            )


else:

    st.info(
        "Importez une image JPG, JPEG, PNG ou HEIC pour commencer."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

footer_left, footer_right = st.columns(2)

with footer_left:
    st.caption("HANDGESTURE")

with footer_right:
    st.caption(
        "Computer Vision / Hu Moments / K-NN"
    )