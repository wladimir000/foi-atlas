# ============================================================
# OPENLENS
# 10. STREAMLIT-APP MIT CHROMADB UND MISTRAL
# ============================================================
#
# DATEINAME:
# OpenLens/10_OpenLens_App.py
#
# START IM VS-CODE-TERMINAL:
#
# python -m streamlit run 10_OpenLens_App.py
#
# WICHTIG:
# Diese Datei ist eine normale Python-Datei.
# Sie wird NICHT in einem Jupyter-Notebook ausgeführt.
#
# Erwartete Projektstruktur:
#
# OpenLens/
# ├── 10_OpenLens_App.py
# ├── ChromaDB/
# │   ├── chroma.sqlite3
# │   └── openlens_chromadb_manifest.json
# ├── Modelle/
# │   ├── paraphrase-multilingual-MiniLM-L12-v2/
# │   └── Mistral-7B-Instruct-v0.3/
# ├── RAG_Ergebnisse/
# └── Datenbank/
#
# ============================================================


# ============================================================
# 1. FEHLENDE PAKETE AUTOMATISCH INSTALLIEREN
# ============================================================

import importlib
import subprocess
import sys


BENOETIGTE_PAKETE = {
    "streamlit": "streamlit",
    "chromadb": "chromadb",
    "sentence_transformers": "sentence-transformers",
    "transformers": "transformers",
    "accelerate": "accelerate",
    "sentencepiece": "sentencepiece",
    "huggingface_hub": "huggingface-hub",
    "pandas": "pandas",
    "numpy": "numpy",
}


def paket_installieren_falls_noetig(
    import_name: str,
    paket_name: str,
) -> None:
    """
    Installiert ein fehlendes Python-Paket im aktuell
    verwendeten Environment.
    """

    try:
        importlib.import_module(import_name)

    except ImportError:

        print(
            f"Das Paket '{paket_name}' fehlt "
            "und wird jetzt installiert ..."
        )

        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--upgrade",
                paket_name,
            ]
        )


for import_name, paket_name in BENOETIGTE_PAKETE.items():

    paket_installieren_falls_noetig(
        import_name=import_name,
        paket_name=paket_name,
    )


try:
    import torch

except ImportError as fehler:

    raise ImportError(
        "\nPyTorch ist nicht installiert.\n\n"
        "Installiere PyTorch passend zu deiner Hardware. "
        "PyTorch wird bewusst nicht automatisch installiert, "
        "damit keine vorhandene CUDA-Version überschrieben wird."
    ) from fehler


# ============================================================
# 2. IMPORTE
# ============================================================

import io
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import chromadb
import numpy as np
import pandas as pd
import streamlit as st
import torch

from huggingface_hub import snapshot_download
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)


# ============================================================
# 3. STREAMLIT-SEITENKONFIGURATION
# ============================================================

st.set_page_config(
    page_title="OpenLens",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# 4. EINSTELLUNGEN
# ============================================================

COLLECTION_NAME = "fragdenstaat_openlens"

EMBEDDING_MODELL_ORDNERNAME = (
    "paraphrase-multilingual-MiniLM-L12-v2"
)

MISTRAL_REPOSITORY = (
    "mistralai/Mistral-7B-Instruct-v0.3"
)

MISTRAL_ORDNERNAME = (
    "Mistral-7B-Instruct-v0.3"
)

INITIAL_TOP_K_STANDARD = 15

FINAL_TOP_K_STANDARD = 5

MIN_SIMILARITY_STANDARD = 0.20

MAX_CHARACTERS_PER_SOURCE_STANDARD = 2200

MAX_CONTEXT_CHARACTERS = 10000

MAX_NEW_TOKENS_STANDARD = 400

TEMPERATURE_STANDARD = 0.0

REPETITION_PENALTY = 1.08

USE_FLOAT16_ON_CUDA = True


# ============================================================
# 5. PROJEKTPFADE
# ============================================================

SCRIPT_PFAD = Path(__file__).resolve()

PROJEKTORDNER = SCRIPT_PFAD.parent

CHROMA_ORDNER = (
    PROJEKTORDNER
    / "ChromaDB"
)

MODELL_ORDNER = (
    PROJEKTORDNER
    / "Modelle"
)

RAG_ORDNER = (
    PROJEKTORDNER
    / "RAG_Ergebnisse"
)

LOKALES_EMBEDDING_MODELL = (
    MODELL_ORDNER
    / EMBEDDING_MODELL_ORDNERNAME
)

LOKALES_MISTRAL_MODELL = (
    MODELL_ORDNER
    / MISTRAL_ORDNERNAME
)

CHROMA_MANIFEST = (
    CHROMA_ORDNER
    / "openlens_chromadb_manifest.json"
)


MODELL_ORDNER.mkdir(
    parents=True,
    exist_ok=True,
)

RAG_ORDNER.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 6. APP-DESIGN
# ============================================================

st.markdown(
    """
    <style>
        .block-container {
            max-width: 1450px;
            padding-top: 1.8rem;
            padding-bottom: 4rem;
        }

        .openlens-header {
            border-bottom: 1px solid rgba(128, 128, 128, 0.28);
            padding-bottom: 1.2rem;
            margin-bottom: 1.5rem;
        }

        .openlens-title {
            font-size: 2.6rem;
            font-weight: 750;
            margin: 0;
        }

        .openlens-subtitle {
            opacity: 0.72;
            margin-top: 0.3rem;
            margin-bottom: 0;
        }

        .answer-box {
            border: 1px solid rgba(128, 128, 128, 0.28);
            border-radius: 12px;
            padding: 1.3rem 1.5rem;
            margin-top: 0.8rem;
            margin-bottom: 1.3rem;
        }

        .status-box {
            border: 1px solid rgba(38, 166, 91, 0.35);
            background: rgba(38, 166, 91, 0.10);
            border-radius: 8px;
            padding: 0.7rem 0.9rem;
        }

        div[data-testid="stMetric"] {
            border: 1px solid rgba(128, 128, 128, 0.18);
            padding: 0.65rem;
            border-radius: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 7. ALLGEMEINE HILFSFUNKTIONEN
# ============================================================

def sicherer_textwert(
    wert: Any,
) -> str:
    """
    Wandelt beliebige Werte sicher in Text um.
    """

    if wert is None:
        return ""

    try:

        if pd.isna(wert):
            return ""

    except Exception:
        pass

    return str(wert).strip()


def normalisiere_text(
    text: str,
) -> str:
    """
    Normalisiert Text zur Duplikaterkennung.
    """

    text = sicherer_textwert(
        text
    ).lower()

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def seitenwert_formatieren(
    wert: Any,
) -> str:
    """
    Entfernt unnötige Nachkommastellen aus Seitenzahlen.
    """

    text = sicherer_textwert(
        wert
    )

    if text.endswith(".0"):
        text = text[:-2]

    return text


def seitenangabe_erstellen(
    page_start: Any,
    page_end: Any,
) -> str:
    """
    Erstellt eine lesbare Seitenangabe.
    """

    start = seitenwert_formatieren(
        page_start
    )

    ende = seitenwert_formatieren(
        page_end
    )

    if not start and not ende:
        return "Seite nicht ermittelt"

    if start and (
        not ende
        or start == ende
    ):
        return f"Seite {start}"

    if not start and ende:
        return f"Seite {ende}"

    return f"Seiten {start}–{ende}"


def chroma_manifest_laden() -> dict:
    """
    Lädt das ChromaDB-Manifest, falls es vorhanden ist.
    """

    if not CHROMA_MANIFEST.exists():
        return {}

    with CHROMA_MANIFEST.open(
        "r",
        encoding="utf-8",
    ) as datei:

        return json.load(datei)


def lokales_modell_vollstaendig(
    modellordner: Path,
) -> bool:
    """
    Prüft, ob Mistral lokal vollständig vorhanden ist.
    """

    if not modellordner.is_dir():
        return False

    config_vorhanden = (
        modellordner
        / "config.json"
    ).exists()

    tokenizer_vorhanden = any(
        (
            modellordner
            / dateiname
        ).exists()
        for dateiname in [
            "tokenizer.json",
            "tokenizer.model",
            "tokenizer_config.json",
        ]
    )

    index_vorhanden = (
        modellordner
        / "model.safetensors.index.json"
    ).exists()

    modell_shards = list(
        modellordner.glob(
            "model-*.safetensors"
        )
    )

    einzelnes_modell = (
        modellordner
        / "model.safetensors"
    ).exists()

    gewichte_vorhanden = (
        index_vorhanden
        or bool(modell_shards)
        or einzelnes_modell
    )

    return (
        config_vorhanden
        and tokenizer_vorhanden
        and gewichte_vorhanden
    )


# ============================================================
# 8. CHROMADB LADEN
# ============================================================

@st.cache_resource(
    show_spinner=False,
)
def chromadb_laden(
    chroma_pfad: str,
    collection_name: str,
):
    """
    Öffnet ChromaDB und lädt die angeforderte Collection.
    """

    client = chromadb.PersistentClient(
        path=chroma_pfad
    )

    vorhandene_namen = []

    for eintrag in client.list_collections():

        name = getattr(
            eintrag,
            "name",
            None,
        )

        if name is None:
            name = str(eintrag)

        vorhandene_namen.append(name)

    if collection_name not in vorhandene_namen:

        raise ValueError(
            "Die erwartete ChromaDB-Collection "
            "wurde nicht gefunden.\n\n"
            f"Erwartet: {collection_name}\n"
            f"Vorhanden: {vorhandene_namen}"
        )

    collection = client.get_collection(
        name=collection_name
    )

    return client, collection


# ============================================================
# 9. EMBEDDING-MODELL LADEN
# ============================================================

@st.cache_resource(
    show_spinner=False,
)
def embedding_modell_laden(
    modellpfad: str,
    geraet: str,
):
    """
    Lädt das lokale Sentence-Transformer-Modell.
    """

    return SentenceTransformer(
        modellpfad,
        device=geraet,
    )


# ============================================================
# 10. MISTRAL HERUNTERLADEN
# ============================================================

def mistral_bereitstellen() -> None:
    """
    Lädt Mistral bei Bedarf einmalig herunter.
    """

    if lokales_modell_vollstaendig(
        LOKALES_MISTRAL_MODELL
    ):
        return

    dateimuster = [
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "model.safetensors.index.json",
        "model-*.safetensors",
        "*.py",
    ]

    with st.spinner(
        "Mistral-7B wird einmalig heruntergeladen. "
        "Der Download umfasst ungefähr 14,5 GB."
    ):

        snapshot_download(
            repo_id=MISTRAL_REPOSITORY,
            local_dir=str(
                LOKALES_MISTRAL_MODELL
            ),
            allow_patterns=dateimuster,
        )

    if not lokales_modell_vollstaendig(
        LOKALES_MISTRAL_MODELL
    ):

        raise FileNotFoundError(
            "Der lokale Mistral-Modellordner ist "
            "nach dem Download unvollständig."
        )


# ============================================================
# 11. MISTRAL LADEN
# ============================================================

@st.cache_resource(
    show_spinner=False,
)
def mistral_laden(
    modellpfad: str,
):
    """
    Lädt Tokenizer und Mistral-Modell.
    """

    tokenizer = AutoTokenizer.from_pretrained(
        modellpfad,
        local_files_only=True,
        use_fast=True,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    cuda_verfuegbar = torch.cuda.is_available()

    if cuda_verfuegbar:

        datentyp = (
            torch.float16
            if USE_FLOAT16_ON_CUDA
            else torch.float32
        )

    else:

        datentyp = torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        modellpfad,
        dtype=datentyp,
        device_map="auto",
        local_files_only=True,
        low_cpu_mem_usage=True,
    )

    model.eval()

    eingabe_geraet = (
        model
        .get_input_embeddings()
        .weight
        .device
    )

    return (
        tokenizer,
        model,
        eingabe_geraet,
        next(model.parameters()).dtype,
    )


# ============================================================
# 12. SEMANTISCHE SUCHE
# ============================================================

def semantische_suche(
    frage: str,
    collection,
    embedding_model,
    initial_top_k: int,
    final_top_k: int,
    min_similarity: float,
) -> pd.DataFrame:
    """
    Sucht semantisch passende Chunks in ChromaDB.
    """

    frage = sicherer_textwert(
        frage
    )

    if not frage:

        raise ValueError(
            "Die Frage darf nicht leer sein."
        )

    frage_embedding = embedding_model.encode(
        [frage],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    frage_embedding = np.asarray(
        frage_embedding,
        dtype=np.float32,
    )

    suchanzahl = min(
        int(initial_top_k),
        collection.count(),
    )

    roh_ergebnis = collection.query(
        query_embeddings=frage_embedding.tolist(),
        n_results=suchanzahl,
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    ids = roh_ergebnis.get(
        "ids",
        [[]],
    )[0]

    dokumente = roh_ergebnis.get(
        "documents",
        [[]],
    )[0]

    metadaten_liste = roh_ergebnis.get(
        "metadatas",
        [[]],
    )[0]

    distanzen = roh_ergebnis.get(
        "distances",
        [[]],
    )[0]

    treffer = []

    bereits_gesehene_textteile = set()

    for (
        chunk_id,
        dokument,
        metadaten,
        distanz,
    ) in zip(
        ids,
        dokumente,
        metadaten_liste,
        distanzen,
    ):

        dokument = sicherer_textwert(
            dokument
        )

        metadaten = metadaten or {}

        similarity_score = (
            1.0
            - float(distanz)
        )

        if similarity_score < float(
            min_similarity
        ):
            continue

        duplikat_schluessel = (
            normalisiere_text(
                dokument
            )[:1200]
        )

        if (
            duplikat_schluessel
            in bereits_gesehene_textteile
        ):
            continue

        bereits_gesehene_textteile.add(
            duplikat_schluessel
        )

        treffer.append(
            {
                "chunk_id": chunk_id,
                "similarity_score": round(
                    similarity_score,
                    4,
                ),
                "distance": round(
                    float(distanz),
                    4,
                ),
                "document_id": metadaten.get(
                    "document_id",
                    "",
                ),
                "title": metadaten.get(
                    "title",
                    "",
                ),
                "page_start": metadaten.get(
                    "page_start",
                    "",
                ),
                "page_end": metadaten.get(
                    "page_end",
                    "",
                ),
                "site_url": metadaten.get(
                    "site_url",
                    "",
                ),
                "file_url": metadaten.get(
                    "file_url",
                    "",
                ),
                "pdf_name": metadaten.get(
                    "pdf_name",
                    "",
                ),
                "text": dokument,
            }
        )

        if len(treffer) >= int(
            final_top_k
        ):
            break

    return pd.DataFrame(treffer)


# ============================================================
# 13. QUELLENKONTEXT ERSTELLEN
# ============================================================

def quellenkontext_erstellen(
    treffer_df: pd.DataFrame,
    max_characters_per_source: int,
) -> str:
    """
    Formatiert die Treffer als nummerierte Quellenblöcke.
    """

    quellenbloecke = []

    bisherige_zeichen = 0

    for quellennummer, (_, row) in enumerate(
        treffer_df.iterrows(),
        start=1,
    ):

        titel = sicherer_textwert(
            row.get(
                "title",
                "",
            )
        ) or "Ohne Titel"

        dokument_id = sicherer_textwert(
            row.get(
                "document_id",
                "",
            )
        )

        seitenangabe = seitenangabe_erstellen(
            row.get(
                "page_start",
                "",
            ),
            row.get(
                "page_end",
                "",
            ),
        )

        text = sicherer_textwert(
            row.get(
                "text",
                "",
            )
        )

        text = text[
            :int(max_characters_per_source)
        ].strip()

        quellenblock = (
            f"[QUELLE {quellennummer}]\n"
            f"Titel: {titel}\n"
            f"Dokument-ID: {dokument_id}\n"
            f"Fundstelle: {seitenangabe}\n"
            f"Text:\n{text}\n"
            f"[/QUELLE {quellennummer}]"
        )

        verbleibender_platz = (
            MAX_CONTEXT_CHARACTERS
            - bisherige_zeichen
        )

        if verbleibender_platz <= 0:
            break

        if len(quellenblock) > verbleibender_platz:

            quellenblock = quellenblock[
                :verbleibender_platz
            ].rstrip()

        quellenbloecke.append(
            quellenblock
        )

        bisherige_zeichen += len(
            quellenblock
        )

    return "\n\n".join(
        quellenbloecke
    )


# ============================================================
# 14. MISTRAL-PROMPT
# ============================================================

SYSTEM_PROMPT = """
Du bist OpenLens, ein sachlicher Assistent zur Analyse von
Behörden- und Informationsfreiheitsdokumenten.

Verbindliche Regeln:

1. Verwende ausschließlich Informationen aus den bereitgestellten Quellen.
2. Erfinde keine Tatsachen, Namen, Daten, Seiten oder Zusammenhänge.
3. Wenn die Quellen nicht ausreichen, sage das ausdrücklich.
4. Belege wesentliche Aussagen mit [Quelle 1], [Quelle 2] usw.
5. Unterscheide dokumentierte Aussagen von vorsichtigen Schlussfolgerungen.
6. Antworte auf Deutsch.
7. Formuliere nüchtern, verständlich und präzise.
8. Behaupte nicht, vollständige Dokumente geprüft zu haben,
   wenn nur Ausschnitte bereitgestellt wurden.
9. Verwende kein externes Wissen.
10. Verweise niemals auf nicht bereitgestellte Quellen.
""".strip()


def mistral_nachrichten_erstellen(
    frage: str,
    quellenkontext: str,
) -> list[dict]:
    """
    Erstellt eine Mistral-kompatible Chatnachricht.
    """

    gesamter_prompt = f"""
ANWEISUNGEN:
{SYSTEM_PROMPT}

FRAGE:
{frage}

QUELLEN:
{quellenkontext}

AUFGABE:
- Beginne mit einer direkten Antwort.
- Erläutere anschließend die wichtigsten Erkenntnisse.
- Setze Quellenverweise direkt hinter die Aussagen.
- Verwende ausschließlich Verweise wie [Quelle 1].
- Erfinde keine Informationen.
- Falls die Quellen nicht ausreichen, schreibe:
  "Die gefundenen Quellen reichen für eine verlässliche Antwort nicht aus."
""".strip()

    return [
        {
            "role": "user",
            "content": gesamter_prompt,
        }
    ]


# ============================================================
# 15. ROBUSTE MODELLEINGABE
# ============================================================

def chat_eingabe_erstellen(
    tokenizer,
    nachrichten: list[dict],
    eingabe_geraet,
) -> dict[str, torch.Tensor]:
    """
    Unterstützt BatchEncoding, Dictionary und Tensor.
    """

    try:

        vorbereitete_eingabe = tokenizer.apply_chat_template(
            nachrichten,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )

    except TypeError:

        vorbereitete_eingabe = tokenizer.apply_chat_template(
            nachrichten,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )

    if isinstance(
        vorbereitete_eingabe,
        torch.Tensor,
    ):

        model_inputs = {
            "input_ids": vorbereitete_eingabe,
            "attention_mask": torch.ones_like(
                vorbereitete_eingabe,
                dtype=torch.long,
            ),
        }

    elif (
        isinstance(
            vorbereitete_eingabe,
            dict,
        )
        or hasattr(
            vorbereitete_eingabe,
            "items",
        )
    ):

        model_inputs = {
            schluessel: wert
            for schluessel, wert
            in vorbereitete_eingabe.items()
            if isinstance(
                wert,
                torch.Tensor,
            )
        }

        if "input_ids" not in model_inputs:

            raise ValueError(
                "Das Chat-Template hat keine input_ids erzeugt."
            )

        if "attention_mask" not in model_inputs:

            model_inputs["attention_mask"] = torch.ones_like(
                model_inputs["input_ids"],
                dtype=torch.long,
            )

    else:

        raise TypeError(
            "Das Chat-Template lieferte einen "
            f"unbekannten Datentyp: "
            f"{type(vorbereitete_eingabe)}"
        )

    return {
        schluessel: wert.to(
            eingabe_geraet
        )
        for schluessel, wert
        in model_inputs.items()
    }


# ============================================================
# 16. MISTRAL-ANTWORT ERZEUGEN
# ============================================================

def mistral_antwort_erzeugen(
    frage: str,
    treffer_df: pd.DataFrame,
    tokenizer,
    model,
    eingabe_geraet,
    max_characters_per_source: int,
    max_new_tokens: int,
    temperature: float,
) -> str:
    """
    Erzeugt eine Antwort ausschließlich aus den Quellen.
    """

    if treffer_df.empty:

        return (
            "Es wurden keine ausreichend ähnlichen "
            "Textstellen gefunden."
        )

    quellenkontext = quellenkontext_erstellen(
        treffer_df=treffer_df,
        max_characters_per_source=(
            max_characters_per_source
        ),
    )

    nachrichten = mistral_nachrichten_erstellen(
        frage=frage,
        quellenkontext=quellenkontext,
    )

    model_inputs = chat_eingabe_erstellen(
        tokenizer=tokenizer,
        nachrichten=nachrichten,
        eingabe_geraet=eingabe_geraet,
    )

    eingabe_laenge = (
        model_inputs[
            "input_ids"
        ].shape[-1]
    )

    generation_parameter = {
        **model_inputs,
        "max_new_tokens": int(
            max_new_tokens
        ),
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "repetition_penalty": REPETITION_PENALTY,
        "use_cache": True,
    }

    if float(temperature) > 0:

        generation_parameter.update(
            {
                "do_sample": True,
                "temperature": float(
                    temperature
                ),
                "top_p": 0.9,
            }
        )

    else:

        generation_parameter.update(
            {
                "do_sample": False,
            }
        )

    with torch.inference_mode():

        ausgabe_ids = model.generate(
            **generation_parameter
        )

    neue_ids = ausgabe_ids[
        0,
        eingabe_laenge:,
    ]

    antwort = tokenizer.decode(
        neue_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    ).strip()

    if not antwort:

        raise ValueError(
            "Mistral hat keine Antwort erzeugt."
        )

    return antwort


# ============================================================
# 17. EXPORT
# ============================================================

def txt_export_erstellen(
    frage: str,
    antwort: str,
    quellen_df: pd.DataFrame,
) -> str:
    """
    Erstellt einen TXT-Export.
    """

    teile = [
        "OPENLENS – MISTRAL-RAG",
        "=" * 70,
        "",
        "FRAGE",
        frage,
        "",
        "ANTWORT",
        antwort,
        "",
        "QUELLEN",
    ]

    for nummer, (_, row) in enumerate(
        quellen_df.iterrows(),
        start=1,
    ):

        titel = sicherer_textwert(
            row.get(
                "title",
                "",
            )
        ) or "Ohne Titel"

        dokument_id = sicherer_textwert(
            row.get(
                "document_id",
                "",
            )
        )

        seitenangabe = seitenangabe_erstellen(
            row.get(
                "page_start",
                "",
            ),
            row.get(
                "page_end",
                "",
            ),
        )

        site_url = sicherer_textwert(
            row.get(
                "site_url",
                "",
            )
        )

        text = sicherer_textwert(
            row.get(
                "text",
                "",
            )
        )

        teile.extend(
            [
                "",
                f"Quelle {nummer}",
                f"Titel: {titel}",
                f"Dokument-ID: {dokument_id}",
                f"Fundstelle: {seitenangabe}",
                f"URL: {site_url}",
                "",
                text,
                "-" * 70,
            ]
        )

    teile.extend(
        [
            "",
            f"Modell: {MISTRAL_REPOSITORY}",
            f"Erstellt: {datetime.now().isoformat(timespec='seconds')}",
        ]
    )

    return "\n".join(teile)


def csv_export_erstellen(
    quellen_df: pd.DataFrame,
) -> bytes:
    """
    Erstellt eine UTF-8-CSV.
    """

    puffer = io.StringIO()

    quellen_df.to_csv(
        puffer,
        index=False,
    )

    return puffer.getvalue().encode(
        "utf-8-sig"
    )


def ergebnis_lokal_speichern(
    frage: str,
    antwort: str,
    quellen_df: pd.DataFrame,
) -> None:
    """
    Speichert Antwort und Quellen lokal.
    """

    zeitstempel = datetime.now().strftime(
        "%Y-%m-%d_%H-%M-%S"
    )

    txt_datei = (
        RAG_ORDNER
        / f"mistral_antwort_{zeitstempel}.txt"
    )

    csv_datei = (
        RAG_ORDNER
        / f"mistral_quellen_{zeitstempel}.csv"
    )

    json_datei = (
        RAG_ORDNER
        / f"mistral_ergebnis_{zeitstempel}.json"
    )

    txt_datei.write_text(
        txt_export_erstellen(
            frage=frage,
            antwort=antwort,
            quellen_df=quellen_df,
        ),
        encoding="utf-8",
    )

    quellen_df.to_csv(
        csv_datei,
        index=False,
        encoding="utf-8-sig",
    )

    json_inhalt = {
        "frage": frage,
        "antwort": antwort,
        "modell": MISTRAL_REPOSITORY,
        "collection": COLLECTION_NAME,
        "anzahl_quellen": int(
            len(quellen_df)
        ),
        "erstellt_am": datetime.now().isoformat(),
        "quellen": quellen_df.to_dict(
            orient="records"
        ),
    }

    with json_datei.open(
        "w",
        encoding="utf-8",
    ) as datei:

        json.dump(
            json_inhalt,
            datei,
            ensure_ascii=False,
            indent=2,
            default=str,
        )


# ============================================================
# 18. SESSION-STATUS
# ============================================================

if "frage" not in st.session_state:
    st.session_state.frage = ""

if "antwort" not in st.session_state:
    st.session_state.antwort = ""

if "quellen_df" not in st.session_state:
    st.session_state.quellen_df = pd.DataFrame()

if "suche_ausgefuehrt" not in st.session_state:
    st.session_state.suche_ausgefuehrt = False

if "verlauf" not in st.session_state:
    st.session_state.verlauf = []


# ============================================================
# 19. HEADER
# ============================================================

st.markdown(
    """
    <div class="openlens-header">
        <p class="openlens-title">OpenLens</p>
        <p class="openlens-subtitle">
            Lokale semantische Suche und KI-Analyse
            für Behörden- und Informationsfreiheitsdokumente
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 20. SYSTEM PRÜFEN UND LADEN
# ============================================================

if not CHROMA_ORDNER.exists():

    st.error(
        "Der ChromaDB-Ordner wurde nicht gefunden."
    )

    st.code(
        str(CHROMA_ORDNER)
    )

    st.stop()


if not LOKALES_EMBEDDING_MODELL.is_dir():

    st.error(
        "Das lokale Embedding-Modell wurde nicht gefunden."
    )

    st.code(
        str(LOKALES_EMBEDDING_MODELL)
    )

    st.stop()


manifest = chroma_manifest_laden()

if manifest.get(
    "collection_name"
):

    COLLECTION_NAME = str(
        manifest["collection_name"]
    )


EMBEDDING_GERAET = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


try:

    with st.spinner(
        "ChromaDB und Embedding-Modell werden geladen ..."
    ):

        chroma_client, collection = chromadb_laden(
            chroma_pfad=str(
                CHROMA_ORDNER
            ),
            collection_name=COLLECTION_NAME,
        )

        embedding_model = embedding_modell_laden(
            modellpfad=str(
                LOKALES_EMBEDDING_MODELL
            ),
            geraet=EMBEDDING_GERAET,
        )

    collection_count = collection.count()

except Exception as fehler:

    st.error(
        "Die Suchdatenbank konnte nicht geladen werden."
    )

    st.exception(fehler)

    st.stop()


# ============================================================
# 21. SIDEBAR
# ============================================================

with st.sidebar:

    st.header("System")

    st.markdown(
        '<div class="status-box">'
        '● ChromaDB verfügbar'
        '</div>',
        unsafe_allow_html=True,
    )

    st.write("")

    st.metric(
        "Gespeicherte Chunks",
        f"{collection_count:,}",
    )

    st.metric(
        "Collection",
        COLLECTION_NAME,
    )

    if torch.cuda.is_available():

        gpu_name = torch.cuda.get_device_name(
            0
        )

        gpu_speicher = (
            torch.cuda.get_device_properties(
                0
            ).total_memory
            / (1024 ** 3)
        )

        st.success(
            f"GPU erkannt: {gpu_name}"
        )

        st.caption(
            f"GPU-Speicher: {gpu_speicher:.1f} GB"
        )

    else:

        st.warning(
            "CPU-Modus aktiv"
        )

        st.caption(
            "Mistral 7B kann auf der CPU "
            "deutlich langsamer sein."
        )

    st.divider()

    st.header("Sucheinstellungen")

    final_top_k = st.slider(
        "Quellen für die Antwort",
        min_value=2,
        max_value=8,
        value=FINAL_TOP_K_STANDARD,
        step=1,
    )

    initial_top_k = st.slider(
        "Erste Suchtreffer",
        min_value=final_top_k,
        max_value=30,
        value=max(
            INITIAL_TOP_K_STANDARD,
            final_top_k,
        ),
        step=1,
    )

    min_similarity = st.slider(
        "Mindestähnlichkeit",
        min_value=0.0,
        max_value=0.8,
        value=MIN_SIMILARITY_STANDARD,
        step=0.05,
    )

    max_characters_per_source = st.slider(
        "Zeichen pro Quelle",
        min_value=750,
        max_value=3500,
        value=MAX_CHARACTERS_PER_SOURCE_STANDARD,
        step=250,
    )

    st.divider()

    st.header("Mistral-Antwort")

    max_new_tokens = st.slider(
        "Maximale Antwortlänge",
        min_value=100,
        max_value=800,
        value=MAX_NEW_TOKENS_STANDARD,
        step=50,
    )

    temperature = st.slider(
        "Kreativität",
        min_value=0.0,
        max_value=1.0,
        value=TEMPERATURE_STANDARD,
        step=0.1,
        help=(
            "Für sachliche Dokumentenanalyse "
            "am besten bei 0 lassen."
        ),
    )

    st.divider()

    if st.button(
        "Modelle neu laden",
        use_container_width=True,
    ):

        st.cache_resource.clear()

        st.rerun()

    if st.button(
        "Verlauf löschen",
        use_container_width=True,
    ):

        st.session_state.verlauf = []

        st.session_state.antwort = ""

        st.session_state.quellen_df = pd.DataFrame()

        st.session_state.suche_ausgefuehrt = False

        st.rerun()


# ============================================================
# 22. STATUSÜBERSICHT
# ============================================================

spalte_1, spalte_2, spalte_3 = st.columns(3)

with spalte_1:

    st.metric(
        "Suchbestand",
        f"{collection_count:,} Chunks",
    )

with spalte_2:

    st.metric(
        "Embedding",
        "Multilingual MiniLM",
    )

with spalte_3:

    st.metric(
        "Antwortmodell",
        "Mistral 7B v0.3",
    )


if lokales_modell_vollstaendig(
    LOKALES_MISTRAL_MODELL
):

    st.caption(
        "Mistral ist lokal verfügbar."
    )

else:

    st.warning(
        "Mistral ist noch nicht lokal vorhanden. "
        "Beim ersten Suchvorgang wird das Modell heruntergeladen."
    )


# ============================================================
# 23. FRAGEFORMULAR
# ============================================================

st.subheader(
    "Dokumente befragen"
)

with st.form(
    "openlens_suchformular",
    clear_on_submit=False,
):

    frage = st.text_area(
        "Frage",
        value=st.session_state.frage,
        height=125,
        placeholder=(
            "Beispiel: Welche Gründe nennen Behörden "
            "für die Ablehnung von Informationsanfragen?"
        ),
    )

    info_spalte, button_spalte = st.columns(
        [3, 1]
    )

    with info_spalte:

        st.caption(
            "Die Dokumente und das Sprachmodell "
            "werden lokal verarbeitet."
        )

    with button_spalte:

        suche_starten = st.form_submit_button(
            "Suche starten",
            type="primary",
            use_container_width=True,
        )


# ============================================================
# 24. SUCHE AUSFÜHREN
# ============================================================

if suche_starten:

    frage = sicherer_textwert(
        frage
    )

    if not frage:

        st.warning(
            "Bitte gib zuerst eine Frage ein."
        )

    else:

        st.session_state.frage = frage

        try:

            with st.status(
                "OpenLens verarbeitet die Frage ...",
                expanded=True,
            ) as status:

                st.write(
                    "1. Die Frage wird in einen "
                    "Embedding-Vektor umgewandelt."
                )

                st.write(
                    "2. ChromaDB sucht passende Textstellen."
                )

                quellen_df = semantische_suche(
                    frage=frage,
                    collection=collection,
                    embedding_model=embedding_model,
                    initial_top_k=initial_top_k,
                    final_top_k=final_top_k,
                    min_similarity=min_similarity,
                )

                if quellen_df.empty:

                    antwort = (
                        "Es wurden keine ausreichend ähnlichen "
                        "Textstellen gefunden."
                    )

                    status.update(
                        label="Keine passenden Quellen gefunden",
                        state="complete",
                    )

                else:

                    st.write(
                        f"3. {len(quellen_df)} relevante "
                        "Quellen wurden gefunden."
                    )

                    st.write(
                        "4. Mistral wird vorbereitet."
                    )

                    mistral_bereitstellen()

                    (
                        tokenizer,
                        mistral_model,
                        eingabe_geraet,
                        modell_datentyp,
                    ) = mistral_laden(
                        modellpfad=str(
                            LOKALES_MISTRAL_MODELL
                        )
                    )

                    st.write(
                        f"5. Mistral läuft mit "
                        f"{modell_datentyp} auf "
                        f"{eingabe_geraet}."
                    )

                    st.write(
                        "6. Die Antwort wird aus den "
                        "gefundenen Quellen erzeugt."
                    )

                    antwort = mistral_antwort_erzeugen(
                        frage=frage,
                        treffer_df=quellen_df,
                        tokenizer=tokenizer,
                        model=mistral_model,
                        eingabe_geraet=eingabe_geraet,
                        max_characters_per_source=(
                            max_characters_per_source
                        ),
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                    )

                    status.update(
                        label="Antwort fertig",
                        state="complete",
                    )

            st.session_state.antwort = antwort

            st.session_state.quellen_df = quellen_df

            st.session_state.suche_ausgefuehrt = True

            st.session_state.verlauf.insert(
                0,
                {
                    "frage": frage,
                    "antwort": antwort,
                    "zeitpunkt": (
                        datetime.now().strftime(
                            "%d.%m.%Y %H:%M"
                        )
                    ),
                    "anzahl_quellen": int(
                        len(quellen_df)
                    ),
                },
            )

            st.session_state.verlauf = (
                st.session_state.verlauf[:10]
            )

            ergebnis_lokal_speichern(
                frage=frage,
                antwort=antwort,
                quellen_df=quellen_df,
            )

        except torch.cuda.OutOfMemoryError:

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            st.error(
                "Der GPU-Speicher reicht für diese Anfrage nicht aus."
            )

            st.info(
                "Reduziere die Quellenanzahl, Zeichen pro Quelle "
                "oder die maximale Antwortlänge."
            )

        except MemoryError:

            st.error(
                "Der Arbeitsspeicher reicht zum Laden "
                "oder Ausführen von Mistral nicht aus."
            )

        except Exception as fehler:

            st.error(
                "Die Anfrage konnte nicht vollständig "
                "verarbeitet werden."
            )

            st.exception(fehler)


# ============================================================
# 25. ANTWORT ANZEIGEN
# ============================================================

if st.session_state.suche_ausgefuehrt:

    st.divider()

    st.subheader("Antwort")

    st.markdown(
        '<div class="answer-box">',
        unsafe_allow_html=True,
    )

    st.markdown(
        st.session_state.antwort
    )

    st.markdown(
        "</div>",
        unsafe_allow_html=True,
    )

    quellen_df = (
        st.session_state.quellen_df
    )

    export_1, export_2, export_3 = st.columns(
        [1, 1, 2]
    )

    with export_1:

        st.download_button(
            label="Antwort als TXT",
            data=txt_export_erstellen(
                frage=st.session_state.frage,
                antwort=st.session_state.antwort,
                quellen_df=quellen_df,
            ),
            file_name="openlens_mistral_antwort.txt",
            mime="text/plain",
            use_container_width=True,
        )

    with export_2:

        st.download_button(
            label="Quellen als CSV",
            data=csv_export_erstellen(
                quellen_df
            ),
            file_name="openlens_mistral_quellen.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with export_3:

        st.caption(
            "Das Ergebnis wurde zusätzlich im Ordner "
            "`RAG_Ergebnisse` gespeichert."
        )


# ============================================================
# 26. QUELLEN ANZEIGEN
# ============================================================

if (
    st.session_state.suche_ausgefuehrt
    and not st.session_state.quellen_df.empty
):

    st.divider()

    st.subheader(
        f"Verwendete Quellen "
        f"({len(st.session_state.quellen_df)})"
    )

    for quellennummer, (_, row) in enumerate(
        st.session_state.quellen_df.iterrows(),
        start=1,
    ):

        titel = sicherer_textwert(
            row.get(
                "title",
                "",
            )
        ) or "Ohne Titel"

        dokument_id = sicherer_textwert(
            row.get(
                "document_id",
                "",
            )
        )

        chunk_id = sicherer_textwert(
            row.get(
                "chunk_id",
                "",
            )
        )

        similarity = float(
            row.get(
                "similarity_score",
                0.0,
            )
        )

        seitenangabe = seitenangabe_erstellen(
            row.get(
                "page_start",
                "",
            ),
            row.get(
                "page_end",
                "",
            ),
        )

        site_url = sicherer_textwert(
            row.get(
                "site_url",
                "",
            )
        )

        file_url = sicherer_textwert(
            row.get(
                "file_url",
                "",
            )
        )

        text = sicherer_textwert(
            row.get(
                "text",
                "",
            )
        )

        with st.expander(
            (
                f"Quelle {quellennummer}: "
                f"{titel} — {seitenangabe}"
            ),
            expanded=(
                quellennummer == 1
            ),
        ):

            metadaten_1, metadaten_2, metadaten_3 = (
                st.columns(3)
            )

            with metadaten_1:

                st.metric(
                    "Ähnlichkeit",
                    f"{similarity:.1%}",
                )

            with metadaten_2:

                st.metric(
                    "Dokument-ID",
                    dokument_id or "–",
                )

            with metadaten_3:

                st.metric(
                    "Fundstelle",
                    seitenangabe,
                )

            st.caption(
                f"Chunk-ID: {chunk_id}"
            )

            link_1, link_2 = st.columns(2)

            with link_1:

                if site_url:

                    st.link_button(
                        "FragDenStaat-Seite öffnen",
                        site_url,
                        use_container_width=True,
                    )

            with link_2:

                if file_url:

                    st.link_button(
                        "Originaldatei öffnen",
                        file_url,
                        use_container_width=True,
                    )

            st.markdown(
                "#### Gefundener Originaltext"
            )

            st.text_area(
                label=(
                    f"Originaltext Quelle "
                    f"{quellennummer}"
                ),
                value=text,
                height=300,
                disabled=True,
                label_visibility="collapsed",
                key=(
                    f"quellentext_"
                    f"{quellennummer}_"
                    f"{chunk_id}"
                ),
            )


# ============================================================
# 27. SUCHVERLAUF
# ============================================================

if st.session_state.verlauf:

    st.divider()

    st.subheader("Letzte Fragen")

    for eintrag in st.session_state.verlauf:

        with st.expander(
            (
                f"{eintrag['zeitpunkt']} – "
                f"{eintrag['frage'][:90]}"
            )
        ):

            st.markdown("**Frage**")

            st.write(
                eintrag["frage"]
            )

            st.markdown("**Antwort**")

            st.markdown(
                eintrag["antwort"]
            )

            st.caption(
                f"Verwendete Quellen: "
                f"{eintrag['anzahl_quellen']}"
            )


# ============================================================
# 28. STARTANSICHT
# ============================================================

if not st.session_state.suche_ausgefuehrt:

    st.divider()

    st.subheader("Beispielfragen")

    beispiele = [
        (
            "Welche Gründe nennen Behörden für die "
            "Ablehnung von Informationsanfragen?"
        ),
        (
            "Welche Dokumente beschäftigen sich mit "
            "staatlichen Entscheidungsprozessen?"
        ),
        (
            "Welche Hinweise gibt es auf verzögerte "
            "oder unvollständige Auskünfte?"
        ),
        (
            "Welche Behörden und Institutionen werden "
            "in den Dokumenten genannt?"
        ),
    ]

    beispiel_spalten = st.columns(2)

    for index, beispiel in enumerate(
        beispiele
    ):

        with beispiel_spalten[
            index % 2
        ]:

            st.info(beispiel)


# ============================================================
# ============================================================
# 29. FOOTER
# ============================================================

st.divider()

st.caption(
    "OpenLens · Lokale Dokumentenanalyse · "
    "ChromaDB + Sentence Transformers + Mistral 7B"
)