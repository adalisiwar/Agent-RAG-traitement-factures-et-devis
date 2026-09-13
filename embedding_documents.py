import os
import psycopg2
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not DATABASE_URL:
    raise ValueError("DATABASE_URL is not defined in .env")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is not defined in .env")

client = genai.Client(api_key=GEMINI_API_KEY)

EMBEDDING_MODEL = "gemini-embedding-001"

conn = psycopg2.connect(DATABASE_URL)
cursor = conn.cursor()

print("Connected to Neon")

cursor.execute("""
    CREATE EXTENSION IF NOT EXISTS vector;
""")

conn.commit()

print("pgvector extension enabled")

cursor.execute("""
    CREATE TABLE IF NOT EXISTS public.document_embedding (
        document_id UUID PRIMARY KEY,
        chunk_text TEXT NOT NULL,
        embedding VECTOR(1536),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        CONSTRAINT document_embedding_document_fk
            FOREIGN KEY (document_id)
            REFERENCES public.document(id)
            ON DELETE CASCADE
    );
""")

conn.commit()

print("Table document_embedding ready")

cursor.execute("""
    ALTER TABLE public.document
    ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ NOT NULL DEFAULT now();
""")

conn.commit()

print("Column document.updated_at ready")


document_query = """
SELECT
    d.id,
    d.type_document,
    d.numero_document,
    d.date_emission,
    d.date_echeance_ou_validite,
    d.montant_ht,
    d.montant_tva,
    d.montant_ttc,
    d.devise,
    d.montant_paye,
    d.reste_a_payer,

    f.nom AS fournisseur_nom,
    f.adresse AS fournisseur_adresse,
    f.identifiant_fiscal AS fournisseur_identifiant_fiscal,

    c.nom AS client_nom,
    c.adresse AS client_adresse,

    s.libelle AS statut_libelle

FROM public.document d

LEFT JOIN public.document_embedding de
    ON d.id = de.document_id

LEFT JOIN public.fournisseur f
    ON d.fournisseur_id = f.id

LEFT JOIN public.client c
    ON d.client_id = c.id

LEFT JOIN public.statut s
    ON d.statut_id = s.id

WHERE de.document_id IS NULL
   OR d.updated_at > de.created_at;
"""

cursor.execute(document_query)

documents = cursor.fetchall()

print(f"{len(documents)} document(s) to embed")


def get_document_lines(document_id):
    query = """
    SELECT
        description,
        quantite,
        prix_unitaire,
        montant_ligne,
        taux_tva
    FROM public.ligne_document
    WHERE document_id = %s
    ORDER BY id
    """

    cursor.execute(query, (document_id,))

    return cursor.fetchall()


def statut_paiement_label(montant_paye, reste_a_payer):
    montant_paye = montant_paye or 0

    if reste_a_payer is None:
        return "Non renseigné"
    if reste_a_payer <= 0 and montant_paye > 0:
        return "Payée"
    if montant_paye > 0 and reste_a_payer > 0:
        return "Partiellement payée"
    return "Impayée"


def build_chunk(document, lines):
    (
        document_id,
        type_document,
        numero_document,
        date_emission,
        date_echeance,
        montant_ht,
        montant_tva,
        montant_ttc,
        devise,
        montant_paye,
        reste_a_payer,
        fournisseur_nom,
        fournisseur_adresse,
        fournisseur_identifiant_fiscal,
        client_nom,
        client_adresse,
        statut_libelle
    ) = document

    statut_paiement = statut_paiement_label(montant_paye, reste_a_payer)

    chunk = f"""
Document : {type_document}
Numéro du document : {numero_document}

Fournisseur :
Nom : {fournisseur_nom or "Non renseigné"}
Adresse : {fournisseur_adresse or "Non renseignée"}
Identifiant fiscal : {fournisseur_identifiant_fiscal or "Non renseigné"}

Client :
Nom : {client_nom or "Non renseigné"}
Adresse : {client_adresse or "Non renseignée"}

Date d'émission : {date_emission or "Non renseignée"}
Date d'échéance / validité : {date_echeance or "Non renseignée"}

Montant HT : {montant_ht or 0} {devise or ""}
Montant TVA : {montant_tva or 0} {devise or ""}
Montant TTC : {montant_ttc or 0} {devise or ""}
Montant payé : {montant_paye or 0} {devise or ""}
Reste à payer : {reste_a_payer if reste_a_payer is not None else "Non renseigné"} {devise or ""}
Statut de paiement : {statut_paiement}

Statut : {statut_libelle or "Non renseigné"}
""".strip()

    if lines:
        chunk += "\n\nLignes du document :"

        for i, line in enumerate(lines, start=1):
            (
                description,
                quantite,
                prix_unitaire,
                montant_ligne,
                taux_tva
            ) = line

            chunk += f"""

Ligne {i} :
Description : {description or "Non renseignée"}
Quantité : {quantite or 0}
Prix unitaire : {prix_unitaire or 0} {devise or ""}
Montant ligne : {montant_ligne or 0} {devise or ""}
Taux TVA : {taux_tva or 0} %
"""

    return chunk.strip()


for document in documents:
    document_id = document[0]

    print("\n")
    print("=" * 60)
    print(f"DOCUMENT : {document_id}")
    print("=" * 60)

    lines = get_document_lines(document_id)

    print(f"Number of lines: {len(lines)}")

    chunk_text = build_chunk(document, lines)

    print("\nCHUNK TEXT :")
    print("-" * 60)
    print(chunk_text)
    print("-" * 60)

    print("\nGenerating embedding...")

    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=chunk_text,
        config=types.EmbedContentConfig(output_dimensionality=1536)
    )

    embedding = response.embeddings[0].values

    dimension = len(embedding)

    print(f"Dimension: {dimension}")

    if dimension != 1536:
        raise ValueError(
            f"Incorrect dimension: {dimension}. Expected dimension: 1536."
        )

    embedding_string = "[" + ",".join(
        str(value) for value in embedding
    ) + "]"

    insert_query = """
        INSERT INTO public.document_embedding (
            document_id,
            chunk_text,
            embedding
        )
        VALUES (
            %s,
            %s,
            %s::vector
        )
        ON CONFLICT (document_id)
        DO UPDATE SET
            chunk_text = EXCLUDED.chunk_text,
            embedding = EXCLUDED.embedding,
            created_at = CURRENT_TIMESTAMP
    """

    cursor.execute(
        insert_query,
        (
            document_id,
            chunk_text,
            embedding_string
        )
    )

    conn.commit()

    print("Embedding saved to Neon")


cursor.execute("""
    SELECT
        document_id,
        vector_dims(embedding) AS dimensions,
        created_at
    FROM public.document_embedding
    ORDER BY created_at DESC;
""")

results = cursor.fetchall()

print("\n")
print("=" * 60)
print("VERIFICATION")
print("=" * 60)

for row in results:
    print(f"""
Document ID : {row[0]}
Dimensions  : {row[1]}
Created at  : {row[2]}
""")

cursor.close()
conn.close()

print("Embedding pipeline completed successfully.")