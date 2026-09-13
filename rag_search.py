import os
import re
from dataclasses import dataclass, field
from typing import Optional

from google import genai
from google.genai import types
from dotenv import load_dotenv

from db_utils import execute_with_retry
from llm_utils import call_with_retry

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")

if not DATABASE_URL or not GEMINI_API_KEY:
    raise ValueError("DATABASE_URL or GEMINI_API_KEY missing")

client = genai.Client(api_key=GEMINI_API_KEY)


@dataclass
class SearchFilters:
    year: Optional[int] = None
    statut: Optional[str] = None
    fournisseur: Optional[str] = None
    paiement: Optional[str] = None
    is_aggregation: bool = False
    agg_type: Optional[str] = None


@dataclass
class DocumentMatch:
    document_id: str
    chunk_text: str
    date_emission: object
    montant_ttc: object
    montant_paye: object
    reste_a_payer: object
    statut: object
    fournisseur: object
    score_vec: float
    score_txt: float
    rrf_score: float


@dataclass
class SearchResult:
    question: str
    filters: SearchFilters
    mode: str
    aggregation: Optional[dict] = None
    matches: list = field(default_factory=list)


BASE_FROM = """
FROM public.document_embedding de
JOIN public.document d          ON d.id = de.document_id
LEFT JOIN public.statut s       ON d.statut_id = s.id
LEFT JOIN public.fournisseur f  ON d.fournisseur_id = f.id
"""


def detect_filters(question):
    question_lower = question.lower()
    filters = SearchFilters()

    year_match = re.search(r"\b(19|20)\d{2}\b", question)
    if year_match:
        filters.year = int(year_match.group(0))

    if "anomalie montant" in question_lower:
        filters.statut = "Anomalie montant"
    elif "anomalie date" in question_lower:
        filters.statut = "Anomalie date"
    elif "anomalie" in question_lower:
        filters.statut = "Anomalie"

    fourn_match = re.search(r"fournisseur\s+([A-Za-z0-9\-\s]+)", question, re.IGNORECASE)
    if fourn_match:
        filters.fournisseur = fourn_match.group(1).strip()

    if any(w in question_lower for w in [
        "impayée", "impayees", "impayée(s)", "non payée", "non payee",
        "non réglée", "non reglee", "en attente de paiement",
    ]):
        filters.paiement = "impayee"
    elif any(w in question_lower for w in [
        "partiellement payée", "partiellement payee", "paiement partiel",
    ]):
        filters.paiement = "partielle"
    elif any(w in question_lower for w in [
        "payée", "payees", "payee", "soldée", "soldee", "réglée", "reglee",
    ]):
        filters.paiement = "payee"

    if any(w in question_lower for w in ["reste à payer", "reste a payer", "montant restant", "encours"]):
        filters.is_aggregation = True
        filters.agg_type = "reste"
    elif any(w in question_lower for w in ["total", "somme", "montant total"]):
        filters.is_aggregation = True
        filters.agg_type = "sum"
    elif any(w in question_lower for w in ["nombre", "combien", "count"]):
        filters.is_aggregation = True
        filters.agg_type = "count"

    return filters


def build_where(filters):
    clauses = []
    params = {}

    if filters.year:
        clauses.append("EXTRACT(YEAR FROM d.date_emission) = %(year)s")
        params["year"] = filters.year

    if filters.statut:
        clauses.append("s.libelle ILIKE %(statut)s")
        params["statut"] = f"%{filters.statut}%"

    if filters.fournisseur:
        clauses.append("f.nom ILIKE %(fournisseur)s")
        params["fournisseur"] = f"%{filters.fournisseur}%"

    if filters.paiement == "impayee":
        clauses.append("COALESCE(d.reste_a_payer, d.montant_ttc, 0) > 0")
    elif filters.paiement == "payee":
        clauses.append("COALESCE(d.montant_paye, 0) > 0 AND COALESCE(d.reste_a_payer, 0) <= 0")
    elif filters.paiement == "partielle":
        clauses.append("COALESCE(d.montant_paye, 0) > 0 AND COALESCE(d.reste_a_payer, 0) > 0")

    where_sql = "WHERE " + " AND ".join(clauses) if clauses else ""
    return where_sql, params


def embed_question(question):
    def call():
        return client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=question,
            config=types.EmbedContentConfig(output_dimensionality=1536)
        )

    response = call_with_retry(call)
    values = response.embeddings[0].values
    return "[" + ",".join(str(v) for v in values) + "]"


def run_aggregation(conn_holder, filters):
    where_sql, params = build_where(filters)

    if filters.agg_type == "sum":
        sql = f"""
        SELECT
            COUNT(*)                         AS nb_factures,
            COALESCE(SUM(d.montant_ttc), 0)  AS total_ttc,
            COALESCE(SUM(d.montant_ht), 0)   AS total_ht,
            COALESCE(SUM(d.montant_tva), 0)  AS total_tva
        {BASE_FROM}
        {where_sql}
        """
    elif filters.agg_type == "reste":
        sql = f"""
        SELECT
            COUNT(*) FILTER (WHERE COALESCE(d.reste_a_payer, d.montant_ttc, 0) > 0) AS nb_impayees,
            COALESCE(SUM(d.reste_a_payer), 0)                                       AS total_reste_a_payer,
            COALESCE(SUM(d.montant_paye), 0)                                        AS total_paye
        {BASE_FROM}
        {where_sql}
        """
    else:
        sql = f"""
        SELECT COUNT(*) AS nb_factures
        {BASE_FROM}
        {where_sql}
        """

    rows = execute_with_retry(DATABASE_URL, conn_holder, sql, params)
    row = rows[0]

    if filters.agg_type == "sum":
        return {
            "nb_factures": row[0],
            "total_ttc": float(row[1]),
            "total_ht": float(row[2]),
            "total_tva": float(row[3]),
        }
    if filters.agg_type == "reste":
        return {
            "nb_impayees": row[0],
            "total_reste_a_payer": float(row[1]),
            "total_paye": float(row[2]),
        }
    return {"nb_factures": row[0]}


def run_hybrid_search(conn_holder, question, filters):
    embedding_string = embed_question(question)
    where_sql, where_params = build_where(filters)

    hybrid_sql = f"""
    WITH vector_candidates AS (
        SELECT
            de.document_id,
            de.chunk_text,
            d.date_emission,
            d.montant_ttc,
            d.montant_paye,
            d.reste_a_payer,
            s.libelle AS statut,
            f.nom AS fournisseur,
            ROW_NUMBER() OVER (ORDER BY de.embedding <=> %(embedding)s::vector) AS rank_vec,
            1 - (de.embedding <=> %(embedding)s::vector) AS score_vec
        {BASE_FROM}
        {where_sql}
        ORDER BY de.embedding <=> %(embedding)s::vector
        LIMIT 30
    ),
    text_candidates AS (
        SELECT
            de.document_id,
            ROW_NUMBER() OVER (
                ORDER BY ts_rank(
                    to_tsvector('french', coalesce(de.chunk_text, '')),
                    websearch_to_tsquery('french', %(question)s)
                ) DESC
            ) AS rank_txt,
            ts_rank(
                to_tsvector('french', coalesce(de.chunk_text, '')),
                websearch_to_tsquery('french', %(question)s)
            ) AS score_txt
        {BASE_FROM}
        {where_sql}
        AND to_tsvector('french', coalesce(de.chunk_text, ''))
            @@ websearch_to_tsquery('french', %(question)s)
        ORDER BY score_txt DESC
        LIMIT 30
    )
    SELECT
        v.document_id,
        v.chunk_text,
        v.date_emission,
        v.montant_ttc,
        v.montant_paye,
        v.reste_a_payer,
        v.statut,
        v.fournisseur,
        v.score_vec,
        COALESCE(t.score_txt, 0) AS score_txt,
        (1.0 / (60 + v.rank_vec)) +
        (CASE WHEN t.rank_txt IS NOT NULL
              THEN 1.0 / (60 + t.rank_txt)
              ELSE 0 END) AS rrf_score
    FROM vector_candidates v
    LEFT JOIN text_candidates t ON v.document_id = t.document_id
    ORDER BY rrf_score DESC
    LIMIT 8;
    """

    exec_params = dict(where_params)
    exec_params["embedding"] = embedding_string
    exec_params["question"] = question

    rows = execute_with_retry(DATABASE_URL, conn_holder, hybrid_sql, exec_params)

    matches = []
    for row in rows:
        (doc_id, chunk, date_em, mttc, mpaye, reste, statut, fourn,
         score_vec, score_txt, rrf) = row
        matches.append(DocumentMatch(
            document_id=str(doc_id),
            chunk_text=chunk,
            date_emission=date_em,
            montant_ttc=mttc,
            montant_paye=mpaye,
            reste_a_payer=reste,
            statut=statut,
            fournisseur=fourn,
            score_vec=float(score_vec),
            score_txt=float(score_txt),
            rrf_score=float(rrf),
        ))

    return matches


def search(question, conn_holder):
    filters = detect_filters(question)

    if filters.is_aggregation:
        aggregation = run_aggregation(conn_holder, filters)
        return SearchResult(question=question, filters=filters, mode="aggregation", aggregation=aggregation)

    matches = run_hybrid_search(conn_holder, question, filters)
    return SearchResult(question=question, filters=filters, mode="hybrid_search", matches=matches)