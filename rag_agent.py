import os

from google import genai
from dotenv import load_dotenv

from db_utils import connect
from llm_utils import call_with_retry
from rag_search import DATABASE_URL, GEMINI_API_KEY, search
from rag_reporting import generate_report, is_report_request

load_dotenv()

GENERATION_MODEL = os.getenv("GENERATION_MODEL", "gemini-3.6-flash")

client = genai.Client(api_key=GEMINI_API_KEY)


def format_context(matches):
    parts = []
    for i, match in enumerate(matches, start=1):
        parts.append(
            f"[Document {i} | ID: {match.document_id} | "
            f"Statut: {match.statut} | Montant TTC: {match.montant_ttc} | "
            f"Reste à payer: {match.reste_a_payer}]\n{match.chunk_text}"
        )
    return "\n\n".join(parts)


def build_qa_prompt(question, matches):
    context = format_context(matches)
    return f"""Tu es un assistant qui répond aux questions sur les factures et devis d'une entreprise.
Utilise uniquement les informations fournies dans le contexte ci-dessous pour répondre.
Si l'information demandée n'est pas présente dans le contexte, dis-le clairement plutôt que d'inventer une réponse.
Cite les documents pertinents par leur identifiant quand c'est utile.

Contexte:
{context}

Question: {question}

Réponds en français, de façon concise et précise.
"""


def build_aggregation_prompt(question, aggregation):
    return f"""Voici le résultat d'une requête agrégée sur la base de facturation:
{aggregation}

Question posée: {question}

Rédige une réponse en français, en une ou deux phrases, en reformulant ces chiffres de façon naturelle.
"""


def build_report_prompt(question, df, group_label, metric_label):
    table_preview = df.to_string(index=False)
    return f"""Voici un tableau de données agrégées extrait de la base de facturation, regroupé par {group_label},
avec pour indicateur principal : {metric_label}.

{table_preview}

Question posée: {question}

Rédige un court paragraphe (3 à 5 phrases) en français, destiné à un responsable financier,
résumant les tendances principales visibles dans ce tableau (valeurs les plus élevées,
concentration éventuelle, points d'attention). Ne recopie pas le tableau, commente-le.
"""


def generate_text(prompt):
    def call():
        return client.models.generate_content(
            model=GENERATION_MODEL,
            contents=prompt,
        )

    response = call_with_retry(call)
    return response.text


def handle_question(question, conn_holder):
    if is_report_request(question):
        def narrative_fn(df, group_label, metric_label):
            prompt = build_report_prompt(question, df, group_label, metric_label)
            return generate_text(prompt)

        report = generate_report(conn_holder, question, narrative_fn=narrative_fn)

        print("\nRAPPORT")
        print("=" * 70)
        print(report["narrative"])
        print("\nRapport HTML   :", report["html_path"])
        print("Tableau Excel  :", report["excel_path"])
        print("Graphique PNG  :", report["chart_path"])
        print("\nAperçu des données:")
        print(report["dataframe"].to_string(index=False))
        return

    result = search(question, conn_holder)

    if result.mode == "aggregation":
        prompt = build_aggregation_prompt(question, result.aggregation)
        answer = generate_text(prompt)
        print("\nRÉPONSE")
        print("=" * 70)
        print(answer)
        print("\nDétail:", result.aggregation)
        return

    if not result.matches:
        print("\nAucun document pertinent trouvé pour cette question.")
        return

    prompt = build_qa_prompt(question, result.matches)
    answer = generate_text(prompt)

    print("\nRÉPONSE")
    print("=" * 70)
    print(answer)
    print("\nDocuments utilisés:")
    for match in result.matches[:5]:
        print(f"- {match.document_id} (score RRF: {match.rrf_score:.4f})")


def main():
    conn_holder = [connect(DATABASE_URL)]

    print("Agent RAG - Factures et devis")
    print("Tapez votre question, ou 'exit' pour quitter.")

    while True:
        question = input("\nQuestion: ").strip()
        if question.lower() in ("exit", "quit"):
            break
        if not question:
            continue

        try:
            handle_question(question, conn_holder)
        except Exception as error:
            print(f"Erreur lors du traitement de la question: {error}")

    conn_holder[0].close()


if __name__ == "__main__":
    main()