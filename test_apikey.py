import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Charger .env
load_dotenv()

# Récupérer la clé
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError(" GEMINI_API_KEY introuvable dans le fichier .env")

print(" Clé Gemini trouvée.")

client = genai.Client(api_key=api_key)

MODEL = "gemini-embedding-001"

text = """
Facture numéro 4095-537.
Fournisseur : ABC Company.
Montant TTC : 249.15 EUR.
"""

print(" Génération de l'embedding...")

response = client.models.embed_content(
    model=MODEL,
    contents=text,
    config=types.EmbedContentConfig(output_dimensionality=1536)
)

embedding = response.embeddings[0].values

print("✅ Embedding généré avec succès !")
print(f" Dimension : {len(embedding)}")

# Afficher seulement les 10 premières valeurs
print("\n 10 premières valeurs :")
print(embedding[:10])

# Vérification
if len(embedding) == 1536:
    print("\n TEST RÉUSSI : vecteur de 1536 dimensions.")
else:
    print(
        f"\n Dimension reçue : {len(embedding)} "
        "au lieu de 1536."
    )