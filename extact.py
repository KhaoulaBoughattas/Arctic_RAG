# extraction.py

import os
import re
from typing import List
from unstructured.partition.auto import partition
from unstructured.cleaners.core import clean

DATA_DIR = "data"
OUTPUT_FILE = os.path.join(DATA_DIR, "texte_nettoye.txt")


# 🔧 Supprimer les footers répétés
def supprimer_footers_robuste(texte: str) -> str:
    motifs = [
        # Variante avec ou sans "Publié sur...", avec espace ou tab entre "Page" et le numéro
        r"Dugas\s+M\.?\s+et\s+al\s*\(2017\)\s*Guide\s+de\s+pratique\s+pour\s+l[’'´]?évaluation\s+et\s+le\s+traitement\s+cognitivo[\s\-]?comportemental\s+du\s+trouble\s+d[’'´]?anxiété\s+généralisée\.?\s*(Publié\s+sur\s+www\.tccmontreal\.com)?\s*Page[\s\t]*\d+",
        
        # Variante plus courte sans "Page"
        r"Dugas\s+M\.?\s+et\s+al\s*\(2017\)\s*Guide\s+de\s+pratique\s+pour\s+l[’'´]?évaluation\s+et\s+le\s+traitement\s+cognitivo[\s\-]?comportemental\s+du\s+trouble\s+d[’'´]?anxiété\s+généralisée\.?",
        
        # Variante alternative du titre (ancienne version du guide)
        r"Guide\s+de\s+pratique\s+pour\s+le\s+diagnostic\s+et\s+le\s+traitement\s+cognitivo\s+comportemental\s+du\s+trouble\s+anxiété\s+généralisée"
    ]

    for motif in motifs:
        texte = re.sub(motif, "", texte, flags=re.IGNORECASE)

    return texte


# 🔢 Supprimer les numéros de page
def supprimer_numeros_de_page(texte: str) -> str:
    lignes = texte.split('\n')
    lignes_nettoyees = []

    for ligne in lignes:
        # Supprimer les lignes contenant uniquement un chiffre (ex: "4", "23")
        if re.fullmatch(r'\s*\d{1,3}\s*', ligne):
            continue
        lignes_nettoyees.append(ligne)

    texte = '\n'.join(lignes_nettoyees)

    # Supprimer les groupes de chiffres perdus dans le texte (ex: "4 5 6 Section")
    texte = re.sub(r'(?:\s*\b\d{1,3}\b){2,}', ' ', texte)

    return texte.strip()


# ✂️ Supprimer la table des matières automatiquement
def supprimer_table_matiere(texte: str) -> str:
    texte = re.sub(r"(?i)table\s+des\s+matières.*?(?=\n\s*Section\s+\d+|Chapitre\s+\d+|\Z)", "", texte, flags=re.DOTALL)
    texte = re.sub(r"^.*?\.{3,}\s*\d+\s*$", "", texte, flags=re.MULTILINE)
    texte = re.sub(r"^\d+(\.\d+)*\s+.*?\.{3,}\s*\d+\s*$", "", texte, flags=re.MULTILINE)
    return texte


# 🧽 Supprimer les métadonnées éditoriales et bibliographiques
def supprimer_metadonnees_bibliographiques(texte: str) -> str:
    motifs = [
        r"ISBN[\s:\d\-]+",
        r"Dépôt légal.*?\d{4}",
        r"1re édition",
        r"Bibliothèque et Archives nationales.*?\d{4}",
        r"Bibliothèque et Archives Canada.*?\d{4}",
        r"Publié sur www\.tccmontreal\.com",
        r"Auteurs?\s?:.*",
        r"Éditeur\s?:.*",
        r"tccmontreal\s?",
        r"Dugas,\s*M\.\s*,?\s*Ngô,\s*T\.\s*L\.\s*et\s*coll\.?",
    ]
    for motif in motifs:
        texte = re.sub(motif, "", texte, flags=re.IGNORECASE)
    return texte


# 🔁 Corriger les puces mal extraites ("!" → "•")
def corriger_puces_exclamees(texte: str) -> str:
    # Remplace les lignes qui commencent par ! suivi d’un espace ou d’un mot, par une puce
    texte = re.sub(r"(?m)^\s*\!\s*", "• ", texte)       # début de ligne
    texte = re.sub(r"(?<=\n)\s*\!\s*", "\n• ", texte)   # après un saut de ligne
    return texte


# 🧹 Nettoyage complet
def nettoyage_complet(texte: str) -> str:
    texte = supprimer_table_matiere(texte)
    texte = supprimer_metadonnees_bibliographiques(texte)
    texte = supprimer_numeros_de_page(texte)
    texte = corriger_puces_exclamees(texte)
    texte = clean(
        text=texte,
        extra_whitespace=True,
        dashes=True,
        bullets=True
    )
    texte = supprimer_footers_robuste(texte)
    texte = re.sub(r"\s{2,}", " ", texte)
    texte = re.sub(r"\n{3,}", "\n\n", texte)
    return texte.strip()


# 🔍 Extraction PDF
def extraire_et_nettoyer_pdf(chemin_pdf: str) -> str:
    try:
        print(f"📄 Extraction : {os.path.basename(chemin_pdf)}")
        elements = partition(filename=chemin_pdf)
        texte_brut = "\n".join(str(element) for element in elements if element)
        texte_final = nettoyage_complet(texte_brut)
        return texte_final
    except Exception as e:
        print(f"❌ Erreur lors de l'extraction de {chemin_pdf} : {e}")
        return ""


# 📂 Traitement en lot
def traiter_tous_les_pdfs(data_dir: str = DATA_DIR) -> str:
    corpus = []
    for fichier in os.listdir(data_dir):
        if fichier.lower().endswith(".pdf"):
            chemin = os.path.join(data_dir, fichier)
            texte = extraire_et_nettoyer_pdf(chemin)
            if texte:
                corpus.append(texte)
    return "\n\n".join(corpus)


# 🚀 Point d'entrée
if __name__ == "__main__":
    texte_final = traiter_tous_les_pdfs()
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(texte_final)
    print(f"\n✅ Fichier généré avec nettoyage intégré : {OUTPUT_FILE}")
