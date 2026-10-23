"""System 1 router: language by word counts, simple/complex by a small classifier (NB7).

It runs on every question, so it must be fast: an LLM router took 600-960 ms in NB5,
this one reuses the query embedding and took 14 ms on a T4.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from app.config import ROUTER_TAU
from app.encoders import embed_query, embedder
from app.search import tokenize

NL_WORDS = {"de", "het", "een", "ik", "mijn", "wordt", "mag", "kan", "wat", "hoe", "hoeveel", "zijn", "ben", "heb", "als",
            "voor", "niet", "er", "bij", "dat", "die", "welke", "geldt", "ook", "van", "op", "aan", "wie", "waar", "wanneer", "moet"}
EN_WORDS = {"the", "a", "an", "my", "can", "what", "how", "much", "are", "do", "does", "if", "for", "not", "to", "from",
            "which", "when", "who", "where", "should", "i", "of", "on", "and"}


def detect_language(question):
    """A tie (for example a bare ECLI number) means Dutch, the language of the sources."""
    words = tokenize(question)
    nl = sum(w in NL_WORDS for w in words)
    en = sum(w in EN_WORDS for w in words)
    return "en" if en > nl else "nl"


ROUTER_EXAMPLES = [   # (question, 0 = simple, 1 = complex): the 32 labelled examples of NB7
    ("Wat is het tarief in box 1 voor 2026?", 0),
    ("Is hypotheekrente aftrekbaar?", 0),
    ("Wat is het eigenwoningforfait?", 0),
    ("Mag ik werkkleding aftrekken?", 0),
    ("Wat betekent hoofdverblijf?", 0),
    ("Is een lijfrentepremie aftrekbaar?", 0),
    ("Wat staat er in artikel 3.114?", 0),
    ("Wanneer is een woning een eigen woning?", 0),
    ("Wat is de kamerverhuurvrijstelling?", 0),
    ("Hoeveel bedraagt de aftrek wegens geen of geringe eigenwoningschuld?", 0),
    ("What is the box 1 tax rate?", 0),
    ("Is mortgage interest deductible?", 0),
    ("What does article 3.16 say?", 0),
    ("Can I deduct the costs of a home office?", 0),
    ("What is the deemed rental value?", 0),
    ("Is there an exemption for renting out a room?", 0),
    # complex: several articles or rulings together, a dispute with the tax office, a comparison
    ("Ik verhuur een kamer via Airbnb en werk thuis in een andere kamer. Geldt de kamerverhuurvrijstelling, mag ik de werkruimte aftrekken, en hoe verhouden de uitspraken zich tot elkaar?", 1),
    ("Mijn inspecteur weigert de aftrek van mijn werkruimte omdat ik ook in loondienst werk. Welke uitspraken ondersteunen mijn bezwaar?", 1),
    ("Twee naast elkaar gelegen appartementen, waarvan één in aanbouw: is dat samen één eigen woning en wat betekent dat volgens de rechtspraak voor het eigenwoningforfait?", 1),
    ("Ik woon tijdelijk in het buitenland en verhuur mijn huis. Blijft het mijn hoofdverblijf, wat gebeurt er met mijn hypotheekrenteaftrek, en is er jurisprudentie over?", 1),
    ("Hoe verhoudt de uitspraak over de podcaststudio zich tot de eisen voor een zelfstandige werkruimte, en zou mijn situatie met een gedeelde zolder anders uitvallen?", 1),
    ("Mijn partner en ik hebben samen een woning en een tweede huis in aanbouw. Wie mag welke rente aftrekken en welke uitspraken zijn relevant?", 1),
    ("Ik heb een geschil met de Belastingdienst over de kamerverhuur aan mijn studerende neef: tellen de familieband en de korte huurperiode mee volgens de rechtspraak?", 1),
    ("Vergelijk de regels voor de aftrek van een werkruimte in de eigen woning met de rechtspraak en geef aan waar de rechters van elkaar afwijken.", 1),
    ("Ik heb bijna geen hypotheek meer, verhuur een kamer en betaal een lijfrente. Hoe werken deze regelingen samen in mijn aangifte?", 1),
    ("Mijn bezwaar over de huurwaarde van mijn leegstaande woning is afgewezen. Welke argumenten uit de wet en de rechtspraak kan ik in beroep gebruiken?", 1),
    ("I rent out a room on Airbnb and also work from home. Which exemptions and deductions apply, and how do the court rulings differ?", 1),
    ("My inspector rejected my home office deduction because I also have a salaried job. Which rulings support my objection?", 1),
    ("Compare how the courts have treated home offices in shared spaces and explain whether my converted garage would qualify.", 1),
    ("We own two adjacent apartments and are combining them during a renovation. Is this one owner-occupied home, and how does case law treat the construction period?", 1),
    ("I moved abroad for two years and rent out my house. Does it stay my main residence, what happens to my mortgage interest deduction, and is there case law?", 1),
    ("How do the rules on a home under construction interact with the deemed rental value, and what did the Supreme Court decide about buying only the land?", 1),
]

# Trained at startup in a few milliseconds. In production: thousands of helpdesk questions labelled by experts.
_X = embedder.encode(["query: " + q for q, _ in ROUTER_EXAMPLES], normalize_embeddings=True)
_y = np.array([label for _, label in ROUTER_EXAMPLES])
router_model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(_X, _y)


def route_question(question):
    """Returns (language, complex, P(complex))."""
    p = float(router_model.predict_proba([embed_query(question)])[0, 1])
    return detect_language(question), p > ROUTER_TAU, p
