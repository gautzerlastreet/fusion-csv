import random
import streamlit as st

# Fonction de lancer de dés (e.g. '1d6')
def roll_dice(dice_str: str) -> int:
    """
    Génère un résultat aléatoire à partir d'une chaîne de dés au format 'NdM'.
    """
    count, sides = map(int, dice_str.lower().split('d'))
    return sum(random.randint(1, sides) for _ in range(count))

# Listes de base
noms = [
    'Arin','Belen','Caelis','Doran','Elyra','Fenris','Galad','Hela','Ivar','Jora',
    'Kael','Lirien','Morthos','Nalia','Orin','Pyris','Quel','Ryn','Sylas','Therin',
    'Ulric','Vyra','Wren','Xand','Ysera','Zorin'
]
races = ['Humain','Elfe','Nain','Orc','Gobelin','Tiefling','Drakéide','Gnome','Demi-elfe','Demi-orc']
classes = ['Guerrier','Mage','Voleur','Clerc','Rôdeur','Paladin','Barde','Druide','Moine','Sorcier']
metiers = ['Forgeron','Apothicaire','Alchimiste','Marchand','Agriculteur','Pêcheur','Chasseur','Herboriste','Tanneur','Menuisier']

physical_sentences = [
    "La silhouette est robuste, avec des épaules larges et une carrure imposante.",
    "Ses yeux perçants brillent d'une lueur intense.",
    "Un tatouage mystérieux orne son bras gauche.",
    "Il porte une cicatrice fine sur la joue droite.",
    "Ses cheveux sont coupés court, d'un noir profond.",
    "Sa posture est droite et assurée.",
    "On devine des muscles saillants sous son armure légère.",
    "Son visage est marqué par l'expérience et la fatigue.",
    "Il se déplace avec une grâce surprenante pour son gabarit.",
    "Une légère boiterie trahit une blessure ancienne."
]
mentality_sentences = [
    "Il est d'un calme presque surnaturel, réfléchissant avant d'agir.",
    "Son esprit vif décèle les faiblesses de ses adversaires.",
    "Il a un sens de l'humour inattendu, même en plein combat.",
    "Sa loyauté envers ses compagnons est sans faille.",
    "Il se méfie naturellement des étrangers.",
    "Son ambition le pousse toujours à se surpasser.",
    "Il est curieux de tout, posant sans cesse des questions.",
    "Sa colère est redoutable lorsqu'elle est éveillée.",
    "Il est animé par un désir de justice.",
    "Sa patience lui permet de résoudre les situations les plus complexes."
]

weapons = [
    {'name': 'Dague', 'dice': '1d4'},
    {'name': 'Épée courte', 'dice': '1d6'},
    {'name': 'Masse', 'dice': '1d8'},
    {'name': 'Épée longue', 'dice': '1d10'},
    {'name': 'Hache de bataille', 'dice': '1d12'}
]
armors = [
    {'name': 'Armure de cuir', 'dice': '1d4'},
    {'name': 'Cotte de mailles', 'dice': '1d6'},
    {'name': 'Plastron', 'dice': '1d8'},
    {'name': 'Armure légère', 'dice': '1d10'},
    {'name': 'Armure lourde', 'dice': '1d12'}
]

magic_types = ['Utilitaire','Offensive','Défensive']
spells = {
    'Utilitaire': ['Lumière','Téléportation mineure','Purification'],
    'Offensive': ['Boule de feu','Éclair','Rayon de givre'],
    'Défensive': ['Bouclier magique','Armure de mage','Barrière']
}

useful_loot = [
    'Potion de soin','Potion de mana','Rations de voyage','Torche','Corde (15m)',
    'Kit de réparation','Parchemin de carte','Trousse de premiers secours','Lunettes de vision nocturne',
    'Ampoule d’eau','Pierre à feu','Anneau de respiration aquatique','Sac sans fond','Bottes de vitesse',
    'Cape d’invisibilité','Bâton de marche','Encens de purification','Gants de force','Médaillon de protection',
    'Pierre de rappel','Sac de couchage','Balise magique','Collier de silence','Sifflet de rappel','Clé universelle'
]
nuisance_loot = [
    'Cuillère en bois','Chaussette trouée','Plume de poule','Pierre polie','Bouquet de fleurs fanées',
    'Coquillage','Ficelle','Galet lisse','Mouchoir sale','Brique','Boîte vide','Plume de corbeau',
    'Graine inconnue','Tonneau vide','Carte d’un royaume inconnu','Miroir brisé','Journal mouillé','Chausson troué',
    'Ossement minuscule','Dé usé','Boîte à musique cassée','Morceau de charbon','Deux pierres identiques',
    'Vieille chaussure','Clou rouillé'
]
loot_pool = useful_loot + nuisance_loot

# Génération complète du personnage
def generate_character() -> dict:
    nom        = random.choice(noms)
    race       = random.choice(races)
    classe     = random.choice(classes)
    metier     = random.choice(metiers)
    physique   = " ".join(random.sample(physical_sentences, 2))
    mental     = " ".join(random.sample(mentality_sentences, 2))
    weapon     = random.choice(weapons)
    armor      = random.choice(armors)
    type_magie = random.choice(magic_types)
    sort       = random.choice(spells[type_magie])
    loot       = random.sample(loot_pool, 5)
    return {
        'Nom': nom,
        'Race': race,
        'Classe': classe,
        'Métier': metier,
        'Physique': physique,
        'Mental': mental,
        'Arme': f"{weapon['name']} (dégâts : {weapon['dice']})",
        'Armure': f"{armor['name']} (protection : {armor['dice']})",
        'Magie': f"{type_magie} - Sort : {sort}",
        'Loot': loot
    }

# Interface Streamlit
def run() -> None:
    st.title("Générateur de personnage RPG")
    if st.button("Générer un personnage"):
        char = generate_character()
        st.subheader(f"{char['Nom']}")
        st.markdown(f"**Race :** {char['Race']}")
        st.markdown(f"**Classe :** {char['Classe']}")
        st.markdown(f"**Métier :** {char['Métier']}")
        st.markdown(f"**Physique :** {char['Physique']}")
        st.markdown(f"**Mental :** {char['Mental']}")
        st.markdown(f"**Arme :** {char['Arme']}")
        st.markdown(f"**Armure :** {char['Armure']}")
        st.markdown(f"**Magie :** {char['Magie']}")
        st.markdown("**Loot :**")
        for item in char['Loot']:
            st.write(f"- {item}")