# TP Fuzzy C-Means - Segmentation d'images

Implémentation de l'algorithme Fuzzy C-Means (FCM) en Python pour la segmentation d'images en régions floues (M2 Informatique VMI - Reconnaissance des formes avancée, Université Paris Cité).

## Fichiers

- `tp_fcm.py` : code principal (implémentation de FCM et tests de segmentation).
- `milky-way-nvg.jpg` : image de test en niveaux de gris.
- `milky-way.jpg` : image de test en couleur (RGB).
- `resultat_nvg_c2.png` : résultat obtenu sur l'image en niveaux de gris ($C = 2$).
- `resultat_rgb_c3.png` : résultat obtenu sur l'image couleur ($C = 3$).

## Prérequis et exécution

Dépendances : `numpy`, `matplotlib`, `Pillow`.

```bash
pip install numpy matplotlib pillow
python tp_fcm.py
```
