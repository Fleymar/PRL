# PRL — Notes de projet IA Rocket League

---

## Résumé général du projet

Objectif : entraîner un bot Rocket League autonome en utilisant du **Reinforcement Learning** (algorithme PPO), puis l'intégrer dans RLBot pour jouer en temps réel contre un humain.

Stack technique :
- **rlgym-ppo** : implémentation PPO spécialisée pour Rocket League
- **RocketSim** : moteur physique de simulation (très rapide, pas besoin du vrai jeu pour entraîner)
- **RLBot** : framework pour faire jouer des bots dans Rocket League
- **PyTorch** + CUDA 12.8 (RTX 5080)
- **TensorBoard** : monitoring de l'entraînement en temps réel

---

## 1. Architecture de l'observation (ce que le bot "voit")

Le bot reçoit à chaque tick un vecteur de **101 floats** décrivant l'état du jeu :

| Bloc | Taille | Contenu |
|---|---|---|
| Balle | 9 | position, vitesse linéaire, vitesse angulaire |
| Boost pads | 34 | timer de recharge de chaque pad sur la map |
| Variables partielles | 9 | est en saut, a un flip disponible, temps en l'air... |
| Voiture self | 20 | position, orientation (forward/up), vitesses, boost, état |
| Voiture adversaire | 20 | même chose pour l'adversaire |
| Vecteurs relatifs | 9 | balle→but adverse, balle→notre cage, voiture→balle |

**Total : 101 floats**

Toutes les valeurs sont normalisées (position divisée par dimensions du terrain, vitesses divisées par max, etc.).

Pour l'équipe orange, tous les vecteurs sont inversés selon l'axe `[-1, -1, 1]` pour que le bot ait toujours la même perspective quelle que soit son équipe.

---

## 2. Architecture de l'action (ce que le bot "fait")

### Évolution au cours du projet

**Version initiale** : `LookupTableAction` — 90 actions discrètes prédéfinies (combinaisons throttle/steer/jump/boost...). Le réseau choisit 1 action parmi 90.

**Version finale** : `MultiDiscrete` — 8 dimensions d'action **indépendantes**, chacune étant un choix discret :

| Dimension | Valeurs possibles |
|---|---|
| Throttle (accélération) | -1, 0, +1 |
| Steer (direction) | -1, 0, +1 |
| Pitch (tangage) | -1, 0, +1 |
| Yaw (lacet) | -1, 0, +1 |
| Roll (roulis) | -1, 0, +1 |
| Jump | 0, 1 |
| Boost | 0, 1 |
| Handbrake | 0, 1 |

Avantage : le bot apprend "accélérer" et "tourner" comme des concepts séparés, au lieu de mémoriser 90 combinaisons. Meilleure généralisation, espace d'action structuré.

Chaque action est répétée **8 frames** (~133ms) avant que le réseau calcule la suivante.

---

## 3. Architecture du réseau de neurones

- **Policy (acteur)** : MultiDiscreteFF — réseau feed-forward `101 → 512 → 512 → 20 sorties` (20 = somme des bins : 5×3 + 3×2)
- **Critic (valeur)** : réseau séparé `101 → 512 → 512 → 1` (estime la valeur d'un état)
- Activation : ReLU
- Optimiseur : Adam

---

## 4. Algorithme PPO — hyperparamètres finaux

| Paramètre | Valeur | Rôle |
|---|---|---|
| `ppo_batch_size` | 60 000 | Timesteps collectés avant chaque update |
| `ts_per_iteration` | 60 000 | Idem |
| `exp_buffer_size` | 180 000 | Buffer d'expérience (3× batch) |
| `ppo_minibatch_size` | 6 000 | Taille des mini-batches pour gradient |
| `ppo_epochs` | 3 | Passes sur chaque batch |
| `ppo_ent_coef` | 0.005 | Coefficient d'entropie (encourage exploration) |
| `policy_lr` | 5e-4 → 5e-5 | Learning rate avec décroissance linéaire |
| `n_proc` | 12 | Nombre d'environnements parallèles |
| `timestep_limit` | 100 000 000 | 100M steps au total |

---

## 5. Fonction de récompense — Curriculum Learning

La récompense évolue automatiquement en 3 phases selon le nombre de steps globaux.

### Phase 1 (0 → 20M steps) : apprendre à bouger et toucher la balle

Poids **fixes** (ne changent pas dans la phase) :

| Reward | Poids | Description |
|---|---|---|
| GoalReward | 5.0 | +1 si but, -1 si but encaissé |
| VelocityBallToGoal | 0.5 | Balle qui se dirige vers le but adverse |
| VelocityPlayerToBall | 2.0 | Voiture qui se dirige vers la balle |
| TouchReward | 3.0 | Toucher la balle |

### Phase 2 (20M → 60M steps) : orienter la balle vers le but

Transition progressive depuis Phase 1 :

| Reward | Poids début | Poids fin |
|---|---|---|
| GoalReward | 5.0 | 15.0 |
| VelocityBallToGoal | 0.5 | 4.0 |
| VelocityPlayerToBall | 2.0 | 0.5 |
| TouchReward | 3.0 | 1.0 |
| DefensivePenalty | 0.0 | 0.002 |
| HighVelocitySave | 0.0 | 3.0 |

### Phase 3 (60M → 100M steps) : marquer / défendre

| Reward | Poids début | Poids fin |
|---|---|---|
| GoalReward | 15.0 | 25.0 |
| VelocityBallToGoal | 4.0 | 2.0 |
| VelocityPlayerToBall | — | 0.3 (fixe) |
| TouchReward | — | 0.5 (fixe) |
| DefensivePenalty | 0.002 | 0.004 |
| HighVelocitySave | 3.0 | 5.0 |

**Rewards personnalisées développées :**
- `VelocityPlayerToBallReward` : dot product vitesse voiture × direction vers balle, normalisé par CAR_MAX_SPEED
- `VelocityBallToGoalReward` : dot product vitesse balle × direction vers but adverse
- `DefensivePenaltyReward` : pénalité continue quand la balle est dans notre camp (évite le "bot kamikaze")
- `HighVelocitySaveReward` : récompense les arrêts défensifs quand la balle fonce vers notre cage à >800 UU/s

**Rewards intentionnellement évitées** (mauvaises pratiques identifiées) :
- `BehindBallReward` : crée un "Goalie Camper" qui reste derrière la balle sans jouer
- `FlipPenaltyReward` : récompenser le "comment" plutôt que le "quoi" (anti-pattern RL)
- `SaveBoostReward` : le bot hoarde le boost au lieu de jouer

---

## 6. Learning Rate Scheduler

Décroissance linéaire de 5e-4 à 5e-5 sur 100M steps, mise à jour à chaque itération via un hook dans le callback TensorBoard :

```
LR(step) = 5e-4 + (5e-5 - 5e-4) × (step / 100M)
```

Implémenté via une variable globale `_learner` accessible depuis le callback `report_metrics`.

---

## 7. Intégration RLBot (bot.py)

Le bot charge automatiquement le dernier checkpoint sauvegardé, reconstruit l'observation de 101 floats depuis le `GamePacket` RLBot, passe l'obs au réseau MultiDiscreteFF, et convertit les 8 indices de sortie en `ControllerState`.

Spécificités de l'implémentation :
- Répétition d'action sur 8 frames (cohérent avec l'entraînement)
- Suivi manuel du `air_time_since_jump` à ~60Hz pour calculer si le bot a encore un flip
- Gestion de l'inversion équipe orange
- Détection de l'état (au sol, en saut, démoli) via `AirState` enum de RLBot

---

## 8. Diagnostic et problèmes rencontrés

### KL Divergence quasi-nulle (~0.000005)
**Cause** : `ppo_epochs=1`, `ppo_minibatch_size=10000` (= batch complet), `lr=5e-5` → la policy ne se mettait quasiment pas à jour.  
**Fix** : epochs=3, minibatch=6000, lr=5e-4.

### Observations incorrectes (shape mismatch)
- Bot produisant 47 floats au lieu de 92 → structure réelle de DefaultObs découverte (9+34+9+20+20)
- Modèle entraîné sur 92 floats alors que le bot envoyait 101 → ajout des 9 vecteurs relatifs dans bot.py
- Ancien checkpoint (92 floats) auto-chargé lors du redémarrage avec nouveau modèle (101) → déplacé dans `checkpoints_old/`

### Erreurs d'attributs RLBot
- `has_wheel_contact` → `air_state == AirState.OnGround`
- `double_jumped` → `has_double_jumped`
- `is_demolished` → `demolished_timeout > 0`

### Support GPU (RTX 5080)
PyTorch installé en version CPU-only par défaut → réinstallé avec CUDA 12.8 :
```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cu128 --upgrade
```

### TensorBoard ne se mettait pas à jour en live
Fix : `SummaryWriter(log_dir=..., flush_secs=10)` pour forcer le flush régulier.

### Nom d'attribut optimizer incorrect
`value_net_optimizer` → `value_optimizer` (nom réel dans rlgym-ppo PPOLearner).

---

## 9. Commandes utiles

```bash
# Depuis le dossier AI/

# Lancer l'entraînement
../venv/Scripts/python train_ppo.py

# TensorBoard (autre terminal)
../venv/Scripts/python -m tensorboard.main --logdir data/tensorboard --reload_interval 10

# Vérifier CUDA
../venv/Scripts/python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"

# Archiver les checkpoints avant de repartir de zéro
mv data/checkpoints/* data/checkpoints_old/ 2>/dev/null
rm -rf data/tensorboard/*
```

---

## 10. Résultats d'entraînement observés

| Run | Steps | Observation |
|---|---|---|
| Run 1 | ~1.7M | KL ~0 → policy bloquée, hyperparamètres corrigés |
| Run 2 | ~3M | Bot va vers la balle et tape, mais n'essaie pas de marquer |
| Run 3 | ~30M | Policy Reward monte 0→600 puis plateau/légère baisse — curriculum Phase 1 trop instable (poids qui diminuaient), ent_coef trop élevé |
| Run 4 (en cours) | — | Phase 1 poids fixes, ent_coef 0.01→0.005 |