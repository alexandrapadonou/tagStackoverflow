# Procfile
# Fichier de configuration pour le déploiement sur Heroku
# Définit les processus à exécuter au démarrage de l'application

# Processus web: démarre le serveur uvicorn pour l'API FastAPI
# --host 0.0.0.0: écoute sur toutes les interfaces réseau (nécessaire pour Heroku)
# --port $PORT: utilise le port fourni par Heroku via la variable d'environnement PORT
web: uvicorn api.main:app --host 0.0.0.0 --port $PORT