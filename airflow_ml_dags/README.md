Инструкция сборки:  
export FERNET_KEY=$(python3 -c "from cryptography.fernet import Fernet; FERNET_KEY = Fernet.generate_key().decode(); print(FERNET_KEY)")  
export DOCKER_BUILDKIT=0  
export COMPOSE_DOCKER_CLI_BUILD=0  
docker compose build  
docker compose up  
