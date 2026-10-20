#!/usr/bin/env bash
# ==============================================================================
# Script de Instalación Automatizada de CELab Bot para Servidor Linux
# Repositorio: https://github.com/Marianinpb/Jarvis-proyect
# ==============================================================================

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

INSTALL_DIR="/opt/celab-bot"
CONFIG_DIR="/etc/celab-bot"
REPO_URL="https://github.com/Marianinpb/Jarvis-proyect.git"

echo -e "${BLUE}======================================================${NC}"
echo -e "${BLUE}   🚀 Instalador Automatizado de CELab Bot            ${NC}"
echo -e "${BLUE}======================================================${NC}"

# 1. Comprobar permisos de root
if [ "$EUID" -ne 0 ]; then
  echo -e "${RED}❌ Este script debe ejecutarse con privilegios de root (usa sudo).${NC}"
  exit 1
fi

REAL_USER="${SUDO_USER:-$USER}"
echo -e "${GREEN}✓ Ejecutando instalación como root (solicitado por ${REAL_USER})${NC}"

# 2. Instalar dependencias base del sistema
echo -e "\n${BLUE}[1/7] Verificando dependencias del sistema...${NC}"
if command -v apt-get &> /dev/null; then
    apt-get update -qq || true
    apt-get install -y -qq git curl python3 python3-venv python3-pip
elif command -v dnf &> /dev/null; then
    dnf install -y git curl python3 python3-pip
elif command -v pacman &> /dev/null; then
    pacman -Sy --noconfirm git curl python python-pip
fi

# Detectar o instalar uv
UV_BIN=$(command -v uv || true)
[ -z "$UV_BIN" ] && [ -x "$HOME/.local/bin/uv" ] && UV_BIN="$HOME/.local/bin/uv"
[ -z "$UV_BIN" ] && [ -x "/root/.local/bin/uv" ] && UV_BIN="/root/.local/bin/uv"

if [ -z "$UV_BIN" ]; then
    echo -e "${YELLOW}Instalando gestor uv para instalación rápida de Python...${NC}"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH:/root/.local/bin"
    UV_BIN=$(command -v uv || true)
    [ -z "$UV_BIN" ] && [ -x "$HOME/.local/bin/uv" ] && UV_BIN="$HOME/.local/bin/uv"
    [ -z "$UV_BIN" ] && [ -x "/root/.local/bin/uv" ] && UV_BIN="/root/.local/bin/uv"
fi

# 3. Crear grupos y usuarios de sistema
echo -e "\n${BLUE}[2/7] Configurando usuarios y grupos de sistema...${NC}"
if ! getent group celab-users >/dev/null; then
    groupadd celab-users
    echo -e "${GREEN}✓ Grupo 'celab-users' creado.${NC}"
fi

if ! getent group celab-admins >/dev/null; then
    groupadd celab-admins
    echo -e "${GREEN}✓ Grupo 'celab-admins' creado.${NC}"
fi

if ! id -u celab-bot &>/dev/null; then
    useradd --system --shell /bin/false --home-dir "$INSTALL_DIR" -g celab-users celab-bot
    echo -e "${GREEN}✓ Usuario de sistema 'celab-bot' creado.${NC}"
fi

# Añadir al usuario actual que ejecuta sudo a celab-admins y celab-users
if [ -n "$REAL_USER" ] && id -u "$REAL_USER" &>/dev/null; then
    usermod -aG celab-users,celab-admins "$REAL_USER"
    echo -e "${GREEN}✓ Usuario '${REAL_USER}' añadido a celab-admins y celab-users.${NC}"
fi

# 4. Clonar o actualizar repositorio en /opt/celab-bot
echo -e "\n${BLUE}[3/7] Descargando código en ${INSTALL_DIR}...${NC}"
git config --global --add safe.directory "$INSTALL_DIR" 2>/dev/null || true
if [ -d "$INSTALL_DIR/.git" ]; then
    echo -e "${YELLOW}El directorio ya existe. Sincronizando con git fetch & reset...${NC}"
    git -C "$INSTALL_DIR" fetch origin
    git -C "$INSTALL_DIR" reset --hard origin/master 2>/dev/null || git -C "$INSTALL_DIR" reset --hard origin/main
else
    mkdir -p "$INSTALL_DIR"
    git clone "$REPO_URL" "$INSTALL_DIR"
fi

# 5. Configurar directorios y permisos iniciales
echo -e "\n${BLUE}[4/7] Configurando directorios y permisos de ejecución...${NC}"
mkdir -p "$CONFIG_DIR"
mkdir -p "$INSTALL_DIR/shared/tools"
mkdir -p "$INSTALL_DIR/shared/recipes"
mkdir -p "$INSTALL_DIR/bin"

if [ ! -f "$CONFIG_DIR/config.toml" ]; then
    cp "$INSTALL_DIR/config.toml.example" "$CONFIG_DIR/config.toml"
    echo -e "${GREEN}✓ Creado ${CONFIG_DIR}/config.toml por defecto.${NC}"
fi

if [ ! -f "$CONFIG_DIR/users.toml" ]; then
    touch "$CONFIG_DIR/users.toml"
    echo -e "${GREEN}✓ Creado ${CONFIG_DIR}/users.toml vacío.${NC}"
fi

# Enlaces simbólicos globales en /usr/local/bin
ln -sf "$INSTALL_DIR/bin/celab-bot" /usr/local/bin/celab-bot
ln -sf "$INSTALL_DIR/bin/celab-exec" /usr/local/bin/celab-exec

# 6. Crear entorno virtual e instalar dependencias
echo -e "\n${BLUE}[5/7] Creando entorno virtual de Python e instalando paquetes...${NC}"
if [ -n "$UV_BIN" ] && [ -x "$UV_BIN" ]; then
    echo -e "${GREEN}Usando uv ($UV_BIN)...${NC}"
    "$UV_BIN" venv "$INSTALL_DIR/venv"
    "$UV_BIN" pip install --python "$INSTALL_DIR/venv" -e "$INSTALL_DIR"
    "$UV_BIN" pip install --python "$INSTALL_DIR/venv" laya torch transformers safetensors
else
    python3 -m venv "$INSTALL_DIR/venv"
    if [ ! -x "$INSTALL_DIR/venv/bin/pip" ]; then
        curl -sS https://bootstrap.pypa.io/get-pip.py | "$INSTALL_DIR/venv/bin/python"
    fi
    "$INSTALL_DIR/venv/bin/pip" install --upgrade pip
    "$INSTALL_DIR/venv/bin/pip" install -e "$INSTALL_DIR"
    "$INSTALL_DIR/venv/bin/pip" install laya torch transformers safetensors
fi

# Ajustar permisos finales tras crear el venv
chown -R celab-bot:celab-users "$INSTALL_DIR"
chown -R celab-bot:celab-users "$CONFIG_DIR"
chmod -R 775 "$INSTALL_DIR/shared"
chmod -R 775 "$CONFIG_DIR"
chmod +x "$INSTALL_DIR/bin/celab-bot"
chmod +x "$INSTALL_DIR/bin/celab-exec"


# 7. Configurar sudoers y systemd
echo -e "\n${BLUE}[6/7] Configurando seguridad sudoers y systemd...${NC}"
cp "$INSTALL_DIR/celab-bot.sudoers" /etc/sudoers.d/celab-bot
chmod 440 /etc/sudoers.d/celab-bot

cp "$INSTALL_DIR/celab-bot.service" /etc/systemd/system/celab-bot.service
systemctl daemon-reload
systemctl enable celab-bot.service
systemctl restart celab-bot.service

echo -e "\n${BLUE}[7/7] Verificando servicio...${NC}"
sleep 2
if systemctl is-active --quiet celab-bot.service; then
    echo -e "${GREEN}✅ ¡CELab Bot está ACTIVO y corriendo en segundo plano!${NC}"
else
    echo -e "${YELLOW}⚠️ El servicio arrancó pero revisa los logs con: journalctl -u celab-bot -n 20${NC}"
fi

echo -e "\n${GREEN}======================================================${NC}"
echo -e "${GREEN}   🎉 ¡Instalación Completada Exitosamente!           ${NC}"
echo -e "${GREEN}======================================================${NC}"
echo -e "Comandos útiles:"
echo -e "  • Vincular usuario: ${YELLOW}sudo celab-bot link \"Nombre OpenWebUI\" usuario_linux${NC}"
echo -e "  • Ver vinculaciones: ${YELLOW}celab-bot users${NC}"
echo -e "  • Ver estado y logs: ${YELLOW}celab-bot status${NC} | ${YELLOW}celab-bot logs${NC}"
echo -e "\nEn OpenWebUI (Settings -> Connections -> OpenAI):"
echo -e "  • URL: ${YELLOW}http://host.docker.internal:8000/v1${NC} (o la IP del host en Docker)"
echo -e "  • API Key: ${YELLOW}celab-secret-key${NC}"
