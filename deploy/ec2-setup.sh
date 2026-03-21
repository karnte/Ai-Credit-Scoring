#!/usr/bin/env bash
# EC2 initialization script for AI Credit Scoring
# Instance: m7i-flex.large (2 vCPU, 8GB RAM)
# Usage: Run as root on a fresh Ubuntu 22.04/24.04 EC2 instance
set -euo pipefail

echo "=== AI Credit Scoring — EC2 Setup ==="

# System updates
apt-get update && apt-get upgrade -y

# Install Docker Engine
apt-get install -y ca-certificates curl gnupg
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
chmod a+r /etc/apt/keyrings/docker.gpg
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  tee /etc/apt/sources.list.d/docker.list > /dev/null
apt-get update
apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

# Create app user
useradd -m -s /bin/bash creditscoring || true
usermod -aG docker creditscoring

# App directory
mkdir -p /opt/credit-scoring
chown creditscoring:creditscoring /opt/credit-scoring

# 2GB swap (helps with 8GB RAM limit)
if [ ! -f /swapfile ]; then
    fallocate -l 2G /swapfile
    chmod 600 /swapfile
    mkswap /swapfile
    swapon /swapfile
    echo '/swapfile none swap sw 0 0' >> /etc/fstab
fi

# Firewall
apt-get install -y ufw
ufw default deny incoming
ufw default allow outgoing
ufw allow 22/tcp    # SSH
ufw allow 8000/tcp  # API
ufw --force enable

# Fail2ban
apt-get install -y fail2ban
systemctl enable fail2ban
systemctl start fail2ban

# UTF-8 locale for Thai text
apt-get install -y locales
locale-gen en_US.UTF-8 th_TH.UTF-8
update-locale LANG=en_US.UTF-8

# AWS CLI (for SSM Parameter Store secrets)
apt-get install -y awscli

# Copy systemd service
cp /opt/credit-scoring/deploy/systemd/credit-scoring.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable credit-scoring

echo "=== Setup complete ==="
echo "Next steps:"
echo "  1. Clone repo to /opt/credit-scoring/"
echo "  2. Copy .env.production to /opt/credit-scoring/.env"
echo "  3. Run: systemctl start credit-scoring"
