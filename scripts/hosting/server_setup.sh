#!/bin/bash
# Purpose: first-boot setup for the Patient Care Guide demo server (Amazon Linux 2023).
# deploy.py passes this to EC2 as user data after filling in the __PLACEHOLDERS__.
#
# The app runs as a systemd service on port 80, so it
# comes back by itself every time the EC2 scheduler starts the instance.
# No secrets are written into this script. At every start the server reads two
# encrypted values from SSM Parameter Store using its IAM role:
#   __PREFIX__/env         the app's .env (OpenAI, Langfuse, Tavily keys)
#   __PREFIX__/deploy-key  a read-only GitHub deploy key for the private repo
set -euxo pipefail

dnf install -y python3.12 python3.12-pip git dejavu-sans-fonts
useradd --system --create-home --home-dir /home/patientcare patientcare || true

# The refresh script runs as root before every start of the app:
# fetch the key + .env, pull the latest main, install any new packages.
cat > /usr/local/bin/patientcare-refresh <<'EOF'
#!/bin/bash
set -eu
APP=/opt/patientcare
SSH_DIR=/home/patientcare/.ssh
mkdir -p "$SSH_DIR"
aws ssm get-parameter --region __REGION__ --with-decryption --name __PREFIX__/deploy-key \
    --query Parameter.Value --output text > "$SSH_DIR/id_ed25519"
ssh-keyscan -t ed25519 github.com > "$SSH_DIR/known_hosts" 2>/dev/null
chmod 700 "$SSH_DIR"; chmod 600 "$SSH_DIR/id_ed25519"
chown -R patientcare:patientcare "$SSH_DIR"

if [ ! -d "$APP/.git" ]; then
    sudo -u patientcare git clone __REPO_SSH__ "$APP"
else
    sudo -u patientcare git -C "$APP" pull --ff-only || true   # a failed pull never blocks the app
fi
aws ssm get-parameter --region __REGION__ --with-decryption --name __PREFIX__/env \
    --query Parameter.Value --output text > "$APP/.env"
chown patientcare:patientcare "$APP/.env"; chmod 600 "$APP/.env"

if [ ! -x "$APP/.venv/bin/python" ]; then
    sudo -u patientcare python3.12 -m venv "$APP/.venv"
    sudo -u patientcare "$APP/.venv/bin/pip" install --upgrade pip
fi
sudo -u patientcare "$APP/.venv/bin/pip" install -q -r "$APP/requirements.txt"
sudo -u patientcare mkdir -p "$APP/data"   # SQLite records + cache live here, on the instance disk
EOF
chmod 755 /usr/local/bin/patientcare-refresh
mkdir -p /opt/patientcare && chown patientcare:patientcare /opt/patientcare

cat > /etc/systemd/system/patientcare.service <<'EOF'
[Unit]
Description=Patient Care Guide Streamlit app
After=network-online.target
Wants=network-online.target

[Service]
User=patientcare
WorkingDirectory=/opt/patientcare
# "+" runs the refresh as root (it writes the key and .env); "-" means a failure doesn't block the app.
ExecStartPre=-+/usr/local/bin/patientcare-refresh
ExecStart=/opt/patientcare/.venv/bin/python -m streamlit run care_app.py --server.port 80 --server.address 0.0.0.0 --server.headless true
# Lets the unprivileged user listen on port 80 without running the app as root.
AmbientCapabilities=CAP_NET_BIND_SERVICE
Restart=always
RestartSec=5
# The first start installs every package, which takes a few minutes.
TimeoutStartSec=900

[Install]
WantedBy=multi-user.target
EOF

# First run of the refresh now (clone + install), then start the service.
/usr/local/bin/patientcare-refresh
systemctl daemon-reload
systemctl enable --now patientcare
