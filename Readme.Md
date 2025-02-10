# UPYOG Chatbot Deployment Guide

## Overview
The UPYOG Chatbot is a Flask-based chatbot that provides responses based on predefined FAQs using sentence embeddings and FAISS for efficient retrieval. This guide covers setting up, deploying, and securing the chatbot on an AWS EC2 instance with Docker and Nginx.

## Prerequisites
Before proceeding, ensure you have the following:
- An AWS EC2 instance running Ubuntu 24.04
- Docker and Docker Compose installed
- Nginx installed for reverse proxy
- Domain name (e.g., `upyogchatbot.niua.in`) mapped to the Elastic IP
- SSL certificate setup via Let's Encrypt

## Project Setup

### Clone the Repository
```sh
cd ~
git clone <repo-url>
cd Deployment_finalize
```

### Create and Activate Virtual Environment (Optional)
```sh
python3 -m venv env
source env/bin/activate
```

### Install Dependencies
```sh
pip install -r requirements.txt
```

## Docker Deployment

### Build and Run the Docker Container
```sh
sudo docker build -t flaskapp .
sudo docker run -d --restart always -p 8000:8000 flaskapp
```

### Check Running Containers
```sh
sudo docker ps
```

## Nginx Configuration

### Install Nginx
```sh
sudo apt update && sudo apt install nginx -y
```

### Configure Nginx for Reverse Proxy
Edit the default Nginx configuration:
```sh
sudo nano /etc/nginx/sites-available/default
```
Replace the content with:
```nginx
server {
    listen 80;
    server_name upyogchatbot.niua.in;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```
Save and exit (CTRL+X, then Y, then ENTER).

### Restart Nginx
```sh
sudo systemctl restart nginx
```

## Enable SSL with Certbot

### Install Certbot
```sh
sudo apt install certbot python3-certbot-nginx -y
```

### Generate SSL Certificate
```sh
sudo certbot --nginx -d upyogchatbot.niua.in
```

Follow the prompts to complete the setup. Certbot will automatically configure Nginx for HTTPS.

### Verify SSL
```sh
curl -I https://upyogchatbot.niua.in
```

## Auto-Renew SSL
Certbot renews certificates automatically, but you can manually test it:
```sh
sudo certbot renew --dry-run
```

## Updating and Restarting the Chatbot

### Rebuild Docker Container After Code Changes
```sh
sudo docker stop flaskapp
sudo docker rm flaskapp
sudo docker build -t flaskapp .
sudo docker run -d --restart always -p 8000:8000 flaskapp
```

### Check Logs
```sh
sudo docker logs -f <container_id>
```

## Updated Requirements
The chatbot requires the following Python packages (as per `requirements.txt`):

```
blinker==1.9.0
certifi==2024.12.14
chardet==3.0.4
charset-normalizer==3.4.0
click==8.1.7
faiss-cpu==1.9.0.post1
filelock==3.16.1
Flask==3.1.0
Flask-Cors==5.0.0
fsspec==2024.10.0
googletrans==4.0.0rc1
h11==0.9.0
h2==3.2.0
hpack==3.0.0
hstspreload==2024.12.1
httpcore==0.9.1
httpx==0.13.3
huggingface-hub==0.26.5
hyperframe==5.2.0
idna==2.10
importlib_metadata==8.5.0
itsdangerous==2.2.0
Jinja2==3.1.4
joblib==1.4.2
langdetect==1.0.9
MarkupSafe==3.0.2
mpmath==1.3.0
networkx==3.2.1
numpy==2.0.2
packaging==24.2
pandas==2.2.3
pillow==11.0.0
python-dateutil==2.9.0.post0
pytz==2024.2
PyYAML==6.0.2
regex==2024.11.6
requests==2.32.3
rfc3986==1.5.0
safetensors==0.4.5
scikit-learn==1.6.0
scipy==1.13.1
sentence-transformers==3.3.1
six==1.17.0
sniffio==1.3.1
sympy==1.13.1
threadpoolctl==3.5.0
tokenizers==0.21.0
torch==2.5.1
tqdm==4.67.1
transformers==4.47.0
typing_extensions==4.12.2
tzdata==2024.2
urllib3==2.2.3
Werkzeug==3.1.3
zipp==3.21.0
```

## Troubleshooting

### Check Running Processes
```sh
sudo systemctl status nginx
sudo docker ps
```

### Restart Services
```sh
sudo systemctl restart nginx
sudo docker restart <container_id>
```

### Check Logs for Errors
```sh
sudo journalctl -u nginx --no-pager | tail -n 20
sudo docker logs <container_id>
```

## Conclusion
This guide ensures the chatbot is properly deployed and secured with SSL. The setup enables automated restarts and SSL renewal for a stable and secure deployment.
