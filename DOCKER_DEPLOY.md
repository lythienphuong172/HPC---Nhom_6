# Docker Deployment Guide

## Build hệ thống
docker compose build

## Chạy multi-container system
docker compose up -d

## Kiểm tra container
docker ps

## Test API
http://localhost/docs

## Dashboard
http://localhost:8501

## Scale API container
docker compose up --scale api=3 -d

## Stop system
docker compose down

## Docker Swarm Ready
docker swarm init

docker stack deploy -c docker-compose.yml fraud-stack