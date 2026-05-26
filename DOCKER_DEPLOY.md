# Docker Deployment Guide (Swarm Mode)

## 1. Chuẩn bị: Build & Push Image
Do Swarm không tự động build image từ mã nguồn ở các node, bạn cần build và push image lên Docker Registry (ví dụ Docker Hub) trước.

*Lưu ý: Thay `your-dockerhub-username` bằng username thực tế của bạn trên Docker Hub.*

```bash
# Build images
docker build -t your-dockerhub-username/fraud-api:latest -f docker/api.Dockerfile .
docker build -t your-dockerhub-username/fraud-dashboard:latest -f docker/dashboard.Dockerfile .

# Push images lên Registry
docker push your-dockerhub-username/fraud-api:latest
docker push your-dockerhub-username/fraud-dashboard:latest
```

## 2. Khởi tạo Docker Swarm (Chỉ chạy 1 lần ở Node Manager)
Nếu máy của bạn chưa bật Swarm mode, hãy khởi tạo nó:
```bash
docker swarm init
```
*(Để thêm các máy khác vào cụm, hãy copy lệnh `docker swarm join-token worker` mà terminal in ra sau lệnh trên)*

## 3. Triển khai Hệ thống (Deploy Stack)
Chạy lệnh sau để deploy toàn bộ các dịch vụ lên cụm Swarm:
```bash
docker stack deploy -c docker-compose.yml fraud-stack
```

## 4. Kiểm tra Trạng thái
Kiểm tra các service đang chạy và số lượng bản sao (replicas):
```bash
docker service ls
```
Xem chi tiết các container của stack đã được phân bổ trên các node nào:
```bash
docker stack ps fraud-stack
```

## 5. Truy cập Ứng dụng
Bạn có thể truy cập thông qua địa chỉ IP của **BẤT KỲ NODE NÀO** trong cụm Swarm (nhờ cơ chế Ingress Routing Mesh).
- **Test API:** `http://<IP-Node>/docs` (Mặc định ở local là http://localhost/docs)
- **Dashboard:** `http://<IP-Node>:8501` (Mặc định ở local là http://localhost:8501)

## 6. Mở rộng (Scale) API
Swarm cho phép scale nóng mà không cần dừng hệ thống. Ví dụ tăng API lên 5 container:
```bash
docker service scale fraud-stack_api=5
```

## 7. Dừng Hệ thống
Gỡ bỏ hoàn toàn stack:
```bash
docker stack rm fraud-stack
```