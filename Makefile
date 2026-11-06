.PHONY: up down test lint eval

up:
	k3d cluster get cc-mini > /dev/null 2>&1 || k3d cluster create cc-mini --servers 1 --agents 1 --api-port 127.0.0.1:6550 --port "8080:80@loadbalancer" --port "8443:443@loadbalancer"
	kubectl create namespace app --dry-run=client -o yaml | kubectl apply -f -
	kubectl create namespace data --dry-run=client -o yaml | kubectl apply -f -
	helm repo add seaweedfs https://seaweedfs.github.io/seaweedfs/ || true
	helm repo update
	helm upgrade --install seaweedfs seaweedfs/seaweedfs --namespace data --create-namespace -f seaweedfs-values.yaml
	kubectl rollout status statefulset/seaweedfs-master -n data --timeout=180s
	kubectl rollout status statefulset/seaweedfs-filer -n data --timeout=180s
	kubectl rollout status statefulset/seaweedfs-volume -n data --timeout=180s
	kubectl rollout status deployment/seaweedfs-s3 -n data --timeout=180s
	python scripts/download_release_model.py
	kubectl port-forward -n data svc/seaweedfs-s3 8333:8333 > port-forward.log 2>&1 & sleep 5; python upload_model.py
	kubectl create secret generic seaweedfs-s3-secret -n app --from-literal=AWS_ACCESS_KEY_ID=ccadmin --from-literal=AWS_SECRET_ACCESS_KEY=ccadmin-secret --dry-run=client -o yaml | kubectl apply -f -
	docker build -t command-parser:s3-v1 .
	k3d image import command-parser:s3-v1 -c cc-mini
	kubectl apply -f k8s/command-parser.yaml
	kubectl apply -f k8s/hpa.yaml
	kubectl apply -f k8s/ingress.yaml
	kubectl rollout status deployment/command-parser -n app --timeout=180s

down:
	k3d cluster delete cc-mini

test:
	python -m pytest tests -q

lint:
	python -m compileall app scripts tests tools

eval:
	python tools/score.py predictions.jsonl data/validation.jsonl --min-validity 95