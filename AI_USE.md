# AI Usage Disclosure

## Purpose

Generative AI tools were used as development assistance during this project. AI assistance was used for explanation, debugging, code suggestions, documentation refinement, and troubleshooting.

## Areas Where AI Assistance Was Used

AI assistance was used for:

- Understanding and troubleshooting Python, FastAPI, Docker, Kubernetes, and GitHub Actions configuration.
- Debugging errors encountered during local development and Kubernetes deployment.
- Reviewing Kubernetes manifests, health/readiness probes, HPA configuration, and deployment configuration.
- Suggesting improvements to test cases and CI workflow structure.
- Explaining model training, evaluation, inference, and deployment concepts.
- Assisting with README/report wording and organization.
- Helping interpret command output and diagnose deployment or CI failures.

## Human Work and Verification

The project implementation, configuration, experiments, model training/evaluation, deployment, testing, and evidence collection were performed and verified by the student.

AI-generated suggestions were not treated as authoritative. Commands, code, configurations, model outputs, Kubernetes behavior, CI results, and evaluation results were tested in the actual project environment before being accepted.

In particular, the following were independently executed and verified:

- Student-model training and evaluation.
- FastAPI application behavior.
- Docker image construction.
- SeaweedFS object-storage integration.
- Kubernetes deployment and service configuration.
- Health and readiness checks.
- HPA configuration.
- Rolling update and rollback behavior.
- Pod recovery behavior.
- Cold-start measurements.
- Load-balancing measurements.
- Network NACL simulation.
- Unit tests and linting.
- GitHub Actions CI pipeline.
- Model evaluation quality gate.
- Deliberate CI failure using an invalid model asset followed by successful restoration.

## Academic Integrity

AI assistance was used as a development aid and not as a substitute for understanding or verification. The student reviewed, adapted, executed, and validated the resulting implementation and is responsible for the submitted work and reported results.