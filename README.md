# Lab 5: Containerizing ML Models with Docker

## Overview

In this lab, you will containerize a machine learning training pipeline and inference server using Docker. You will train a Wine classifier, serve it via Flask, and orchestrate both containers with Docker Compose. A key focus of this lab is **Docker volume management** — you will use both named volumes and bind mounts to share data between containers and persist artifacts on the host.

### Deliverables

- [ ] **Deliverable 1**: Show that the training script ran in a container and saved the model to a shared volume. Explain why Docker is useful for reproducibility and portability in ML training.
- [ ] **Deliverable 2**: Serve predictions from the inference container on localhost port 8081 and show the TA `./logs/predictions.log` on your host. Explain what the Dockerfile does for the inference service.
- [ ] **Deliverable 3**: Attempt to call the inference service health endpoint before and after removing the Compose named volume. Explain the difference between named volumes and bind mounts in Docker.

## Step 0: Setup Docker

Install Docker using the [instructions for your operating system](https://docs.docker.com/get-started/get-docker/), then verify your installation:

```bash
docker run hello-world
```

## Step 1: Containerize the Training Pipeline

Create a Docker container for the training code. When launched, the container should train a Wine classifier and save the model file to a shared volume. You can use the partially completed Dockerfile and code in `docker/training/`.

### 1a. Complete `train.py`

- Create a `RandomForestClassifier` and train it on the Wine dataset
- Save the trained model using `joblib.dump()` to `/app/models/wine_model.pkl`

### 1b. Complete the Training Dockerfile

Open `docker/training/Dockerfile`. It is nearly complete. Fill in the TODO to set the command that runs the training script.

### 1c. Build and Run with a Named Volume

Build the training image and run it, mounting a **named volume** for model storage:

```bash
docker build -t mlip-training -f docker/training/Dockerfile .
docker run --rm -v wine_model_storage:/app/models mlip-training
```

You should see output showing the test accuracy and a message that the model was saved.

**What is a named volume?** When you use `-v wine_model_storage:/app/models`, Docker creates a named volume called `wine_model_storage` that is managed by Docker. The data in this volume persists even after the container exits.

**Think about it:** Was the model trained during `docker build` or `docker run`? What in your Dockerfile and terminal output tells you?

## Step 2: Containerize the Inference Server

Create a Docker container that loads the trained model from the shared volume and serves predictions via a Flask API. The server also logs predictions to a bind-mounted directory so you can inspect them from the host.

### 2a. Complete `server.py`

In `docker/inference/server.py`, load the trained model from the shared volume, extract features from the incoming JSON request, run inference, and log each prediction to a host-mounted log file (`/app/logs/predictions.log`).

The server includes a `/health` endpoint that reports whether the model file exists and whether it was loaded — this is useful for debugging volume issues.

### 2b. Create the Inference Dockerfile

Create a new file `docker/inference/Dockerfile` from scratch. It should:

- Use `python:3.11-slim` as the base image
- Set the working directory to `/app`
- Copy and install dependencies from a requirements file
- Copy `server.py` to the working directory
- Expose the port the Flask app listens on inside the container (see `server.py`)
- Set the command to run `server.py`

**Hint:** Look at the training Dockerfile for reference. The build context is the project root, so paths should be `docker/inference/...`.

You will also need to create `docker/inference/requirements.txt` with the necessary packages (Flask, scikit-learn, joblib, numpy).

### 2c. Build and Run with Both Volume Types

Create a local directory for logs, then run the inference container with both a named volume (for the model) and a bind mount (for logs):

```bash
mkdir -p ./logs
docker build -t mlip-inference -f docker/inference/Dockerfile .
docker run --rm -p 127.0.0.1:8081:8080 \
  -v wine_model_storage:/app/models \
  -v $(pwd)/logs:/app/logs \
  mlip-inference
```

Notice the two `-v` flags:

- `wine_model_storage:/app/models` — **named volume** (Docker-managed, shared with training)
- `$(pwd)/logs:/app/logs` — **bind mount** (maps your local `./logs/` directory into the container)

### 2d. Test the Inference Server

Check the health endpoint:

```bash
curl http://localhost:8081/health
```

Send a prediction request (13 Wine features):

```bash
curl -X POST http://localhost:8081/predict \
  -H 'Content-Type: application/json' \
  -d '{"input": [13.2, 1.78, 2.14, 11.2, 100, 2.65, 2.76, 0.26, 1.28, 4.38, 1.05, 3.40, 1050]}'
```

Test error handling with a bad request:

```bash
curl -X POST http://localhost:8081/predict \
  -H 'Content-Type: application/json' \
  -d '{"bad_key": [1,2,3]}'
```

After sending predictions, check your local `./logs/` directory — you should see a `predictions.log` file with timestamped entries. This is the bind mount in action: the container writes to `/app/logs/` and the file appears on your host filesystem.

**Think about it:** If you run the training container again while the inference container stays up, will the next prediction use the newly saved model? What in `server.py` determines this?

Stop the manually run inference container before moving to Step 3.

## Step 3: Docker Compose

Docker Compose allows you to define and manage multi-container applications without long command-line parameters. Complete the `docker-compose.yml` file to set up both services.

You need to fill in:

- Build context and Dockerfile path for each service
- **Named volume** `wine_model_storage` mounted to `/app/models` on both services (for sharing the model)
- **Bind mount** `./logs` mapped to `/app/logs` on the inference service (for prediction logs)
- Port mapping for the inference service, accessible only from localhost
- Named volume definition in the `volumes:` section at the bottom

Then run:

```bash
docker compose up --build
```

After both services start, test with the same curl commands from Step 2d. Verify that:

- The prediction endpoint returns a valid wine class
- The `./logs/predictions.log` file on your host is being written to

**Think about it:** During startup, did you see any errors? If not, could an error appear on another run? What might cause it, and how would you fix it? **Hint:** Look at the current `depends_on` setting and [Docker's Compose startup order documentation](https://docs.docker.com/compose/how-tos/startup-order/).

Shut it down:

```bash
docker compose down
```

## Step 4: Volume Lifecycle

This step demonstrates the differences in how Docker manages data persistence and host-container sharing.

### 4a. Inspect the Named Volume

Check where Docker reports storing the named volume:

```bash
docker volume ls
docker volume inspect wine_model_storage
```

Observe the `Mountpoint` field. On Docker Desktop, this path may be inside its Linux VM.

### 4b. Persistence Verification

Verify that the trained model persists across training container destruction. Start both containers to run training and inference:

```bash
docker compose up --build
```

Confirm health:

```bash
curl http://localhost:8081/health
```

Stop and remove containers while retaining the named volume:

```bash
docker compose down
```

Restart only the inference container:

```bash
docker compose up inference --no-deps
```

Run the health check again and confirm it still returns healthy, demonstrating that the model was successfully persisted in the named volume.

### 4c. Removing Volumes

To fully reset the environment and delete the model:

```bash
docker compose down -v
```

The `-v` flag removes the Compose named volume. Now run only the inference container again, try the health check, and explain what you observe to the TA.

**Think about it:** If you give a teammate only your inference image, can they use it to serve predictions on their machine? What else would they need?

To finish, press Ctrl+C to stop inference. If Compose hangs after showing `Stopped`, press Ctrl+C again; you may see `Error while Killing` even though the container has stopped.

## Optional Cleanup

After TA checkoff, stop and remove any remaining Compose containers and the named volume:

```bash
docker compose down -v
```

You may also want to remove the images and the standalone named volume you created earlier in the lab. Use `docker image ls` and `docker volume ls` to identify them. Only remove resources you are sure you created for this lab and no longer need, using `docker image rm IMAGE_NAME` or `docker volume rm VOLUME_NAME`.

## Additional Resources

- [Docker Curriculum](https://docker-curriculum.com/)
- [Docker volumes](https://docs.docker.com/engine/storage/volumes/)
- [Docker bind mounts](https://docs.docker.com/engine/storage/bind-mounts/)

## Troubleshooting

If you encounter issues:

- Check Docker daemon status
- Verify port availability (is port 8081 already in use?)
- Review service logs with `docker compose logs`
- If inference fails during startup, compare the training and inference logs
- Use `docker compose exec` to inspect a running container's file system
- If the model file is missing, compare the volume names in your commands or Compose file and check them with `docker volume ls`
- If logs are not appearing on the host, verify your bind mount path
