# 🌿 PlantProject: Plant Disease Detection & Care Assistant

## 🔗 Live Demo
**[Visit the Live Application](https://plant-app-129345399998.us-central1.run.app/)**

## 📝 Description
PlantProject is a comprehensive plant health management platform that combines cutting-edge machine learning with agricultural expertise. The application helps gardeners, farmers, and plant enthusiasts identify plant diseases from images and receive tailored care recommendations based on scientifically-validated sources.

### ✨ Key Features
- **AI-Powered Disease Detection**: Upload images of plant leaves to identify 38 different plant diseases across various crops
- **Personalized Care Recommendations**: Get specific treatment and prevention advice for detected diseases
- **Plant Care Chatbot**: Ask questions about plant health, care routines, and disease management
- **User Authentication**: Create an account to track your plant health history
- **History Tracking**: Review past diagnoses and monitor plant health over time
- **Responsive Interface**: User-friendly design that works on mobile and desktop devices

## 🔬 Technical Implementation

### Disease Detection Model
- **Architecture**: ConvNext Tiny (PyTorch)
- **Training**: Transfer learning on 87,000+ plant leaf images
- **Accuracy**: 97.5% on validation data
- **Classes**: 38 disease categories across 14 plant species
- **Deployment**: Optimized for low-latency inference

### RAG-Powered Chatbot
- **Vector Database**: ChromaDB for efficient similarity search
- **Knowledge Base**: 35+ scientific agricultural documents
- **LLM Integration**: DeepSeek API for natural language generation
- **Context Window**: 16K tokens for comprehensive responses
- **Retrieval Method**: Hybrid search with re-ranking

## 👨‍💻 Development Setup

### Prerequisites
- Python 3.8+ installed
- Git for version control
- PyTorch-compatible system
- DeepSeek API key for chatbot functionality

### Quick Start
1. **Clone the repository**
   ```bash
   git clone https://github.com/asent1234/PlantProject.git
   cd PlantProject
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment**
   Create a `.env` file in the project root:
   ```
   DEEPSEEK_API_KEY=your_api_key_here
   ```

5. **Run the application**
   ```bash
   flask --app application run
   # or
   python application.py
   ```

6. **Access the application**
   Open your browser and navigate to [http://127.0.0.1:5000](http://127.0.0.1:5000)

## Deployment to Google Cloud Run

### Prerequisites
1. Google Cloud Account
2. Docker installed on your development machine
3. Docker Hub account (or Google Container Registry)

### Local Testing with Docker
1. Build the Docker image locally:
   ```
   docker build -t plant-app-test .
   ```
2. Run the container locally to test:
   ```
   docker run -p 8080:8080 --env-file .env plant-app-test
   ```
3. Visit http://localhost:8080 to verify the application works properly
4. Test the health check endpoint: http://localhost:8080/health

### Docker Hub Deployment
1. Tag and push your image to Docker Hub:
   ```bash
   docker tag plant-app-test yourusername/plant-app:latest
   docker push yourusername/plant-app:latest
   ```

### Google Cloud Run Deployment
Deploy using one of these methods:

#### Option 1: Docker Desktop Extension (Easiest)
1. Install the Google Cloud Run extension in Docker Desktop
2. Right-click on your plant-app image
3. Select "Deploy to Cloud Run"
4. Configure:
   - Service name: `plant-disease-detection`
   - Region: Choose nearest to you
   - Memory: 2GB (required for ML model)
   - CPU: 1
   - Environment variables: Add `DEEPSEEK_API_KEY`
   - Port: 8080
   - Authentication: Allow unauthenticated invocations

#### Option 2: Google Cloud Console
1. Go to Cloud Run in the console
2. Click "Create Service"
3. Select "Deploy from existing container image"
4. Choose your Docker Hub image
5. Set service name, region, memory (2GB), CPU, and port (8080)
6. Add the `DEEPSEEK_API_KEY` environment variable
7. Set authentication to allow unauthenticated invocations
8. Click "Create"

### Monitoring & Troubleshooting
1. Monitor your application in the Google Cloud Console:
   - View logs in the Cloud Run service details page
   - Monitor memory and CPU usage
   - Check request latency and error rates

2. Common Troubleshooting:
   - If you see "Container failed to start" errors, check your memory allocation (2GB recommended)
   - Verify the `/health` endpoint is responding within timeout limits
   - Check container logs for model loading errors
   - Ensure your DEEPSEEK_API_KEY environment variable is set correctly

## Environment Variables

- `DEEPSEEK_API_KEY`: Required for chatbot functionality
- `PYTORCH_CUDA_ALLOC_CONF`: Optional for fine-tuning PyTorch memory usage

## Tech Stack

- **Framework**: Flask with Gunicorn WSGI server
- **ML**: PyTorch with ConvNext model for plant disease detection
- **Database**: SQLAlchemy with SQLite (upgradable to PostgreSQL)
- **Authentication**: Flask-Login with Bcrypt password hashing
- **Chatbot**: RAG-based implementation using Chroma vector database and DeepSeek API

## Project Structure

- `application.py`: Main Flask application
- `ConvNextAIPredictionModel/`: Disease detection ML model
- `PlantRagChatbot/`: Plant care chatbot with RAG implementation
- `static/`: Frontend assets and user uploads
- `templates/`: HTML templates
- `Dockerfile`: Container configuration for Google Cloud Run deployment

## Notes
- The application uses SQLite by default, but can be configured to use any database supported by SQLAlchemy
- In production, use a persistent database like PostgreSQL
- The `/health` endpoint ensures fast responses during container startup
- Background initialization helps prevent cold start timeouts
- The Docker image is optimized for Google Cloud Run's container requirements
