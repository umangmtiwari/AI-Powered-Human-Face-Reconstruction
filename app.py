from flask import Flask, render_template, request, redirect, flash
from PIL import Image
import torchvision.transforms as transforms
from my_code import regenerate1, regenerate2  # Assuming these functions are imported from your my_code.py
from attention_mechanism import regenerate3
import random
from io import BytesIO
import base64
import uuid
import boto3
from flask import Flask, request, jsonify, render_template
from langchain.llms.bedrock import Bedrock
from threading import Thread
import logging

app = Flask(__name__)

app.secret_key = 'your_secret_key_here'  # Change this to something more secure
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Transformation for the uploaded images (same as your code)
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Bedrock Client for Claude
bedrock = boto3.client(service_name="bedrock-runtime")

# In-memory task store to manage tasks
tasks = {}

# Set up Claude LLM (Anthropic)
def get_claude_llm():
    llm = Bedrock(model_id="anthropic.claude-v2", client=bedrock)
    return llm

# Define the new doctor-like prompt template for Claude
prompt_template = """
Human: As a virtual doctor, please answer the following question with empathy, professionalism, and expertise. Offer personalized advice and explain things clearly.
Question: {question}
Assistant:
"""

# Function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Function to damage the image by adding a black occlusion (similar to the one in my_code.py)
def damage_image(image, mask_size=32):
    damaged = image.clone()
    _, h, w = image.shape
    x = random.randint(0, w - mask_size)
    y = random.randint(0, h - mask_size)
    damaged[:, y:y+mask_size, x:x+mask_size] = 0  # Black patch
    return damaged

# regenerate route
@app.route('/regenerate', methods=['GET', 'POST'])
def regenerate_image():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'image' not in request.files:
            flash('No file part')
            return redirect(request.url)

        file = request.files['image']

        # If no file is selected
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        # If the file is allowed, process it without saving
        if file and allowed_file(file.filename):

            # No need to save to disk, we can directly open and process the image
            test_image = Image.open(file).convert("RGB")  # Open the image directly without saving
            test_image = transform(test_image)  # Apply the transformations

            # Damage the image once and pass it to both regeneration functions
            damaged_image = damage_image(test_image)  # Apply the damage to the image

            # Regenerate the images using the functions
            restored_image_1, evaluation_metrics_1 = regenerate1(damaged_image)
            restored_image_2, evaluation_metrics_2 = regenerate2(damaged_image)
            restored_image_3, evaluation_metrics_3 = regenerate3(damaged_image)

            # Convert images to BytesIO to send directly without saving to disk
            img_byte_arr1 = BytesIO()
            img_byte_arr2 = BytesIO()
            img_byte_arr3 = BytesIO()
            damaged_byte_arr = BytesIO()  # Create a BytesIO object for the damaged image

            # Save the images to the BytesIO object
            restored_image_1 = transforms.ToPILImage()(restored_image_1)  # Convert tensor to PIL image
            restored_image_2 = transforms.ToPILImage()(restored_image_2)
            restored_image_3 = transforms.ToPILImage()(restored_image_3)
            damaged_image = transforms.ToPILImage()(damaged_image)  # Convert tensor to PIL image for damaged image

            restored_image_1.save(img_byte_arr1, format='PNG')
            restored_image_2.save(img_byte_arr2, format='PNG')
            restored_image_3.save(img_byte_arr3, format='PNG')
            damaged_image.save(damaged_byte_arr, format='PNG')  # Save the damaged image to the BytesIO object

            # Make sure to seek back to the beginning of the BytesIO object
            img_byte_arr1.seek(0)
            img_byte_arr2.seek(0)
            img_byte_arr3.seek(0)
            damaged_byte_arr.seek(0)

            # Convert the BytesIO objects to base64 strings
            img_base64_1 = base64.b64encode(img_byte_arr1.getvalue()).decode('utf-8')
            img_base64_2 = base64.b64encode(img_byte_arr2.getvalue()).decode('utf-8')
            img_base64_3 = base64.b64encode(img_byte_arr3.getvalue()).decode('utf-8')
            damaged_base64 = base64.b64encode(damaged_byte_arr.getvalue()).decode('utf-8')  # Base64 for damaged image
            
            # Format evaluation metrics to 4 decimal places
            formatted_eval_metrics_1 = [f"{metric:.4f}" for metric in evaluation_metrics_1]
            formatted_eval_metrics_2 = [f"{metric:.4f}" for metric in evaluation_metrics_2]
            formatted_eval_metrics_3 = [f"{metric:.4f}" for metric in evaluation_metrics_3]

            # Return everything to the same template
            return render_template(
                'regenerate.html',
                damaged_image_url=damaged_base64,  # Send damaged image
                image1_url=img_base64_1,
                image2_url=img_base64_2,
                image3_url=img_base64_3,
                eval_metrics_1=formatted_eval_metrics_1,  # Pass formatted metrics for regeneration 1
                eval_metrics_2=formatted_eval_metrics_2,  # Pass formatted metrics for regeneration 2
                eval_metrics_3=formatted_eval_metrics_3   # Pass formatted metrics for regeneration 3
            )

        else:
            flash('Allowed image types are png, jpg, jpeg')
            return redirect(request.url)

    return render_template('regenerate.html')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/results')
def results():
    return render_template('results.html')

@app.route('/project-flow')
def project_flow():
    return render_template('project-flow.html')

@app.route('/ask-doubts')
def ask_doubts():
    return render_template('ask-doubts.html')

# Function to generate the prompt
def generate_prompt(question):
    return prompt_template.format(question=question)


# Function to get the response from Claude
def get_response_from_claude(llm, question):
    prompt = generate_prompt(question)
    
    # Generate the response from Claude using the formatted prompt
    response = llm.generate([prompt])  # response will be an LLMResult object
    
    # Extract the generated response from the 'generations' attribute
    if response.generations and len(response.generations) > 0:
        answer = response.generations[0][0].text  # The first generation's first choice text
        return answer
    else:
        return "Sorry, I couldn't generate an answer."

# Function to process tasks (background job to handle the question)
def process_task(task_id, question):
    try:
        logger.info(f"Processing task ID: {task_id} with question: {question}")

        # Get Claude model
        llm = get_claude_llm()

        # Generate the response from Claude
        answer = get_response_from_claude(llm, question)
        logger.info(f"Generated answer: {answer}")

        # Store the result in the task store
        tasks[task_id] = {"status": "completed", "result": answer}
    except Exception as e:
        logger.error(f"Error processing task ID: {task_id}", exc_info=True)
        tasks[task_id] = {"status": "error", "result": str(e)}

# API route to handle user questions
@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.json
    user_question = data.get('question')

    # Validate the question input
    if not user_question:
        return jsonify({"status": "error", "result": "Question is required"}), 400

    # Generate a unique task ID for the question
    task_id = str(uuid.uuid4())
    tasks[task_id] = {"status": "pending", "result": None}

    # Process the question asynchronously using a background thread
    thread = Thread(target=process_task, args=(task_id, user_question))
    thread.start()

    # Return the task ID to the user so they can check the result later
    return jsonify({"task_id": task_id})

# API route to get the result of a specific task
@app.route('/result/<task_id>', methods=['GET'])
def get_result(task_id):
    task = tasks.get(task_id, None)
    if not task:
        return jsonify({"status": "error", "result": "Invalid task ID"}), 404

    return jsonify(task)

if __name__ == '__main__':
    app.run(debug=True)