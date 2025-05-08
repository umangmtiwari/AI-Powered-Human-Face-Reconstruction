import logging
import torchvision.transforms as transforms
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackContext
from my_code import regenerate1, regenerate2
from attention_mechanism import regenerate3
from PIL import Image, ImageDraw, ImageFont
import random
import os
import matplotlib.pyplot as plt
from torchvision.utils import save_image

# Set up logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the token of your bot (replace with your bot token)
TOKEN = "paste your token here of your telegram bot"

# Transformation for the uploaded images (same as your code)
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# Function to start the bot
async def start(update: Update, context: CallbackContext):
    await update.message.reply_text("👋 Hello! I'm here to help you regenerate your images. 🖼️\n\n"
                                    "Send me a file, and I'll process it for you! Or just ask me to send a file if you need one. 📤😊")

# Function to damage the image by adding a black occlusion (similar to the one in my_code.py)
def damage_image(image, mask_size=32):
    damaged = image.clone()
    _, h, w = image.shape
    x = random.randint(0, w - mask_size)
    y = random.randint(0, h - mask_size)
    damaged[:, y:y+mask_size, x:x+mask_size] = 0  # Black patch
    return damaged

# Function to combine 3 images vertically with text in the gap below each and save as result.png
def combine_images_and_save_with_text():
    # Load the images
    image_1 = Image.open(os.path.join("restored", "1.png"))
    image_2 = Image.open(os.path.join("restored", "2.png"))
    image_3 = Image.open(os.path.join("restored", "3.png"))
    
    # Resize images to the same width (optional, if they have different widths)
    max_width = max(image_1.width, image_2.width, image_3.width)
    image_1 = image_1.resize((max_width, int(image_1.height * max_width / image_1.width)))
    image_2 = image_2.resize((max_width, int(image_2.height * max_width / image_2.width)))
    image_3 = image_3.resize((max_width, int(image_3.height * max_width / image_3.width)))

    # Define the gap space between the image and the text
    gap_height = 30  # Gap between the image and text
    image_gap = 20    # Gap between the text and the next image
    
    # Add a label to each image (text will be below the image in the gap)
    font = ImageFont.load_default()  # Default font
    
    def add_text_below_image(image, text, gap_height):
        # Create a drawing context
        draw = ImageDraw.Draw(image)
        
        # Calculate the text size and position using textbbox
        bbox = draw.textbbox((0, 0), text, font=font)  # This gets the bounding box (x0, y0, x1, y1)
        text_width = bbox[2] - bbox[0]  # Width of the text
        text_height = bbox[3] - bbox[1]  # Height of the text
        text_position = ((image.width - text_width) // 2, image.height + gap_height)

        # Create a new image with extra space for the text
        new_image = Image.new("RGB", (image.width, image.height + gap_height + text_height), (255, 255, 255))
        new_image.paste(image, (0, 0))

        # Add text below the image
        draw = ImageDraw.Draw(new_image)
        draw.text(text_position, text, font=font, fill="black")
        
        return new_image

    # Add text below each image
    image_1 = add_text_below_image(image_1, "🖼️ Restored Image 1", gap_height)
    image_2 = add_text_below_image(image_2, "🖼️ Restored Image 2", gap_height)
    image_3 = add_text_below_image(image_3, "🖼️ Restored Image 3", gap_height)

    # Create a new image with the total height (including gaps and text)
    total_height = image_1.height + image_gap + image_2.height + image_gap + image_3.height
    combined_image = Image.new("RGB", (max_width, total_height), (255, 255, 255))

    # Paste the images one below the other with the image_gap between them
    combined_image.paste(image_1, (0, 0))
    combined_image.paste(image_2, (0, image_1.height + image_gap))
    combined_image.paste(image_3, (0, image_1.height + image_gap + image_2.height + image_gap))

    # Save the combined image
    result_path = os.path.join("restored", "result.png")
    combined_image.save(result_path)

    logger.info(f"Combined image saved as {result_path}")

# Updated handle_file function
async def handle_file(update: Update, context: CallbackContext):
    file = update.message.document
    save_path_1 = os.path.join("restored", "1.png")
    save_path_2 = os.path.join("restored", "2.png")
    save_path_3 = os.path.join("restored", "3.png")
    
    if file:
        logger.info(f"Received file: {file.file_name}")  # Log the file name

        # Download the file using get_file() and save it to a local path
        downloaded_file = await file.get_file()  # Await the asynchronous method
        
        # Download the file to the specified path
        file_path = "temp_image.jpg"  # Temporary path to save the file
        await downloaded_file.download_to_drive(file_path)  # Use download_to_drive instead of download

        # Open the image using PIL
        test_image = Image.open(file_path).convert("RGB")  # Open the image directly without saving
        test_image = transform(test_image)  # Apply the transformations

        # Damage the image once and pass it to both regeneration functions
        damaged_image = damage_image(test_image)  # Apply the damage to the image

        # Regenerate the images using the functions
        restored_image_1, evaluation_metrics_1 = regenerate1(damaged_image)
        restored_image_2, evaluation_metrics_2 = regenerate2(damaged_image)
        restored_image_3, evaluation_metrics_3 = regenerate3(damaged_image)
        print("Evaluation Metrics 1: ", evaluation_metrics_1)
        print("Evaluation Metrics 2: ", evaluation_metrics_2)
        print("Evaluation Metrics 3: ", evaluation_metrics_3)

        # Saving the images
        plt.imsave(save_path_1, restored_image_1)
        save_image(restored_image_2, save_path_2)
        plt.imsave(save_path_3, restored_image_3)

        # Saving the combined image
        combine_images_and_save_with_text()
        
        # Log the success message
        logger.info("Restored image saved successfully.")
        
        # Send the result image (combined image with text) to the user
        result_path = os.path.join("restored", "result.png")
        with open(result_path, 'rb') as result_file:
            await context.bot.send_document(
                chat_id=update.message.chat_id, 
                document=result_file, 
                caption=f"Here are your restored images! ⬆️\n\n"
                        f"✨ Restoration Results ✨\n\n"
                        f"🔹 Evaluation Metrics 1:\nMSE Loss: {evaluation_metrics_1[0]}\nPSNR Value: {evaluation_metrics_1[1]}\nSSIM Value: {evaluation_metrics_1[2]} 📊\n\n"
                        f"🔹 Evaluation Metrics 2:\nMSE Loss: {evaluation_metrics_2[0]}\nPSNR Value: {evaluation_metrics_2[1]}\nSSIM Value: {evaluation_metrics_2[2]} 📊\n\n"
                        f"🔹 Evaluation Metrics 3:\nMSE Loss: {evaluation_metrics_3[0]}\nPSNR Value: {evaluation_metrics_3[1]}\nSSIM Value: {evaluation_metrics_3[2]} 📊\n\n"
                        f"📥 Checkout the result and look out the improvements! 📊"
            )
        
    else:
        logger.info("No file received.")
        await update.message.reply_text("⚠️ Oops! It seems like there was no file received. Please try again. 🙏")

# Function to send a file to the user
async def send_file(update: Update, context: CallbackContext):
    # Specify the file path you want to send
    file_path = "C:/Users/umang/Downloads/data/celeba/img_align_celeba/011704.jpg"
    
    try:
        # Send the file
        with open(file_path, 'rb') as file:
            await context.bot.send_document(chat_id=update.message.chat_id, document=file)
        await update.message.reply_text("📂 Here's your file! Enjoy! 🎉")
    except Exception as e:
        await update.message.reply_text(f"❌ Failed to send file. Error: {e}")

# Function to handle unknown commands
async def unknown(update: Update, context: CallbackContext):
    await update.message.reply_text("🤖 Sorry, I didn't understand that command. Please try again! 🙏")

# Main function to set up the bot
async def main():
    # Create an Application object and pass it your bot's token
    application = Application.builder().token(TOKEN).build()

    # Add handlers for commands and messages
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.Document.ALL, handle_file))
    application.add_handler(CommandHandler("sendfile", send_file))

    # Add a handler for unknown commands
    application.add_handler(MessageHandler(filters.COMMAND, unknown))

    # Start polling for updates
    await application.run_polling()

if __name__ == '__main__':
    # Check if we are already inside an event loop (like in Jupyter notebook or certain IDEs)
    try:
        import nest_asyncio
        nest_asyncio.apply()  # Apply a patch to allow running the asyncio event loop
    except ImportError:
        pass

    # Use await to run the bot in environments with an existing event loop
    import asyncio
    asyncio.run(main())  # This works outside environments with existing loops
