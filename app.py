from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import base64
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input
import numpy as np

app = Flask(__name__)
CORS(app, resources={r"/upload": {"origins": "http://skripsi-bagja.ikulatluh.cloud"}})
@app.route('/upload', methods=['POST'])
@cross_origin()  # Add this decorator to allow cross-origin requests
def upload_image():
    data = request.json

    if 'encryptedImage' not in data:
        return jsonify({'message': 'No image data'}), 400

    encrypted_image = data['encryptedImage']
    # Load the trained model
    model = load_model('model_final.h5')  # Ganti 'model.h5' dengan nama model Anda

    # Define class labels
    class_labels = ['Berat', 'Normal', 'Ringan']  # Ganti dengan label kelas Anda

    try:
        decoded_image = base64.b64decode(encrypted_image)

        # Create a PIL Image object from the decoded data and convert to RGB
        img = image.load_img(BytesIO(decoded_image), target_size=(299, 299))

        # Simpan gambar ke dalam file
        image_path = 'uploaded_image.png'
        img.save(image_path)

       

        #        Load and preprocess the uploaded image
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        preprocessed_img = img_array / 255.0
        # Resize gambar ke ukuran yang diharapkan oleh model InceptionV3
        # Make predictions
        prediction = model.predict(preprocessed_img)
        predicted_class = np.argmax(prediction)
        confidence = prediction[0][predicted_class]

# Get the class label
        class_label = class_labels[predicted_class]

# Tambahkan kelas dan persentase ke gambar
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype("Arial", 16) # Ganti font dan ukuran sesuai kebutuhan
        text = f'Kelas: {class_label}, Persentase: {confidence:.2f}'
        draw.text((10, 10), text, fill=(255, 255, 255), font=font)


        # Simpan gambar ke dalam file
        save_path = "output_image.png"# Ganti dengan path dan nama file yang Anda inginkan

        img.save(save_path)

        with open("output_image.png", "rb") as image_file:
                # Membaca isi file sebagai byte
            image_bytes = image_file.read()

# Mendekode byte gambar menjadi base64
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")

        # Menampilkan kelas dan persentase
        print(f'Kelas: {class_label}, Persentase: {confidence:.2f}')

        return jsonify({
            'message': image_base64,
            "gambar": "sukses"
            }), 200
    except Exception as e:
        error_message = str(e)
        print(f'Error: {error_message}')
        return jsonify({'message': f'Error decoding or processing the image: {error_message}'}), 400


if __name__ == '__main__':
    app.run(debug=True)
