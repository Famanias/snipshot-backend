import base64
import requests
import json

API_URL = "http://127.0.0.1:8000/translate-image"

IMAGE_PATH = "14.jpg"

TARGET_LANG = "en"


def image_to_base64(path):
    with open(path, "rb") as image_file:
        return base64.b64encode(
            image_file.read()
        ).decode("utf-8")


def main():
    try:
        image_base64 = image_to_base64(IMAGE_PATH)

        payload = {
            "image_base64": image_base64,
            "target_lang": TARGET_LANG
        }

        response = requests.post(
            API_URL,
            json=payload,
            timeout=120
        )

        print(f"\nStatus Code: {response.status_code}\n")

        try:
            data = response.json()

            print("\n========== RESULT ==========\n")

            print(f"Detected Language:\n{data.get('detected_language', 'unknown')}\n")

            print("----- Extracted Text -----\n")
            print(data.get("extracted_text", ""))

            print("\n----- Translation -----\n")
            print(data.get("translated_text", ""))

            print("\n----- Summary -----\n")
            print(data.get("summary", ""))

            print("\n============================\n")

        except Exception:
            print(response.text)

    except Exception as e:
        print(f"Test failed: {e}")


if __name__ == "__main__":
    main()