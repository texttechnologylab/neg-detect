from typing import Literal
import os
import requests


BP = os.path.realpath(os.path.join(os.path.realpath(__file__), "../../.."))


def download_udpipe(lang: Literal["ar", "hi"]):
    # URL of the file
    if lang == "ar":
        url = "https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-3131/arabic-padt-ud-2.5-191206.udpipe?sequence=8"

        # Output file name
        output_file = f"{BP}/data/arabic-padt-ud-2.5-191206.udpipe"
    elif lang == "hi":
        url = "https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-3131/hindi-hdtb-ud-2.5-191206.udpipe?sequence=42&isAllowed=y"

        # Output file name
        output_file = f"{BP}/data/hindi-hdtb-ud-2.5-191206.udpipe"
    else:
        raise Exception(f"Unknown language: {lang}")

    # Set headers to mimic a browser request
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    try:
        # Send GET request
        response = requests.get(url, headers=headers, stream=True)

        # Check if the request was successful
        response.raise_for_status()

        # Save the file
        with open(output_file, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        print(f"File downloaded successfully as {output_file}")

    except requests.exceptions.RequestException as e:
        print(f"Error downloading file: {e}")