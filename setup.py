import requests
import os
import zipfile
import tarfile

def download_and_extract_zip(url, target_dir):
    # Effectuer une requête GET sur l'URL du fichier
    response = requests.get(url, allow_redirects=True)

    # Vérifier si la requête a réussi (code de statut HTTP 200)
    if response.status_code == 200:
        # Créer un chemin de fichier temporaire pour enregistrer le fichier
        temp_file_path = "./temp"

        # Enregistrer le contenu de la réponse dans un fichier temporaire
        with open(temp_file_path, 'wb') as f:
            f.write(response.content)

        # Extraire le contenu du fichier temporaire
        with zipfile.ZipFile(temp_file_path, 'r') as zip_ref:
            zip_ref.extractall(target_dir)

        # Supprimer le fichier temporaire
        os.remove(temp_file_path)

        print(f"Le fichier a été téléchargé et son contenu a été extrait avec succès dans '{target_dir}'")
    else:
        print(f"Échec du téléchargement. Code de statut HTTP : {response.status_code}")

# URL du fichier à télécharger
url = "https://github.com/survet02/devLog/raw/main/autoencoder_v7.pth"

# Répertoire où extraire le contenu du fichier
target_dir = "./autoencoder_v7.pth"

# Télécharger le fichier et extraire son contenu
download_and_extract_zip(url, target_dir)


def download_and_extract_tar(url, target_dir):
    # Effectuer une requête GET sur l'URL du fichier
    response = requests.get(url, allow_redirects=True)

    # Vérifier si la requête a réussi (code de statut HTTP 200)
    if response.status_code == 200:
        # Créer un chemin de fichier temporaire pour enregistrer le fichier
        temp_file_path = "./temp.tar.gz"

        # Enregistrer le contenu de la réponse dans un fichier temporaire
        with open(temp_file_path, 'wb') as f:
            f.write(response.content)

        # Extraire le contenu du fichier temporaire
        with tarfile.open(temp_file_path, 'r:gz') as tar:
            tar.extractall(target_dir)

        # Supprimer le fichier temporaire
        os.remove(temp_file_path)

        print(f"Le fichier a été téléchargé et son contenu a été extrait avec succès dans '{target_dir}'")
    else:
        print(f"Échec du téléchargement. Code de statut HTTP : {response.status_code}")

# URL du fichier à télécharger
url = "https://github.com/survet02/devLog/raw/main/images_tgz/images.tar.gz"

# Répertoire où extraire le contenu du fichier TAR
target_dir = "./images"

# Télécharger le fichier et extraire son contenu
download_and_extract_tar(url, target_dir)


# -----------------------------------------
def download_github_folder(url, target_dir):
    # Make a GET request to the GitHub API to get the contents of the folder
    response = requests.get(url)
    contents = response.json()

    # Check if the request was successful (HTTP status code 200)
    if response.status_code == 200:
        # Create the target directory if it does not exist
        os.makedirs(target_dir, exist_ok=True)

        # Iterate over the contents of the folder
        for item in contents:
            # Get the download URL of the item
            download_url = item['download_url']

            # Determine the target file path
            file_path = os.path.join(target_dir, item['name'])

            # If the item is a file, download it
            if item['type'] == 'file':
                # Check if the file is a tar.gz
                if file_path.endswith('.tar.gz'):
                    # Call the function to download and extract tar.gz files
                    download_and_extract_tar(download_url, target_dir)
                else:
                    download_file(download_url, file_path)

        print(f"All files in {url} have been downloaded to {target_dir}")
    else:
        print(f"Failed to fetch folder contents. HTTP status code: {response.status_code}")

def download_file(url, target_path):
    # Make a GET request to download the file
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        # Write the content to a file
        with open(target_path, 'wb') as file:
            file.write(response.content)
        print(f"Downloaded file: {target_path}")
    else:
        print(f"Failed to download file: {url}")

# URL of the GitHub repository folder
repo_url = "https://api.github.com/repos/survet02/devLog/contents/datasets/celeba"

# Directory where the files will be downloaded
target_dir = "./datasets/celeba"

# Download the folder content
download_github_folder(repo_url, target_dir)

# URL of the GitHub repository folder
repo_url = "https://api.github.com/repos/survet02/devLog/contents/datasets/celeba/img_align_celeba_tgz"

# Directory where the files will be downloaded
target_dir = "./datasets/celeba/img_align_celeba"

# Download the folder content
download_github_folder(repo_url, target_dir)
