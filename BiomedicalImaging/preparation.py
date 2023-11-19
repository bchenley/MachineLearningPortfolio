{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "machine_shape": "hm",
      "gpuType": "V100",
      "authorship_tag": "ABX9TyMXEBcgInHiuY3bAFs8aM3R",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    },
    "accelerator": "GPU"
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/bchenley/MachineLearningPortfolio/blob/main/BiomedicalImaging/preparation.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "## Import modules\n",
        "import os, random, pickle\n",
        "\n",
        "## import access_images\n",
        "from src.access_images import access_images"
      ],
      "metadata": {
        "id": "hmYi-AjW8Hoy"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Get user input for a given prompt\n",
        "def get_user_input(prompt):\n",
        "    return input(prompt)\n",
        "\n",
        "# Validate user input\n",
        "def validate_input(input_text, prompt):\n",
        "    while True:\n",
        "        user_input = get_user_input(prompt)\n",
        "        confirm = input(f\"Confirm {input_text} (y/n): \").lower()\n",
        "        if confirm == 'y':\n",
        "            return user_input\n",
        "\n",
        "# Initialize variables\n",
        "task_path = \"\"\n",
        "save_dir = \"\"\n",
        "sample_size = 100\n",
        "train_size = 0\n",
        "test_size = 0\n",
        "register_images = False\n",
        "\n",
        "# '/content/drive/MyDrive/data/MSD/Task01_BrainTumour'\n",
        "\n",
        "# Get user input and validate\n",
        "task_path = validate_input(\"path to DICOM dataset\", \"Enter path to DICOM dataset:\")\n",
        "save_dir = validate_input(\"path to save images\", \"Enter path to save images:\")\n",
        "sample_size = int(validate_input(\"sample size (default = 100)\", \"Enter sample size (default = 100):\"))\n",
        "train_size = int(sample_size * float(validate_input(\"% training size [0, 1]\", \"Enter % training size [0, 1]:\")))\n",
        "test_size = sample_size - train_size\n",
        "register_images_input = validate_input(\"register Images to T1 (y/n)\", \"Register Images to T1 (y/n)?:\")\n",
        "register_images = True if register_images_input == 'y' else False\n",
        "\n",
        "# Display user input for confirmation\n",
        "print(f\"------------------------------------\")\n",
        "print(f\"Path to DICOM dataset = {task_path}\")\n",
        "print(f\"Path to save images = {save_dir}\")\n",
        "print(f\"Desired Sample size = {sample_size}\")\n",
        "print(f\"Training size = {train_size}\")\n",
        "print(f\"Test size = {test_size}\")\n",
        "print(f\"T1 Image Registration {'enabled.' if register_images else 'disabled.'}\")"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "OALXb_El9Ii0",
        "outputId": "7339994d-bb28-47a2-f577-01da02ae7212"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Enter path to DICOM dataset:/content/drive/MyDrive/data/MSD/Task01_BrainTumour\n",
            "Confirm path to DICOM dataset (y/n): y\n",
            "Enter path to save images:/content/drive/MyDrive/data/MSD\n",
            "Confirm path to save images (y/n): y\n",
            "Enter sample size (default = 100):100\n",
            "Confirm sample size (default = 100) (y/n): y\n",
            "Enter % training size [0, 1]:0.8\n",
            "Confirm % training size [0, 1] (y/n): y\n",
            "Register Images to T1 (y/n)?:y\n",
            "Confirm register Images to T1 (y/n) (y/n): y\n",
            "------------------------------------\n",
            "Path to DICOM dataset = /content/drive/MyDrive/data/MSD/Task01_BrainTumour\n",
            "Path to save images = /content/drive/MyDrive/data/MSD\n",
            "Desired Sample size = 100\n",
            "Training size = 80\n",
            "Test size = 20\n",
            "T1 Image Registration enabled.\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "## Get images, performing registration if desired\n",
        "\n",
        "images = access_images(task_path = task_path,\n",
        "                       sample_size = sample_size,\n",
        "                       register_images = register_images)\n"
      ],
      "metadata": {
        "id": "tHmP4h4ejglW"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "## Split images to train and test sets\n",
        "\n",
        "random.shuffle(images)\n",
        "\n",
        "train_images = images[:train_size]\n",
        "test_images = images[test_size:]"
      ],
      "metadata": {
        "id": "c3C-ga--7Okm"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "## Save training and test images\n",
        "\n",
        "train_path = f\"{save_dir}/train.pkl\"\n",
        "with open(train_path, \"wb\") as file:\n",
        "  pickle.dump(train_images, file)\n",
        "\n",
        "test_path = f\"{save_dir}/test.pkl\"\n",
        "with open(test_path, \"wb\") as file:\n",
        "  pickle.dump(test_images, file)"
      ],
      "metadata": {
        "id": "U8oYGDjhz-5Y"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}