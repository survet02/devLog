# faceguessr

**A software to create new faces from existing images using genetic algorithm**

# 👋 Introduction

This software uses CelebA faces images and use a genetic algorithm to modify them in order to create new faces.

# 📦 Installation

Install the faceguessr software using PyPI.

<details>
<summary>Install from PyPI</summary>
First, you can create a virtual environment : 
  
```bash
python3 -m venv launch_faceguessr
```
```bash
source launch_faceguessr/bin/activate
```

Installing the library with pip is the easiest way to get started with faceguessr.

```bash
python3 -m pip install --index-url https://test.pypi.org/simple/ --no-deps faceguessr
```

Then go to the folder where faceguessr is : 

```bash
cd launch_faceguessr/lib/python3.9/site-packages/faceguessr
```

If you do not have all the required packages, use : 

```bash
pip install -r requirements.txt
```
Launch setup.py the first time you download faceguessr in order to download the remaining files needed to use faceguessr.

```bash
python3 setup_before_launch.py 
```



</details>

# 📦 How to use

<details>
<summary>Launch the software in your Terminal.</summary>

  
You have to be in the faceguessr directory where the file application.py is.


Type the following code and wait for the software to launch :
  
```bash
python3 application.py
```

</details>

<details>
<summary>Quick start</summary>
  
1. Click on this icon :
<p align="center">
  <img src="https://github.com/survet02/devLog/blob/main/images/open2.png" width="100">
</p>

2. Select :
- gender (Male, Female default = Female)
- hair color (Blond, Brown, default = both)
- skin tone (Pale, Dark, default = both)
- number of images to display (from 1 to 9)

3. Click on this icon to select images :

<p align="center">
  <img src="https://github.com/survet02/devLog/blob/main/images/select2.png" width="100">
</p>

4. Select at least two images that you want to use to create new faces from them.

5. Click on this icon to launch face modifications :

<p align="center">
  <img src="https://github.com/survet02/devLog/blob/main/images/round.png" width="100">
</p>

The selected images are displayed in the history panel. Newly generated images are surrounded by a golden rectangle. Non-selected images are replaced by new images from CelebA.

6. Click on this icon to select the final image that you are happy with :

<p align="center">
  <img src="https://github.com/survet02/devLog/blob/main/images/select2.png" width="100">
</p>

7. Click on this icon to see the image in a pop up frame :

<p align="center">
  <img src="https://github.com/survet02/devLog/blob/main/images/export_87484.png" width="100">
</p>

</details>
