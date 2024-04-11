# faceguessr

**A software to create new faces from existing images using genetic algorithm**

# 👋 Introduction

This software uses CelebA faces images and use a genetic algorithm to modify them in order to create new faces.

# 📦 Installation

Install the faceguessr software using PyPI.

<details>
<summary>Install from PyPI</summary>
Installing the library with pip is the easiest way to get started with faceguessr.

```bash
python3 -m pip install --index-url https://test.pypi.org/simple/ --no-deps faceguessr
```

If you do not have all the required packages, use : 

```bash
pip install -r requirements.txt
```

Then go to the folder where faceguessr is : 

```bash
cd lib/python3.9/site-packages/faceguessr
```

```bash
python3 setup.py 
```



</details>

# 📦 How to use

<details>
<summary>Launch the software in your Terminal. You have to be in the faceguessr directory where the file application.py is.</summary>

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
- number of images to display (1 to 9).

3. Click on this icon and select at least two images that you want to use.

<p align="center">
  <img src="https://github.com/survet02/devLog/blob/main/images/select2.png" width="100">
</p>

4. Click on this icon to launch face modifications 


<p align="center">
  <img src="https://github.com/survet02/devLog/blob/main/images/round.png" width="100">
</p>

The selected images are displayed in the hsitory panel. The newly generated images are on top. Non-selected images are replaced by others. 
</details>
