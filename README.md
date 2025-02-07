# OpenProject CV Workshop

This repository contains materials for an online workshop focusing on image segmentation tasks with MediaPipe.

## Prerequisites

- Python: version 3.9 - 3.12
- PIP: version 20.3+
- Streamlit
- OpenCV, NumPy, etc.

## Usage

1. Clone the repository. Go to the folder you want to store the repo first. 
```
git clone https://github.com/jogong2718/openproject_CV_workshop.git
```

2. Install python version 3.12.6 from [here](https://www.python.org/downloads/release/python-3126/). You can also use pyenv or some other package manager for more experienced devs. For this project we are usingg mediapipe which doesn't support python 3.13+

Verify you downloaded the version of python by running 

Windows:
```
py -0
```

Unix OS:
```
which -a python3
```

3. Then create a venv with python 3.12

Windows:
```
py -3.12 -m venv venv
```

Unix OS:
```
python3.12 -m venv venv
```

Activate your venv:

Windows:
```
venv/scripts/activate
```

Unix OS:
```
source venv/bin/activate
```

3. Install required dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Launch the Streamlit app:
   ```
   streamlit run app.py
   ```

## Workshop Outline

- Introduction to computer vision concepts
- Hands-on segmentation and masking
- Working with Streamlit for live demos
