# -*- coding: utf-8 -*-
"""Copy of deployment.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/12pqHdeQux-Gy1Q9k4PNVzZlasP54WB2a
"""

!pip install fastapi
!pip install uvicorn
!pip install transformers

!pip install torchtext==0.6.0

!uvicorn main:app --reload

