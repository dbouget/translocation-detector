#!/bin/bash
isort --sl translocdet
black --line-length 120 translocdet
flake8 translocdet