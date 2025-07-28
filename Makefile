# a makefile to setup the environment for the project
# call it with `make` to run the setup
# call it with `make clean` to clean up the environment
.PHONY: all clean

all: setup
	@echo "Environment setup complete."


setup:
	@echo "Setting up the environment..."
	@pip install --upgrade pip
	@echo "Pip upgraded."
	@pip install -r requirements.txt
	@echo "Environment setup done."
	@git submodule update --init --recursive
	@echo "Submodules updated."
	@pip install git+https://github.com/eduard626/CLIP.git
	@echo "CLIP installed from GitHub repository."
	@cd eTAM && pip install -e . && cd ..
	@echo "eTAM installed in editable mode."
	@cd eTAM/checkpoints && bash download_checkpoints.sh
	@echo "Checkpoints downloaded."
	

clean:
	@echo "Cleaning up..."
	@rm -rf eTAM/checkpoints/*.pt
	@rm -rf __pycache__
	@rm -rf *.pyc
	@echo "Cleanup done."
	